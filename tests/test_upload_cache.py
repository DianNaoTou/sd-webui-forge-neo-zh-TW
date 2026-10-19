"""Gradio HTTP upload regressions, including simulated Windows rename fallback."""
import importlib.util
from contextlib import contextmanager
from io import BytesIO
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
try:
    import gradio as gr
    from fastapi.testclient import TestClient
    from PIL import Image, PngImagePlugin
except ImportError:
    gr = None


@unittest.skipIf(gr is None, 'requires pinned Gradio/FastAPI UI dependencies')
class UploadCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shared = types.SimpleNamespace(opts=types.SimpleNamespace(temp_dir=''), demo=None)
        import modules
        spec = importlib.util.spec_from_file_location('upload_cache_test', ROOT/'modules/ui_tempdir.py')
        self.mod = importlib.util.module_from_spec(spec)
        with patch.object(modules, 'shared', self.shared, create=True), patch.dict('sys.modules', {'modules.shared': self.shared}):
            spec.loader.exec_module(self.mod)
        buffer = BytesIO()
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text('parameters', 'a cat\nSteps: 28, Seed: 42, Size: 64x64')
        Image.new('RGB', (64, 64), 'red').save(buffer, format='PNG', pnginfo=metadata)
        self.png = buffer.getvalue()

    @contextmanager
    def locked_preview(self, path):
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                          wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
            kernel.CreateFileW.restype = wintypes.HANDLE
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle.restype = wintypes.BOOL
            # Permit parallel readers, reject truncation/replacement while served.
            handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
            if handle == wintypes.HANDLE(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                yield
            finally:
                kernel.CloseHandle(handle)
        else:
            with patch.object(gr.route_utils.shutil, 'move', side_effect=PermissionError('simulated preview sharing violation')):
                yield

    def client(self, name):
        cache = self.root/name
        cache.mkdir()
        with patch.dict(os.environ, {'GRADIO_TEMP_DIR': str(cache)}):
            with gr.Blocks(analytics_enabled=False) as demo:
                gr.Image(type='pil')
            demo.max_file_size = None
            app = gr.routes.App.create_app(demo)
        return TestClient(app), cache

    def test_duplicate_upload_keeps_live_file_and_http_metadata_for_both_platform_caches(self):
        rename = os.rename
        def windows_rename(source, dest):
            if Path(dest).exists():
                raise FileExistsError('simulated Windows rename to existing file')
            return rename(source, dest)
        paths = []
        clients = []
        for name in ('cuda-cache', 'xpu-official-cache'):
            with self.subTest(cache=name):
                client, cache = self.client(name)
                with patch.object(gr.routes.os, 'rename', windows_rename):
                    first = client.post('/upload', files={'files': ('example.png', self.png, 'image/png')})
                    self.assertEqual(first.status_code, 200)
                    path = Path(first.json()[0])
                    self.assertTrue(path.is_relative_to(cache))
                    before = path.stat().st_mtime_ns
                    # Old Gradio fallback rewrites an existing file: simulate a Windows
                    # sharing violation while its preview is open in another request.
                    with self.locked_preview(path):
                        with self.assertRaises(PermissionError):
                            client.post('/upload', files={'files': ('example.png', self.png, 'image/png')})
                        with patch.object(gr.routes, 'move_uploaded_files_to_cache', self.mod.move_uploaded_files_to_cache):
                            again = client.post('/upload', files={'files': ('example.png', self.png, 'image/png')})
                    self.assertEqual(again.json(), first.json())
                    self.assertEqual(path.stat().st_mtime_ns, before)
                    response = client.get('/file='+str(path))
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.content, self.png)
                    self.assertIn('Steps: 28', Image.open(BytesIO(response.content)).info['parameters'])
                    paths.append(path)
                    clients.append(client)
        self.assertNotEqual(paths[0], paths[1])
        self.assertEqual(clients[0].get('/file='+str(paths[1])).status_code, 403)
        self.assertEqual(clients[1].get('/file='+str(paths[0])).status_code, 403)

    def test_background_copy_publishes_only_after_writer_closes(self):
        source = self.root/'source.png'
        source.write_bytes(self.png)
        dest = self.root/'copy.png'
        replace = os.replace
        handles = []
        factory = tempfile.NamedTemporaryFile
        def tracked(*a, **k):
            handle = factory(*a, **k)
            handles.append(handle)  # prevent GC from hiding an unclosed writer
            return handle
        def publish(staged, destination):
            self.assertTrue(all(h.closed for h in handles))
            self.assertFalse(dest.exists())
            self.assertEqual(Path(staged).read_bytes(), self.png)
            return replace(staged, destination)
        with patch.object(self.mod.tempfile, 'NamedTemporaryFile', tracked), patch.object(self.mod.os, 'replace', publish):
            self.mod.move_uploaded_files_to_cache([source], [dest])
        self.assertEqual(dest.read_bytes(), self.png)
        self.assertFalse(source.exists())
        self.assertFalse(list(self.root.glob('*.upload')))

    def test_http_upload_uses_safe_fallback_when_rename_cannot_move(self):
        client, cache = self.client('fallback-cache')
        with patch.object(gr.routes.os, 'rename', side_effect=OSError('cross-device move')), \
             patch.object(gr.routes, 'move_uploaded_files_to_cache', self.mod.move_uploaded_files_to_cache):
            result = client.post('/upload', files={'files': ('example.png', self.png, 'image/png')})
        path = Path(result.json()[0])
        self.assertTrue(path.is_relative_to(cache))
        self.assertEqual(client.get('/file='+str(path)).content, self.png)

    def test_persistent_permission_error_is_not_hidden_and_source_is_kept(self):
        source, dest = self.root/'source.png', self.root/'dest.png'
        source.write_bytes(self.png)
        dest.write_bytes(b'existing different data')
        with patch.object(self.mod.os, 'replace', side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError):
                self.mod.move_uploaded_files_to_cache([source], [dest])
        self.assertEqual(source.read_bytes(), self.png)
        self.assertEqual(dest.read_bytes(), b'existing different data')
        self.assertFalse(list(self.root.glob('*.upload')))

    def test_concurrent_identical_publish_does_not_rewrite_live_file(self):
        source, dest = self.root/'source.png', self.root/'dest.png'
        source.write_bytes(self.png)
        def competing_publish(*args):
            dest.write_bytes(self.png)
            raise PermissionError('another uploader published first')
        with patch.object(self.mod.os, 'replace', competing_publish):
            self.mod.move_uploaded_files_to_cache([source], [dest])
        self.assertEqual(dest.read_bytes(), self.png)
        self.assertFalse(source.exists())
        self.assertFalse(list(self.root.glob('*.upload')))

    def test_pil_preview_writer_closes_before_return_and_keeps_metadata(self):
        handles = []
        factory = tempfile.NamedTemporaryFile
        def tracked(*a, **k):
            handle = factory(*a, **k)
            handles.append(handle)
            return handle
        image = Image.open(BytesIO(self.png))
        with patch.object(self.mod.tempfile, 'NamedTemporaryFile', tracked):
            path = self.mod.save_pil_to_file(image, cache_dir=self.root)
        self.assertTrue(all(h.closed for h in handles))
        self.assertEqual(Image.open(path).info['parameters'], image.info['parameters'])

    def test_installer_changes_only_gradio_upload_fallback_and_existing_hooks(self):
        with patch.object(gr.processing_utils, 'save_pil_to_cache'), patch.object(gr.processing_utils, 'async_move_files_to_cache'), patch.object(gr.routes, 'move_uploaded_files_to_cache'):
            self.mod.install_ui_tempdir_override()
            self.assertIs(gr.routes.move_uploaded_files_to_cache, self.mod.move_uploaded_files_to_cache)
            self.assertIs(gr.processing_utils.save_pil_to_cache, self.mod.save_pil_to_file)
