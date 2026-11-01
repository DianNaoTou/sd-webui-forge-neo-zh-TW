"""Real file I/O rollback tests using tiny fake repos/models, never user data."""
import importlib.util
import json
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('selection_repair', ROOT / 'tools/repair_model_selection.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.root = self.parent / 'project'
        self.root.mkdir()
        self.models = self.root / 'models'
        self.models.mkdir()
        self.model = self.models / 'large-model.safetensors'
        self.model.write_bytes(b'pretend this is a 100GB model')
        self.before = b'# original\r\n'
        self.after = b'# patched\n'
        for name in repair.FILES:
            target = self.root / name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(self.before)
        (self.root / 'config.json').write_bytes(b'{"user":"settings"}')
        self.model_before = (self.model.read_bytes(), self.model.stat().st_mtime_ns)

        def git(root, *args):
            if args == ('branch', '--show-current'):
                return b'neo-test\n'
            if args == ('rev-parse', 'HEAD'):
                return repair.BASE.encode() + b'\n'
            if args[0] == 'show':
                if args[1].endswith(':tools/repair-model-selection.ps1'):
                    return b'# powershell fixture\n'
                return self.before if args[1].startswith(repair.BASE + ':') else self.after
            return b''
        self.real_git = repair.git
        self.git = patch.object(repair, 'git', side_effect=git)
        self.git.start()
        self.addCleanup(self.git.stop)

    def apply(self):
        return repair.apply(self.root, 'f' * 40, self.models, self.parent / 'backups')

    def assert_models_and_settings_unchanged(self):
        self.assertEqual((self.model.read_bytes(), self.model.stat().st_mtime_ns), self.model_before)
        self.assertEqual((self.root / 'config.json').read_bytes(), b'{"user":"settings"}')

    def test_apply_and_rollback_restore_exact_bytes(self):
        transaction = self.apply()
        for name in repair.FILES:
            self.assertEqual((self.root / name).read_bytes(), self.after)
        repair.rollback(self.root, transaction)
        for name in repair.FILES:
            self.assertEqual((self.root / name).read_bytes(), self.before)
        self.assertEqual(json.loads((transaction / 'manifest.json').read_text())['status'], 'rolled-back')
        self.assert_models_and_settings_unchanged()

    def test_local_edits_stop_before_any_write(self):
        changed = self.root / repair.FILES[-1]
        changed.write_bytes(b'# personal edits\n')
        with self.assertRaisesRegex(RuntimeError, 'Local edits'):
            self.apply()
        self.assertEqual(changed.read_bytes(), b'# personal edits\n')
        self.assertFalse((self.parent / 'backups').exists())
        self.assert_models_and_settings_unchanged()

    def test_partial_apply_failure_automatically_restores(self):
        write = repair.write_atomic
        calls = []

        def fail_second(path, data):
            if path == self.root / repair.FILES[1] and not calls:
                calls.append(True)
                raise OSError('simulated failed write')
            return write(path, data)

        with patch.object(repair, 'write_atomic', side_effect=fail_second):
            with self.assertRaisesRegex(OSError, 'simulated'):
                self.apply()
        for name in repair.FILES:
            self.assertEqual((self.root / name).read_bytes(), self.before)
        self.assert_models_and_settings_unchanged()

    def test_rollback_refuses_to_overwrite_subsequent_edits(self):
        transaction = self.apply()
        changed = self.root / repair.FILES[-1]
        changed.write_bytes(b'# new personal edit\n')
        with self.assertRaisesRegex(RuntimeError, 'changed after repair'):
            repair.rollback(self.root, transaction)
        self.assertEqual(changed.read_bytes(), b'# new personal edit\n')
        self.assertEqual((self.root / repair.FILES[0]).read_bytes(), self.after)

    def test_corrupt_backup_stops_rollback_before_writes(self):
        transaction = self.apply()
        (transaction / 'original' / repair.FILES[-1]).write_bytes(b'broken backup')
        with self.assertRaisesRegex(RuntimeError, 'Backup checksum'):
            repair.rollback(self.root, transaction)
        self.assertEqual((self.root / repair.FILES[0]).read_bytes(), self.after)

    def test_backup_inside_models_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Backups must be outside'):
            repair.apply(self.root, 'f' * 40, self.models, self.models / 'backup')
        self.assert_models_and_settings_unchanged()

    def test_code_path_redirect_into_models_is_rejected(self):
        source = self.root / 'modules'
        moved = self.models / 'modules'
        source.rename(moved)
        try:
            source.symlink_to(moved, target_is_directory=True)
        except OSError:
            self.skipTest('Symlink permission unavailable on this Windows runner')
        with self.assertRaisesRegex(RuntimeError, 'redirects'):
            self.apply()
        self.assertEqual((moved / 'sd_models.py').read_bytes(), self.before)
        self.assert_models_and_settings_unchanged()

    def test_real_model_junction_inventory_on_windows(self):
        import os
        if os.name != 'nt':
            self.skipTest('Native Windows junction test')
        import _winapi
        target = self.parent / 'actual-models'
        target.mkdir()
        model = target / 'weight.safetensors'
        model.write_bytes(b'untouched')
        link = self.models / 'VAE'
        _winapi.CreateJunction(str(target), str(link))
        try:
            transaction = self.apply()
            manifest = json.loads((transaction / 'manifest.json').read_text())
            self.assertTrue(any(Path(item['resolved']) == target for item in manifest['model_inventory']))
            repair.rollback(self.root, transaction)
            self.assertTrue(link.is_junction())
            self.assertEqual(model.read_bytes(), b'untouched')
        finally:
            link.rmdir()

    def test_healthy_environment_is_reused_without_pip_or_original_config_writes(self):
        transaction = self.parent / 'transaction'
        transaction.mkdir()
        python = self.root / 'venv-xpu-official' / 'Scripts' / 'python.exe'
        python.parent.mkdir(parents=True)
        python.write_bytes(b'fixture interpreter')
        with patch.object(repair.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
             patch.object(repair, 'run_logged') as run:
            repair.launch(self.root, transaction, self.models, False)
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertEqual(command[0], str(python))
        self.assertIn('--skip-prepare-environment', command)
        self.assertIn('--disable-all-extensions', command)
        self.assert_models_and_settings_unchanged()

    def test_rebuild_installs_only_into_new_environment(self):
        transaction = self.parent / 'transaction'
        transaction.mkdir()
        with patch.object(repair.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
             patch.object(repair, 'run_logged') as run, \
             patch.dict(repair.os.environ, {'PIP_TARGET': str(self.models)}):
            repair.launch(self.root, transaction, self.models, True)
        self.assertEqual(run.call_count, 4)
        self.assertEqual(run.call_args_list[0].args[0][-1], str(transaction / 'venv-xpu'))
        for call in run.call_args_list:
            self.assertNotIn('PIP_TARGET', call.kwargs['env'])
        for call in run.call_args_list[1:]:
            self.assertEqual(call.args[0][0], str(transaction / 'venv-xpu' / 'Scripts' / 'python.exe'))
        self.assertFalse((self.root / 'venv-xpu-official').exists())
        self.assert_models_and_settings_unchanged()

    def test_real_git_objects_apply_and_rollback_without_moving_branch(self):
        def git(*args):
            return subprocess.check_output(['git', '-C', str(self.root), *args], stderr=subprocess.STDOUT)
        git('init', '-b', 'neo-test')
        git('config', 'user.name', 'Repair Test')
        git('config', 'user.email', 'repair-test@example.invalid')
        git('config', 'core.autocrlf', 'false')
        tools = self.root / 'tools'
        tools.mkdir()
        (tools / 'repair-model-selection.ps1').write_bytes(b'# fixture\n')
        git('add', *repair.FILES, 'tools/repair-model-selection.ps1')
        git('commit', '-m', 'baseline fixture')
        base = git('rev-parse', 'HEAD').decode().strip()
        git('switch', '-c', 'repair-fixture')
        for name in repair.FILES:
            (self.root / name).write_bytes(self.after)
        git('add', *repair.FILES)
        git('commit', '-m', 'repair fixture')
        revision = git('rev-parse', 'HEAD').decode().strip()
        git('switch', 'neo-test')
        with patch.object(repair, 'git', self.real_git), patch.object(repair, 'BASE', base):
            transaction = repair.apply(self.root, revision, self.models, self.parent / 'backups')
            self.assertEqual(git('rev-parse', 'HEAD').decode().strip(), base)
            self.assertEqual((self.root / repair.FILES[0]).read_bytes(), self.after)
            repair.rollback(self.root, transaction)
            self.assertEqual(git('diff'), b'')
        self.assert_models_and_settings_unchanged()


if __name__ == '__main__':
    unittest.main()
