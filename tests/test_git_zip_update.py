"""Real Git/ZIP update tests; no model, Python environment or network required."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

from tools.package_git_zip import build, git, git_env, assert_data_policy


class GitZipUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.origin = self.base / 'origin'
        self.origin.mkdir()
        git(self.origin, 'init', '--initial-branch', 'neo-test')
        git(self.origin, 'config', 'user.email', 'fixture@example.invalid')
        git(self.origin, 'config', 'user.name', 'Fixture')
        self.write('.gitignore', '/config.json\n/models/\n/outputs/\n/runtime/\n/venv-nvidia/\n/extensions/custom/\n')
        self.write('app.txt', 'version one\n')
        self.write('webui-user.bat', 'rem shipped launcher\n')
        self.commit('initial')
        self.initial = git(self.origin, 'rev-parse', 'HEAD')

    def write(self, name, text):
        path = self.origin / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    def commit(self, message):
        git(self.origin, 'add', '-A')
        git(self.origin, 'commit', '-m', message)

    def package(self, branch='neo-test', commit=None):
        output = self.base / ('package-' + branch + '.zip')
        summary = build(self.origin.as_uri(), branch, commit or self.initial, output)
        extracted = self.base / ('extracted-' + branch)
        with zipfile.ZipFile(output) as archive:
            archive.extractall(extracted)
        return extracted / ('sd-webui-forge-neo-zh-TW-' + branch), summary, output

    def pull(self, root):
        return subprocess.run(['git', '-C', str(root), 'pull'], env=git_env(),
                              capture_output=True, text=True)

    def test_exact_commit_portable_clean_and_checksum(self):
        self.write('app.txt', 'newer tip\n')
        self.commit('advance before package')
        root, summary, output = self.package()
        self.assertEqual(git(root, 'rev-parse', 'HEAD'), self.initial)
        self.assertEqual(git(root, 'symbolic-ref', '--short', 'HEAD'), 'neo-test')
        self.assertEqual(git(root, 'rev-parse', '--abbrev-ref', '@{u}'), 'origin/neo-test')
        self.assertEqual(git(root, 'status', '--porcelain'), '')
        self.assertTrue((root / '.git/shallow').is_file())
        self.assertFalse((root / '.git/objects/info/alternates').exists())
        self.assertFalse((root / '.git/hooks').exists())
        git(root, 'fsck', '--full')
        self.assertEqual(summary['sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertIn(summary['sha256'], output.with_suffix('.zip.sha256').read_text())
        self.assertGreater(summary['git_compressed_bytes'], 0)

    def test_plain_pull_twice_retains_user_data(self):
        root, _, _ = self.package()
        moved = self.base / '測試 folder'
        root.rename(moved)
        root = moved
        data = {'config.json': b'{"api_key":"private"}', 'models/example.safetensors': b'weight',
                'outputs/one.png': b'image', 'runtime/python/existing': b'python',
                'venv-nvidia/keep': b'environment', 'extensions/custom/keep': b'plugin'}
        for name, value in data.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        for version in ('two', 'three'):
            self.write('app.txt', version + '\n')
            self.commit(version)
            result = self.pull(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / 'app.txt').read_text(), version + '\n')
            for name, value in data.items():
                self.assertEqual((root / name).read_bytes(), value)

    def test_modified_tracked_launcher_stops_without_overwrite(self):
        root, _, _ = self.package()
        (root / 'webui-user.bat').write_text('rem personal parameters\n')
        self.write('webui-user.bat', 'rem updated launcher\n')
        self.commit('change launcher')
        result = self.pull(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((root / 'webui-user.bat').read_text(), 'rem personal parameters\n')
        self.assertEqual(git(root, 'rev-parse', 'HEAD'), self.initial)
        self.assertEqual(git(root, 'stash', 'list'), '')

    def test_ignored_collision_aborts_instead_of_overwriting(self):
        root, _, _ = self.package()
        (root / 'config.json').write_text('private settings\n')
        self.write('config.json', 'bad upstream settings\n')
        git(self.origin, 'add', '-f', 'config.json')
        self.commit('accidental tracked config')
        result = self.pull(root)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((root / 'config.json').read_text(), 'private settings\n')
        self.assertEqual(git(root, 'rev-parse', 'HEAD'), self.initial)

    def test_untracked_collision_keeps_personal_file(self):
        root, _, _ = self.package()
        (root / 'new-feature.txt').write_text('personal file\n')
        self.write('new-feature.txt', 'upstream feature\n')
        self.commit('new tracked file')
        self.assertNotEqual(self.pull(root).returncode, 0)
        self.assertEqual((root / 'new-feature.txt').read_text(), 'personal file\n')
        self.assertEqual(git(root, 'rev-parse', 'HEAD'), self.initial)

    def test_divergence_aborts_without_merge_or_stash(self):
        root, _, _ = self.package()
        git(root, 'config', 'user.email', 'fixture@example.invalid')
        git(root, 'config', 'user.name', 'Fixture')
        (root / 'local.txt').write_text('personal commit\n')
        git(root, 'add', 'local.txt')
        git(root, 'commit', '-m', 'local')
        local = git(root, 'rev-parse', 'HEAD')
        self.write('app.txt', 'remote change\n')
        self.commit('remote')
        self.assertNotEqual(self.pull(root).returncode, 0)
        self.assertEqual(git(root, 'rev-parse', 'HEAD'), local)
        self.assertFalse((root / '.git/MERGE_HEAD').exists())

    def test_formal_branch_tracks_formal_and_rejects_data(self):
        git(self.origin, 'branch', 'neo')
        root, _, _ = self.package('neo')
        self.assertEqual(git(root, 'rev-parse', '--abbrev-ref', '@{u}'), 'origin/neo')
        self.write('config.json', 'must never ship\n')
        git(self.origin, 'add', '-f', 'config.json')
        with self.assertRaisesRegex(ValueError, 'tracked user data'):
            assert_data_policy(self.origin)

    def test_package_never_overwrites_previous_output(self):
        _, _, output = self.package()
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            build(self.origin.as_uri(), 'neo-test', self.initial, output)
        self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
