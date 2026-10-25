"""Version warning contracts without importing launcher dependency installation."""
import ast
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
from modules import xpu_profiles as profiles
from modules_forge import forge_version

ROOT = Path(__file__).resolve().parents[1]


class SettingsVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        _, self.state = profiles.paths(self.root, 'official')
        self.state.mkdir(parents=True)
        self.config = self.state/'config.json'
        tree = ast.parse((ROOT/'modules/launch_utils.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify_version')
        self.input = Mock(return_value='')
        self.namespace = dict(os=os, json=json, args=types.SimpleNamespace(ui_settings_file=str(self.config)),
                              VERSION_UID=forge_version.settings_version_uid, script_path=str(self.root), input=self.input)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<verify_version>', 'exec'), self.namespace)

    def verify(self, profile='official'):
        with patch.dict(os.environ, {'FORGE_XPU_AB_PROFILE': profile}), patch('sys.stdout', new_callable=io.StringIO) as output, patch.object(os, 'system'):
            self.namespace['verify_version']()
        return output.getvalue()

    def test_fresh_official_config_never_prompts(self):
        profiles.initialize_settings(self.root, self.state)
        self.assertEqual(self.verify(), '')
        self.input.assert_not_called()

    def test_existing_unmarked_official_config_is_preserved_until_explicit_continue(self):
        data = {'localization': 'zh_Hant', 'never_oom': True, 'custom': ['keep me']}
        self.config.write_text(json.dumps(data), encoding='utf-8')
        def acknowledge(_):
            self.assertEqual(json.loads(self.config.read_text()), data)
            self.assertEqual(list(self.state.glob('config-before-version-*.json')), [])
        self.input.side_effect = acknowledge
        warning = self.verify()
        self.assertIn('Settings version is missing', warning)
        self.assertNotIn('clean reinstall', warning)
        self.assertIn('backs up official Intel', warning)
        self.assertEqual(json.loads(self.config.read_text()), {**data, 'VERSION_UID': forge_version.settings_version_uid})
        self.assertEqual(len(list(self.state.glob('config-before-version-*.json'))), 1)
        self.input.reset_mock()
        self.assertEqual(self.verify(), '')
        self.input.assert_not_called()

    def test_cancelled_warning_does_not_mark_settings(self):
        self.config.write_text('{"custom": 27}')
        self.input.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.verify()
        self.assertEqual(json.loads(self.config.read_text()), {'custom': 27})
        self.assertEqual(list(self.state.glob('config-before-version-*.json')), [])

    def test_nvidia_existing_settings_are_not_marked_or_backed_up(self):
        self.config = self.root/'config.json'
        self.config.write_text('{"custom": 27}')
        self.namespace['args'].ui_settings_file = str(self.config)
        self.verify(profile='')
        self.input.assert_called_once()
        self.assertEqual(json.loads(self.config.read_text()), {'custom': 27})
        self.assertEqual(list(self.state.glob('config-before-version-*.json')), [])

    def test_shared_uid_retains_existing_nvidia_settings_format(self):
        self.assertEqual(forge_version.settings_version_uid, 'PY313')
        source = (ROOT/'modules/launch_utils.py').read_text()
        self.assertIn('VERSION_UID: Final[str] = forge_version.settings_version_uid', source)
