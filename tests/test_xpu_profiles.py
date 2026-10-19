"""Official XPU environment isolation and retired-profile rejection contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
from modules import dependency_utils as deps
from modules import xpu_profiles as profiles

ROOT = Path(__file__).resolve().parents[1]


class XPUProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'project with spaces'
        self.root.mkdir()
        (self.root / 'requirements.txt').write_text('Pillow==10.4.0\ncomfy-kitchen==0.2.38\n', encoding='utf-8')

    def test_official_environment_is_isolated_and_keeps_existing_paths(self):
        base = {'PYTHONPATH':'other environment', 'COMMANDLINE_ARGS':'--nunchaku', 'FORGE_ORT_CUDA13':'1', 'PIP_TARGET':'other-site-packages'}
        env = profiles.environment(self.root,'official',base)
        self.assertEqual(profiles.PROFILES,{'official':('venv-xpu-official',7862)})
        venv,state = profiles.paths(self.root,'official')
        self.assertEqual(Path(env['VENV_DIR']),venv)
        self.assertEqual(state,self.root/'runtime'/'xpu-ab'/'official')
        for name in ('HF_HOME','TORCH_HOME','PIP_CACHE_DIR','GRADIO_TEMP_DIR','TEMP','TMP','UV_CACHE_DIR'):
            self.assertTrue(Path(env[name]).is_relative_to(state))
        for name in ('PYTHONPATH','COMMANDLINE_ARGS','FORGE_ORT_CUDA13','PIP_TARGET'):
            self.assertNotIn(name,env)
        self.assertIn('2.14.1+xpu',env['TORCH_COMMAND'])
        self.assertIn('0.29.1+xpu',env['TORCH_COMMAND'])
        self.assertEqual(base['FORGE_ORT_CUDA13'],'1')

    def test_settings_are_first_run_copies_not_overwrites(self):
        (self.root/'config.json').write_text(json.dumps({'seed':27,'outdir_samples':'old'}))
        _,state = profiles.paths(self.root,'official')
        profiles.initialize_settings(self.root,state)
        config=state/'config.json'
        self.assertEqual(json.loads(config.read_text())['seed'],27)
        self.assertTrue(json.loads(config.read_text())['outdir_samples'].startswith(str(state)))
        config.write_text(json.dumps({'seed':42}))
        profiles.initialize_settings(self.root,state)
        self.assertEqual(json.loads(config.read_text())['seed'],42)
        self.assertEqual(json.loads((self.root/'config.json').read_text())['outdir_samples'],'old')

    def test_fresh_settings_have_current_version_marker(self):
        _, state = profiles.paths(self.root, 'official')
        profiles.initialize_settings(self.root, state)
        data = json.loads((state/'config.json').read_text())
        self.assertEqual(data['VERSION_UID'], profiles.settings_version_uid)
        self.assertEqual(data['localization'], 'zh_Hant')

    def test_copied_old_settings_require_acknowledgement_and_backup(self):
        original = {'seed': 27, 'VERSION_UID': 'older', 'localization': 'zh_Hant'}
        (self.root/'config.json').write_text(json.dumps(original))
        _, state = profiles.paths(self.root, 'official')
        profiles.initialize_settings(self.root, state)
        config = state/'config.json'
        before = config.read_bytes()
        self.assertEqual(json.loads(before)['VERSION_UID'], 'older')
        profiles.acknowledge_settings_version(self.root, config)
        after = json.loads(config.read_text())
        self.assertEqual(after, {**json.loads(before), 'VERSION_UID': profiles.settings_version_uid})
        backups = list(state.glob('config-before-version-*.json'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), before)
        profiles.acknowledge_settings_version(self.root, config)
        self.assertEqual(list(state.glob('config-before-version-*.json')), backups)
        self.assertEqual(json.loads((self.root/'config.json').read_text()), original)
        with self.assertRaises(ValueError):
            profiles.acknowledge_settings_version(self.root, self.root/'config.json')

    def test_flags_keep_models_settings_styles_port_and_sdpa_choices(self):
        _,state=profiles.paths(self.root,'official')
        for mode in ('native','math','sliced'):
            args=profiles.command_args(self.root,state,'official',['--xpu-sdpa',mode])
            self.assertEqual(args[-2:],['--xpu-sdpa',mode])
            for flag,value in [('--data-dir',state),('--model-ref',self.root/'models'),
                               ('--ui-settings-file',state/'config.json'),('--styles-file',state/'styles.csv'),('--port','7862')]:
                self.assertEqual(args[args.index(flag)+1],str(value))
        for bad in ('--port=7860','--ui-settings-file','--nunchaku','--use-ck-attention',
                    '--update-all-extensions','--build-native','--kernel-wheel=x.whl','--kernel-sha256'):
            with self.assertRaises(ValueError):
                profiles.command_args(self.root,state,'official',[bad])

    def test_official_requirements_preserve_core_comfy_kitchen(self):
        _,state=profiles.paths(self.root,'official')
        self.assertEqual(profiles.requirements_for(self.root,state,'official'),self.root/'requirements.txt')
        self.assertIn('comfy-kitchen==0.2.38',(self.root/'requirements.txt').read_text())
        self.assertFalse(state.exists())

    def test_retired_profile_rejected_without_environment_or_cache_mutations(self):
        old=self.root/'runtime'/'xpu-ab'/'kitchen'
        old.mkdir(parents=True)
        sentinel=old/'keep-user-settings.json'
        sentinel.write_text('user settings')
        for name,call in [('paths',lambda:profiles.paths(self.root,'kitchen')),
                          ('environment',lambda:profiles.environment(self.root,'kitchen',{})),
                          ('requirements',lambda:profiles.requirements_for(self.root,old,'kitchen')),
                          ('verify',lambda:profiles.verify_runtime('kitchen')),
                          ('arguments',lambda:profiles.command_args(self.root,old,'kitchen',[]))]:
            with self.subTest(name=name),self.assertRaises(ValueError):
                call()
        self.assertEqual(list(old.iterdir()),[sentinel])
        self.assertEqual(sentinel.read_text(),'user settings')
        self.assertFalse((self.root/'venv-xpu-official').exists())

    def test_dependency_installs_reject_stale_retired_profile_environment(self):
        with patch.object(deps,'ROOT',self.root),patch.dict(os.environ,{'FORGE_XPU_AB_PROFILE':'kitchen'},clear=True):
            for call in (deps.pip_constraint_args,deps.torch_constraint_file):
                with self.assertRaisesRegex(RuntimeError,'only official'):
                    call()
        self.assertFalse((self.root/'tmp').exists())

    def test_official_constraints_do_not_touch_cuda_or_retired_state(self):
        with patch.object(deps,'ROOT',self.root),patch.object(deps,'installed_torch_pins',return_value=['torch==2.14.1+xpu']),patch.dict(os.environ,{'FORGE_XPU_AB_PROFILE':'official'},clear=True):
            official=deps.torch_constraint_file()
            self.assertIn(str(self.root/'requirements.txt'),deps.pip_constraint_args())
        with patch.object(deps,'ROOT',self.root),patch.object(deps,'installed_torch_pins',return_value=['torch==2.13.0+cu130']),patch.dict(os.environ,{},clear=True):
            cuda=deps.torch_constraint_file()
        self.assertNotEqual(official,cuda)
        self.assertIn('+xpu',official.read_text())
        self.assertIn('+cu130',cuda.read_text())
        self.assertFalse((self.root/'runtime'/'xpu-ab'/'kitchen').exists())

    def test_launcher_rejects_retired_profile_before_platform_checks(self):
        result=subprocess.run([sys.executable,str(ROOT/'tools/launch_xpu_profile.py'),'kitchen'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('invalid choice',result.stderr)
        for name in ('webui-user-intel-xpu-kitchen.bat','build-intel-xpu-kitchen.bat'):
            self.assertFalse((ROOT/name).exists())

    def test_runtime_version_fallback_and_custom_backend_rejection(self):
        torch=types.SimpleNamespace(__version__=profiles.TORCH,xpu=types.SimpleNamespace(is_available=lambda:True,get_device_name=lambda index:'Intel B580'))
        kitchen=types.SimpleNamespace(list_backends=lambda:{'eager':{'available':True}})
        with patch.dict('sys.modules',{'torch':torch,'comfy_kitchen':kitchen}):
            self.assertEqual(profiles.verify_runtime('official')['kitchen'],'unknown')
            kitchen.list_backends=lambda:{'xpu':{'available':True}}
            with self.assertRaisesRegex(RuntimeError,'custom XPU'):
                profiles.verify_runtime('official')

    def test_data_directory_junction_creation_and_existing_path_protection(self):
        source=self.root/'extensions'
        source.mkdir()
        state=self.root/'profile'
        state.mkdir()
        fake=types.SimpleNamespace(CreateJunction=Mock())
        with patch.dict('sys.modules',{'_winapi':fake}):
            profiles.initialize_extension_link(self.root,state)
        fake.CreateJunction.assert_called_once_with(str(source.resolve()),str(state/'extensions'))
        (state/'extensions').mkdir()
        with self.assertRaisesRegex(RuntimeError,'leaving it untouched'):
            profiles.initialize_extension_link(self.root,state)
        self.assertTrue((state/'extensions').is_dir())
