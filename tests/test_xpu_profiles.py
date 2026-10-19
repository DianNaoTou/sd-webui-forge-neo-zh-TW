"""Portable A/B launcher contracts; no Windows/GPU or dependency installation."""
from importlib import metadata
import json
import os
from pathlib import Path
import tempfile
import unittest
import types
from unittest.mock import patch
from modules import dependency_utils as deps
from modules import xpu_profiles as profiles


class XPUProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'project with spaces'
        self.root.mkdir()
        (self.root / 'requirements.txt').write_text('Pillow==10.4.0\ncomfy-kitchen==0.2.38\n', encoding='utf-8')

    def test_venvs_settings_ports_and_cache_locations_are_distinct(self):
        base = {'PYTHONPATH':'other environment', 'COMMANDLINE_ARGS':'--nunchaku', 'FORGE_ORT_CUDA13':'1', 'PIP_TARGET':'other-site-packages'}
        envs = [profiles.environment(self.root, key, base) for key in ('official','kitchen')]
        self.assertNotEqual(envs[0]['VENV_DIR'], envs[1]['VENV_DIR'])
        for name in ('HF_HOME','TORCH_HOME','PIP_CACHE_DIR','GRADIO_TEMP_DIR','TEMP','TMP','UV_CACHE_DIR'):
            self.assertNotEqual(envs[0][name], envs[1][name])
        for env in envs:
            self.assertNotIn('PYTHONPATH', env)
            self.assertNotIn('COMMANDLINE_ARGS', env)
            self.assertNotIn('FORGE_ORT_CUDA13', env)
            self.assertNotIn('PIP_TARGET', env)
            self.assertIn('2.14.1+xpu', env['TORCH_COMMAND'])
            self.assertIn('0.29.1+xpu', env['TORCH_COMMAND'])
        self.assertEqual(base['FORGE_ORT_CUDA13'], '1')
        self.assertNotEqual(profiles.PROFILES['official'][1], profiles.PROFILES['kitchen'][1])

    def test_settings_are_first_run_copies_and_outputs_do_not_overlap(self):
        (self.root/'config.json').write_text(json.dumps({'seed':27,'outdir_samples':'old'}))
        states = []
        for key in ('official','kitchen'):
            _, state = profiles.paths(self.root,key)
            profiles.initialize_settings(self.root,state)
            states.append(state)
            self.assertEqual(json.loads((state/'config.json').read_text())['seed'],27)
        config = states[0]/'config.json'
        config.write_text(json.dumps({'seed':42}))
        profiles.initialize_settings(self.root,states[0])
        self.assertEqual(json.loads(config.read_text())['seed'],42)
        self.assertEqual(json.loads((states[1]/'config.json').read_text())['seed'],27)
        self.assertEqual(json.loads((self.root/'config.json').read_text())['outdir_samples'],'old')

    def test_flags_share_models_but_keep_settings_styles_and_port_owned(self):
        for key in ('official','kitchen'):
            _, state = profiles.paths(self.root,key)
            args = profiles.command_args(self.root,state,key,['--api'])
            self.assertEqual(args[args.index('--data-dir')+1], str(state))
            self.assertEqual(args[args.index('--model-ref')+1], str(self.root/'models'))
            self.assertEqual(args[args.index('--ui-settings-file')+1], str(state/'config.json'))
            self.assertEqual(args[args.index('--styles-file')+1], str(state/'styles.csv'))
            self.assertIn('--api',args)
            for bad in ('--port=7860','--ui-settings-file','--nunchaku','--use-ck-attention','--update-all-extensions'):
                with self.assertRaises(ValueError):
                    profiles.command_args(self.root,state,key,[bad])

    def test_attention_comparison_flags_keep_the_same_profile_and_environment(self):
        for key in ('official', 'kitchen'):
            _, state = profiles.paths(self.root,key)
            for mode in ('native', 'math', 'sliced'):
                args = profiles.command_args(self.root,state,key,['--xpu-sdpa',mode])
                self.assertEqual(args[-2:],['--xpu-sdpa',mode])
                self.assertEqual(args[args.index('--data-dir')+1],str(state))
                self.assertEqual(args[args.index('--port')+1],str(profiles.PROFILES[key][1]))

    def test_fork_constraints_do_not_get_overwritten_by_official_requirements(self):
        _, state = profiles.paths(self.root,'kitchen')
        requirement = profiles.requirements_for(self.root,state,'kitchen')
        self.assertIn('comfy-kitchen==0.2.37', requirement.read_text())
        self.assertIn('comfy-kitchen==0.2.38', (self.root/'requirements.txt').read_text())
        with patch.object(deps,'ROOT',self.root), patch.object(deps,'installed_torch_pins',return_value=['torch==2.14.1+xpu']), patch.dict(os.environ,{'FORGE_XPU_AB_PROFILE':'kitchen','REQS_FILE':str(requirement)},clear=True):
            command = deps.pip_constraint_args()
            self.assertIn(str(requirement),command)
            self.assertIn(str(state/'constraints-torch.txt'),command)
            with patch.dict(os.environ,{'REQS_FILE':str(self.root/'requirements.txt')}):
                with self.assertRaises(RuntimeError):
                    deps.pip_constraint_args()
        self.assertFalse((self.root/'tmp'/'constraints-torch.txt').exists())

    def test_torch_constraints_are_not_shared_with_nvidia_or_other_profile(self):
        paths = []
        for key in ('official','kitchen'):
            with patch.object(deps,'ROOT',self.root),patch.object(deps,'installed_torch_pins',return_value=['torch==2.14.1+xpu']),patch.dict(os.environ,{'FORGE_XPU_AB_PROFILE':key},clear=True):
                paths.append(deps.torch_constraint_file())
        self.assertNotEqual(paths[0],paths[1])
        with patch.object(deps,'ROOT',self.root),patch.object(deps,'installed_torch_pins',return_value=['torch==2.13.0+cu130']),patch.dict(os.environ,{},clear=True):
            default = deps.torch_constraint_file()
        self.assertEqual(default,self.root/'tmp'/'constraints-torch.txt')
        self.assertIn('+xpu',paths[0].read_text())

    def test_changed_official_kitchen_baseline_requires_review(self):
        (self.root/'requirements.txt').write_text('comfy-kitchen==0.3.0')
        with self.assertRaises(RuntimeError):
            profiles.requirements_for(self.root,self.root/'state','kitchen')

    def test_unavailable_native_kernel_stops_before_pip_or_fake_eager_run(self):
        with patch.object(metadata,'version',side_effect=metadata.PackageNotFoundError),patch.object(profiles,'pip') as pip:
            with self.assertRaisesRegex(RuntimeError,'No eager substitute'):
                profiles.kernel_preflight()
            pip.assert_not_called()
        with patch.object(profiles.shutil,'which',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'No packages were changed'):
                profiles.kernel_preflight(build=True)

    def test_native_identity_rejects_wrong_abi_target_or_missing_capability(self):
        good = [profiles.TORCH, profiles.TORCH, 'bmg','bmg',True,['quantize_int8_rowwise']]
        profiles.check_native_identity(*good)
        for index, value in [(1,'2.13.0+xpu'),(2,'ptl-h'),(3,''),(4,False),(5,[])]:
            bad = good.copy()
            bad[index] = value
            with self.assertRaises(RuntimeError):
                profiles.check_native_identity(*bad)

    def test_local_wheel_hash_and_platform_are_checked_before_install(self):
        wheel = self.root/'omni_xpu_kernel-0.2.0b2+torch214.bmg-cp313-cp313-linux_x86_64.whl'
        wheel.write_bytes(b'test')
        digest = profiles.hashlib.sha256(b'test').hexdigest()
        with patch.object(profiles,'pip') as pip:
            with self.assertRaisesRegex(RuntimeError,'sha256'):
                profiles.install_kernel(self.root,wheel,'wrong')
            with self.assertRaisesRegex(RuntimeError,'Windows x64'):
                profiles.install_kernel(self.root,wheel,digest)
            pip.assert_not_called()

    def test_new_source_checkout_pins_revision_without_false_dirty_error(self):
        with patch.object(profiles,'run') as run,patch.object(profiles.subprocess,'check_output',return_value='abc\n') as output:
            profiles.checkout(self.root,'test','https://github.com/owner/repo.git','abc')
            self.assertEqual(output.call_count,1)  # no index/worktree yet on --no-checkout clone
            self.assertIn('abc',run.call_args.args[0])

    def test_existing_dirty_checkout_is_left_untouched(self):
        path = self.root/'sources'/'test'
        path.mkdir(parents=True)
        with patch.object(profiles,'run') as run,patch.object(profiles.subprocess,'check_output',return_value=b'M file'):
            with self.assertRaisesRegex(RuntimeError,'local changes'):
                profiles.checkout(self.root,'test','https://github.com/owner/repo.git','abc')
            run.assert_not_called()

    def test_data_directory_junction_creation_and_existing_path_protection(self):
        source = self.root/'extensions'
        source.mkdir()
        state = self.root/'profile'
        state.mkdir()
        fake = types.SimpleNamespace(CreateJunction=unittest.mock.Mock())
        with patch.dict('sys.modules',{'_winapi':fake}):
            profiles.initialize_extension_link(self.root,state)
        fake.CreateJunction.assert_called_once_with(str(source.resolve()),str(state/'extensions'))
        (state/'extensions').mkdir()
        with self.assertRaisesRegex(RuntimeError,'leaving it untouched'):
            profiles.initialize_extension_link(self.root,state)
        self.assertTrue((state/'extensions').is_dir())
