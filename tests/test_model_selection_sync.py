"""Exercise real selection/reload functions without importing the GPU runtime."""
import ast
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

ROOT = Path(os.environ.get('FORGE_REPAIR_TEST_ROOT', Path(__file__).resolve().parents[1]))


def functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in selected:
        node.decorator_list = []
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class Options:
    def __init__(self, **values):
        object.__setattr__(self, "data", values)
        object.__setattr__(self, "save", Mock())

    def __getattr__(self, key):
        try:
            return self.data[key]
        except KeyError:
            raise AttributeError(key) from None

    def set(self, key, value, **kwargs):
        self.data[key] = value


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vae = str(Path(self.temp.name) / "qwen_image_vae.safetensors")
        self.te = str(Path(self.temp.name) / "qwen_3_06b_base.safetensors")
        for path in (self.vae, self.te):
            Path(path).write_bytes(b"fixture only")
        self.opts = Options(sd_model_checkpoint="SDXL", forge_additional_modules=[],
                            forge_unet_storage_dtype="Automatic", forge_preset="xl",
                            sd_modules_dropdown_use_short=True)
        self.checkpoints = {k: NS(filename=k + ".safetensors") for k in ("SDXL", "ANIMA")}
        self.data = NS(forge_loading_parameters={}, forge_hash="", sd_model=NS(end_image=None))
        self.data.set_sd_model = lambda model: setattr(self.data, "sd_model", model)
        self.sd = ModuleType("modules.sd_models")
        self.sd.model_data = self.data
        self.sd.get_closet_checkpoint_match = self.checkpoints.get
        self.sd.select_checkpoint = lambda: self.checkpoints[self.opts.sd_model_checkpoint]
        self.shared = NS(opts=self.opts, config_filename="not-written.json")
        self.ns = functions("modules_forge/main_entry.py", {
            "validate_additional_modules", "refresh_model_loading_parameters", "checkpoint_change",
            "modules_change", "dtype_change", "_load_presets", "on_preset_change",
        }, dict(os=os, shared=self.shared, sd_models=self.sd,
                module_list={os.path.basename(x): x for x in (self.vae, self.te)}, module_map={},
                forge_unet_storage_dtype_options={"Automatic": (None, False), "test-dtype": ("FP8", False)},
                refresh_models=Mock(), dynamic_args=NS(), processing=NS(need_global_unload=False),
                logger=logging.getLogger("selection-test"), print_json=Mock(), torch=NS(float16="fp16", bfloat16="bf16"),
                gr=NS(update=lambda **kw: kw, skip=lambda: None), use_shift=lambda _: False,
                use_distill=lambda _: False, is_video=lambda _: 1))
        modules = ModuleType("modules")
        modules.sd_models = self.sd
        modules.shared = self.shared
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"modules": modules, "modules.sd_models": self.sd}).start()

    def apply(self, ckpt="SDXL", modules=None, dtype="Automatic", preset="xl"):
        self.ns["_load_presets"](ckpt, modules, dtype, preset)

    def test_round_trip_clears_qwen_at_loader_boundary(self):
        for ckpt, modules, preset in [("SDXL", [], "xl"), ("ANIMA", [self.vae, self.te], "anima"), ("SDXL", [], "xl")]:
            self.apply(ckpt, modules, preset=preset)
            self.assertIs(self.data.forge_loading_parameters["checkpoint_info"], self.checkpoints[ckpt])
            self.assertEqual(self.data.forge_loading_parameters["additional_modules"], sorted(modules))

    def test_same_checkpoint_empty_modules_replaces_stale_snapshot(self):
        self.opts.set("forge_additional_modules", [self.vae, self.te])
        self.ns["refresh_model_loading_parameters"]()
        self.apply(modules=[])
        self.assertEqual(self.data.forge_loading_parameters["additional_modules"], [])
        self.opts.save.assert_called_once()

    def test_already_empty_options_still_repair_stale_snapshot(self):
        self.data.forge_loading_parameters = {"additional_modules": ["gone/qwen.safetensors"]}
        self.apply(modules=[])
        self.assertEqual(self.data.forge_loading_parameters["additional_modules"], [])

    def test_none_and_dtype_only_same_checkpoint(self):
        self.apply(modules=None, dtype="test-dtype")
        self.assertEqual(self.data.forge_loading_parameters["additional_modules"], [])
        self.assertEqual(self.data.forge_loading_parameters["unet_storage_dtype"], "FP8")

    def test_missing_selected_module_raises_instead_of_silent_success(self):
        self.apply(modules=[])
        with self.assertRaises(FileNotFoundError):
            self.apply(modules=["gone.safetensors"])
        self.assertEqual(self.data.forge_loading_parameters, {})
        self.assertEqual(self.opts.save.call_count, 1)

    def test_missing_preset_selection_remains_visible(self):
        self.opts.set('forge_additional_modules_anima', ['gone.safetensors'])
        updates = self.ns['on_preset_change']('anima')
        self.assertEqual(updates[1]['value'], ['gone.safetensors'])
        self.assertIn('gone.safetensors', updates[1]['choices'])

    def test_unset_preset_checkpoint_uses_current_checkpoint(self):
        self.opts.set('forge_checkpoint_xl', None)
        self.assertEqual(self.ns['on_preset_change']('xl')[0]['value'], 'SDXL')

    def test_api_dtype_and_module_overrides_publish_one_tuple(self):
        main_entry = NS(**{name: self.ns[name] for name in
                          ('checkpoint_change', 'modules_change', 'dtype_change', 'refresh_model_loading_parameters')})
        forge = ModuleType('modules_forge')
        forge.main_entry = main_entry
        with patch.dict(sys.modules, {'modules_forge': forge}):
            ns = functions('modules/sysinfo.py', {'set_config'}, dict(Any=object))
            ns['set_config']({'forge_additional_modules': [self.vae], 'forge_unet_storage_dtype': 'test-dtype'}, save_config=False)
        self.assertEqual(self.data.forge_loading_parameters['additional_modules'], [self.vae])
        self.assertEqual(self.data.forge_loading_parameters['unet_storage_dtype'], 'FP8')

    def test_api_dtype_only_refreshes_same_checkpoint(self):
        self.apply(modules=[])
        main_entry = NS(**{name: self.ns[name] for name in
                          ('checkpoint_change', 'modules_change', 'dtype_change', 'refresh_model_loading_parameters')})
        forge = ModuleType('modules_forge')
        forge.main_entry = main_entry
        with patch.dict(sys.modules, {'modules_forge': forge}):
            ns = functions('modules/sysinfo.py', {'set_config'}, dict(Any=object))
            ns['set_config']({'forge_unet_storage_dtype': 'test-dtype'}, save_config=False)
        self.assertEqual(self.data.forge_loading_parameters['unet_storage_dtype'], 'FP8')

    def test_missing_global_path_not_downloaded_or_discarded(self):
        self.opts.set("forge_additional_modules", [str(Path(self.temp.name) / "missing.safetensors")])
        with self.assertRaises(FileNotFoundError):
            self.ns["refresh_model_loading_parameters"]()

    def test_same_checkpoint_callback_refreshes_stale_parameters(self):
        self.assertFalse(self.ns["checkpoint_change"]("SDXL", "xl"))
        self.assertEqual(self.data.forge_loading_parameters["additional_modules"], [])

    def test_empty_global_still_clears_old_preset_slot(self):
        self.opts.set('forge_additional_modules_xl', [self.vae, self.te])
        self.assertFalse(self.ns['modules_change']([], 'xl'))
        self.assertEqual(self.opts.forge_additional_modules_xl, [])

    def test_unchanged_checkpoint_still_updates_preset_slot(self):
        self.opts.set('forge_checkpoint_xl', 'ANIMA')
        self.assertFalse(self.ns['checkpoint_change']('SDXL', 'xl'))
        self.assertEqual(self.opts.forge_checkpoint_xl, 'SDXL')

    def test_invalid_checkpoint_is_not_silently_reused(self):
        with self.assertRaises(ValueError):
            self.ns["checkpoint_change"]("missing", "xl")

    def test_actual_reload_cache_and_snapshot(self):
        self.apply(modules=[])
        engine = NS(extra_generation_params={}, comments=[], forge_objects=NS(vae=NS(upscale_ratio=8)))
        ckpt = self.checkpoints["SDXL"]
        ckpt.calculate_shorthash = lambda: "test"
        ckpt.sha256 = "test"
        memory = NS(unload_all_models=Mock(), soft_empty_cache=Mock())
        received = []

        def load(path, additional_state_dicts):
            received.append((path, additional_state_dicts.copy()))
            # Publish the next UI choice during this load; this load's hash must
            # continue to describe the snapshot actually passed to the loader.
            self.data.forge_loading_parameters = dict(self.data.forge_loading_parameters, additional_modules=[self.vae])
            return engine

        reload_ns = functions("modules/sd_models.py", {"forge_model_reload"}, dict(
            model_data=self.data, memory_management=memory, gc=NS(collect=Mock()),
            Timer=lambda: NS(record=Mock(), summary=lambda: "fixture"), torch=NS(Tensor=object),
            dynamic_args=NS(), cmd_opts=NS(embeddings_dir="fixture"), forge_loader=Mock(side_effect=load),
            FakeInitialModel=lambda: NS(end_image=None), shared=self.shared,
            processing=NS(), script_callbacks=NS(model_loaded_callback=Mock()), print=lambda *a: None))
        expected_hash = str(self.data.forge_loading_parameters)
        self.assertTrue(reload_ns["forge_model_reload"]()[1])
        self.assertEqual(received, [("SDXL.safetensors", [])])
        self.assertEqual(self.data.forge_hash, expected_hash)
        # A newly published module selection cannot hit the previous cache.
        reload_ns["forge_loader"].side_effect = lambda *a, **k: engine
        self.assertTrue(reload_ns["forge_model_reload"]()[1])
        self.assertFalse(reload_ns["forge_model_reload"]()[1])

    def test_initial_ui_orders_choices_preset_and_commit(self):
        tree = ast.parse((ROOT / "modules_forge/main_entry.py").read_text())
        ui = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "forge_main_entry")
        self.assertTrue(any(isinstance(n, ast.Attribute) and n.attr == "then" and
                            isinstance(n.value, ast.Name) and n.value.id == "_event_initial_models" for n in ast.walk(ui)))
        calls = [n for n in ast.walk(ui) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertTrue(any(n.func.id == "_load_presets" for n in calls))


class ModelFileProtectionTests(unittest.TestCase):
    def test_loader_exception_keeps_original_and_prior_backup_bytes(self):
        from functools import wraps
        import warnings
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model.safetensors"
            backup = Path(str(model) + ".corrupted")
            model.write_bytes(b"original")
            backup.write_bytes(b"old backup")
            error = RuntimeError("not necessarily corruption")
            target = NS(__name__="fixture", load=Mock(side_effect=error))
            ns = functions("modules_forge/patch_basic.py", {"build_loaded"},
                           dict(os=os, warnings=warnings, wraps=wraps, display=Mock(), print=lambda *a: None))
            ns["build_loaded"](target, "load")
            with self.assertRaises(BufferError) as caught:
                target.load(str(model))
            self.assertIs(caught.exception.__cause__, error)
            self.assertEqual(model.read_bytes(), b"original")
            self.assertEqual(backup.read_bytes(), b"old backup")


if __name__ == "__main__":
    unittest.main()
