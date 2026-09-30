import importlib.util
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
EMPHASIS_PATH = ROOT / "backend" / "text_processing" / "emphasis.py"


def _load_emphasis():
    """Load emphasis.py directly; stub torch when it is not installed."""
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False
        stub = types.ModuleType("torch")
        stub.Tensor = object
        sys.modules.setdefault("torch", stub)
    # Register lightweight parent packages so the module's relative import
    # (from .parsing import ...) resolves without importing all of backend.
    for name, path in (("backend", ROOT / "backend"), ("backend.text_processing", EMPHASIS_PATH.parent)):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(path)]
            sys.modules[name] = pkg
    spec = importlib.util.spec_from_file_location("backend.text_processing.emphasis", EMPHASIS_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        # Do not leak the torch stub into other tests.
        if not has_torch and sys.modules.get("torch") is stub:
            del sys.modules["torch"]
    return module, has_torch


emphasis, HAS_TORCH = _load_emphasis()


class EmphasisModeTests(unittest.TestCase):
    def test_none_and_ignore_return_input_unchanged(self):
        z = object()
        multipliers = object()
        for cls in (emphasis.EmphasisNone, emphasis.EmphasisIgnore):
            with self.subTest(mode=cls.name):
                self.assertIs(cls()(z, multipliers), z)

    def test_options_include_expected_modes(self):
        names = {cls.name for cls in emphasis.options}
        self.assertTrue({"None", "Ignore", "Original"}.issubset(names))

    def test_uses_emphasis_ignores_break(self):
        self.assertFalse(emphasis.uses_emphasis("a BREAK b"))
        self.assertTrue(emphasis.uses_emphasis("(a:1.2) BREAK b"))

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_original_modes_return_tensor_of_same_shape(self):
        import torch

        z = torch.ones(1, 3, 4)
        multipliers = torch.tensor([[1.0, 1.2, 0.8]])
        for cls in (emphasis.EmphasisOriginal, emphasis.EmphasisOriginalNoNorm):
            with self.subTest(mode=cls.name):
                out = cls()(z.clone(), multipliers)
                self.assertIsInstance(out, torch.Tensor)
                self.assertEqual(tuple(out.shape), (1, 3, 4))


if __name__ == "__main__":
    unittest.main()
