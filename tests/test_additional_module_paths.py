"""Regression tests for moved Forge Neo additional-module paths.

Extract the standalone resolver without importing Gradio or the GPU runtime.
"""
import ast
import logging
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "modules_forge" / "main_entry.py").read_text(encoding="utf-8")
tree = ast.parse(source)
resolver = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_additional_modules")
module = ast.Module(body=[resolver], type_ignores=[])
namespace = {"os": os, "logger": logging.getLogger("path-test")}
exec(compile(ast.fix_missing_locations(module), "<resolver>", "exec"), namespace)
resolve = namespace["validate_additional_modules"]


class AdditionalModulePathTests(unittest.TestCase):
    def test_existing_path_unchanged_and_no_download(self):
        with tempfile.TemporaryDirectory() as temp:
            existing = Path(temp) / "qwen_image_vae.safetensors"
            existing.write_bytes(b"test")
            self.assertEqual(resolve([str(existing)], {}), [str(existing)])

    def test_moved_path_recovered_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as temp:
            current = Path(temp) / "models" / "VAE" / "qwen_image_vae.safetensors"
            current.parent.mkdir(parents=True)
            current.write_bytes(b"test")
            old = str(Path(temp) / "old" / current.name)
            self.assertEqual(resolve([old], {current.name: str(current)}), [str(current)])
            self.assertEqual(old, str(Path(temp) / "old" / current.name))

    def test_missing_or_ambiguous_path_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "a" / "same.safetensors"
            second = Path(temp) / "b" / "same.safetensors"
            for path in [first, second]:
                path.parent.mkdir()
                path.write_bytes(b"test")
            old = str(Path(temp) / "gone" / "same.safetensors")
            self.assertEqual(resolve([old], {str(first): str(first), str(second): str(second)}), [])
            self.assertEqual(resolve([str(Path(temp) / "missing.safetensors")], {}), [])


if __name__ == "__main__":
    unittest.main()
