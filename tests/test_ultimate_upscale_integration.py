"""Static integration checks that do not require a GPU or Forge startup."""
import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "extensions-builtin/ultimate-upscale-for-automatic1111"
SCRIPT = EXTENSION / "scripts/ultimate-upscale.py"


class UltimateUpscaleIntegrationTests(unittest.TestCase):
    def test_vendored_extension_is_complete_and_pinned(self):
        self.assertTrue(SCRIPT.is_file())
        self.assertTrue((EXTENSION / "LICENSE").is_file())
        tracking = (EXTENSION / "UPSTREAM.md").read_text(encoding="utf-8")
        self.assertIn("2322caa480535b1011a1f9c18126d85ea444f146", tracking)

    def test_extension_has_no_installer_or_extra_requirements(self):
        self.assertFalse((EXTENSION / "install.py").exists())
        self.assertFalse((EXTENSION / "requirements.txt").exists())

    def test_script_is_img2img_only_and_has_no_cuda_hardcoding(self):
        source = SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        script_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Script"
        )
        methods = {node.name: node for node in script_class.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({"title", "show", "ui", "run"}.issubset(methods))
        self.assertEqual(ast.unparse(methods["show"].body[0].value), "is_img2img")
        lowered = source.lower()
        for cuda_only in (".cuda(", 'device="cuda"', "device='cuda'", "torch.float16", "torch.float8"):
            self.assertNotIn(cuda_only, lowered)

    def test_native_sd_upscale_remains_available(self):
        source = (ROOT / "scripts/sd_upscale.py").read_text(encoding="utf-8")
        self.assertIn('return "SD Upscale"', source)

    def test_traditional_chinese_ui_strings(self):
        localization = json.loads((ROOT / "localizations/zh_Hant.json").read_text(encoding="utf-8"))
        localization.update(json.loads((EXTENSION / "localizations/zh_Hant.json").read_text(encoding="utf-8")))
        expected = {
            "Ultimate SD upscale": "終極 SD 放大",
            "Upscaler": "放大演算法",
            "Tile width": "分塊寬度",
            "Tile height": "分塊高度",
            "Padding": "邊界填充",
            "Mask blur": "遮罩模糊",
            "Seams fix": "接縫修復",
            "Denoise": "重繪幅度",
            "Target size type": "目標尺寸類型",
            "Redraw options:": "重繪選項：",
            "Save options:": "儲存選項：",
            "Band pass": "帶通處理",
            "Half tile offset pass": "半分塊偏移處理",
            "Half tile offset pass + intersections": "半分塊偏移處理＋交會處",
        }
        for source, translated in expected.items():
            self.assertEqual(localization.get(source), translated)


if __name__ == "__main__":
    unittest.main()
