"""Validate the main zh_Hant localization file.

Checks JSON syntax, duplicate keys, string-only values, and that strings
added for Forge Neo 2.30 (DeGrid, Qwen 2.1, Batch Cond/Uncond, dropdown
options) stay translated.
"""

import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ZH_HANT = ROOT / "localizations" / "zh_Hant.json"

REQUIRED_2_30 = (
    "DeGrid",
    "Threshold",
    "0.0 for Auto",
    "[Qwen 2.1] Enable Reference",
    "Batch Cond/Uncond",
    "(do both positive and negative denoising in the same batch ; increase speed & VRAM usage)",
    "(less effective on macOS)",
    "Show filenames without folder in the VAE / Text Encoder dropdown",
    "(if disabled, modules under subdirectories will be listed like sdxl/clip-l-anime.safetensors)",
)


def _load_pairs():
    text = ZH_HANT.read_text(encoding="utf-8")
    return json.loads(text, object_pairs_hook=lambda pairs: pairs)


class ZhHantLocalizationTests(unittest.TestCase):
    def test_valid_json_without_duplicate_keys(self):
        pairs = _load_pairs()
        seen, dupes = set(), []
        for key, _ in pairs:
            if key in seen:
                dupes.append(key)
            seen.add(key)
        self.assertEqual(dupes, [], f"Duplicate keys: {dupes}")

    def test_values_are_non_empty_strings(self):
        bad = [k for k, v in _load_pairs() if not isinstance(v, str) or not v.strip()]
        self.assertEqual(bad, [])

    def test_forge_2_30_strings_translated(self):
        data = dict(_load_pairs())
        missing = [k for k in REQUIRED_2_30 if k not in data]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
