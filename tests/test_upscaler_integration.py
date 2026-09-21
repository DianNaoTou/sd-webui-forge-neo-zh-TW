"""CPU-only tests for shared upscaler installation and UI metadata."""

import hashlib
import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from modules import upscaler_models
from modules.upscaler_ui import named_upscaler_choices, upscaler_choices, upscaler_display_name


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class UpscalerModelInstallTests(unittest.TestCase):
    def model(self, body=b"model"):
        return upscaler_models.UpscalerModel(
            filename="test.pth",
            url="https://example.test/test.pth",
            sha256=hashlib.sha256(body).hexdigest(),
        )

    def test_atomic_verified_download(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "test.pth"
            with patch.object(upscaler_models, "urlopen", return_value=FakeResponse(b"model")):
                upscaler_models.download_upscaler_model(self.model(), target)
            self.assertEqual(target.read_bytes(), b"model")
            self.assertFalse(target.with_name("test.pth.partial").exists())

    def test_hash_failure_cleans_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "test.pth"
            with patch.object(upscaler_models, "urlopen", return_value=FakeResponse(b"corrupt")):
                with self.assertRaises(ValueError):
                    upscaler_models.download_upscaler_model(self.model(), target)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name("test.pth.partial").exists())

    def test_existing_user_file_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "test.pth"
            target.write_bytes(b"custom")
            with patch.object(upscaler_models, "DEFAULT_UPSCALER_MODELS", (self.model(),)), patch.object(upscaler_models, "urlopen") as request:
                result = upscaler_models.ensure_upscaler_models(folder)
            self.assertEqual(result.installed, [])
            self.assertEqual(result.skipped, [target])
            self.assertEqual(target.read_bytes(), b"custom")
            request.assert_not_called()

    def test_one_failure_does_not_block_other_models(self):
        good = self.model(b"good")
        bad = upscaler_models.UpscalerModel("bad.pth", "https://example.test/bad", hashlib.sha256(b"wanted").hexdigest())
        with tempfile.TemporaryDirectory() as folder, patch.object(upscaler_models, "DEFAULT_UPSCALER_MODELS", (bad, good)), patch.object(
            upscaler_models, "urlopen", side_effect=[FakeResponse(b"wrong"), FakeResponse(b"good")]
        ):
            result = upscaler_models.ensure_upscaler_models(folder)
            self.assertEqual([path.name for path in result.installed], ["test.pth"])
            self.assertEqual([path.name for path in result.failed], ["bad.pth"])

    def test_manifest_contains_the_documented_shared_set(self):
        self.assertEqual(
            {model.filename for model in upscaler_models.DEFAULT_UPSCALER_MODELS},
            {
                "RealESRGAN_x4plus.pth",
                "RealESRGAN_x4plus_anime_6B.pth",
                "4x-AnimeSharp.pth",
                "4x-UltraSharp.pth",
                "4x_foolhardy_Remacri.pth",
            },
        )
        for model in upscaler_models.DEFAULT_UPSCALER_MODELS:
            self.assertEqual(len(model.sha256), 64)


class UpscalerMetadataTests(unittest.TestCase):
    def test_known_model_has_helpful_label_and_original_value(self):
        item = types.SimpleNamespace(name="RealESRGAN_x4plus_anime_6B")
        self.assertEqual(
            upscaler_choices([item]),
            [("R-ESRGAN 4x+ Anime6B（動畫／插畫／二次元）", "RealESRGAN_x4plus_anime_6B")],
        )

    def test_unknown_and_latent_names_remain_compatible(self):
        self.assertEqual(upscaler_display_name("Custom_4x"), "Custom_4x")
        self.assertEqual(named_upscaler_choices(["Latent"]), [("Latent", "Latent")])

    def test_every_downloaded_model_has_a_description(self):
        for model in upscaler_models.DEFAULT_UPSCALER_MODELS:
            name = Path(model.filename).stem
            self.assertNotEqual(upscaler_display_name(name), name)


if __name__ == "__main__":
    unittest.main()
