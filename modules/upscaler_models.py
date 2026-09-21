"""Shared first-run downloads for the bundled image upscalers.

The weights live in the normal ESRGAN model directory, so Forge's built-in
upscaler, SD Upscale, and Ultimate SD Upscale all discover the same files.
This module intentionally uses only the Python standard library because it is
called while the launch environment is still being prepared.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class UpscalerModel:
    filename: str
    url: str
    sha256: str


@dataclass
class UpscalerInstallResult:
    installed: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    failed: list[Path] = field(default_factory=list)


# Real-ESRGAN weights come from the author's official GitHub releases. The other
# entries preserve the model set previously offered by download-upscalers.bat.
DEFAULT_UPSCALER_MODELS = (
    UpscalerModel(
        filename="RealESRGAN_x4plus.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    ),
    UpscalerModel(
        filename="RealESRGAN_x4plus_anime_6B.pth",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        sha256="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
    ),
    UpscalerModel(
        filename="4x-AnimeSharp.pth",
        url="https://huggingface.co/sudolink/upscale_models/resolve/main/4x-AnimeSharp.pth",
        sha256="e7a7de2dafd7331c1992862bbbcd9e9712a9f9f8e6303f0aaa59b4341d359bab",
    ),
    UpscalerModel(
        filename="4x-UltraSharp.pth",
        url="https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth",
        sha256="a5812231fc936b42af08a5edba784195495d303d5b3248c24489ef0c4021fe01",
    ),
    UpscalerModel(
        filename="4x_foolhardy_Remacri.pth",
        url="https://huggingface.co/sudolink/upscale_models/resolve/main/4x_foolhardy_Remacri.pth",
        sha256="e1a73bd89c2da1ae494774746398689048b5a892bd9653e146713f9df8bca86a",
    ),
)


def download_upscaler_model(model: UpscalerModel, destination: Path) -> None:
    """Atomically download and verify one model without replacing an existing file."""
    if destination.exists():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial")
    partial.unlink(missing_ok=True)
    request = Request(model.url, headers={"User-Agent": "Forge-Neo-zh-TW/upscaler-installer"})

    try:
        digest = hashlib.sha256()
        with urlopen(request, timeout=60) as response, partial.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
                digest.update(chunk)

        actual_hash = digest.hexdigest()
        if actual_hash != model.sha256:
            raise ValueError(f"SHA-256 mismatch: expected {model.sha256}, got {actual_hash}")

        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def ensure_upscaler_models(model_dir: str | os.PathLike[str]) -> UpscalerInstallResult:
    """Ensure the recommended shared model set exists; failures are non-fatal."""
    directory = Path(model_dir)
    result = UpscalerInstallResult()

    for model in DEFAULT_UPSCALER_MODELS:
        destination = directory / model.filename
        if destination.exists():
            # Existing files may be user-managed variants; never overwrite them.
            result.skipped.append(destination)
            continue

        print(f"[Upscalers] Downloading recommended model: {model.filename}")
        try:
            download_upscaler_model(model, destination)
        except Exception as error:
            print(f"[Upscalers] Warning: failed to download {model.filename}: {type(error).__name__}: {error}")
            result.failed.append(destination)
        else:
            result.installed.append(destination)

    return result


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    result = ensure_upscaler_models(project_root / "models" / "ESRGAN")

    print(
        f"[Upscalers] Ready: {len(result.installed)} downloaded, "
        f"{len(result.skipped)} already present, {len(result.failed)} failed."
    )
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
