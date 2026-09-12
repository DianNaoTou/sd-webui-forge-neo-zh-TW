"""Installer for the Image2Prompt extension (Forge Neo).

Ensures the required libraries are present. Set the environment variable
IMG2PROMPT_SKIP_INSTALL=1 to skip all checks (e.g. offline usage).
"""
import os
import launch

if os.environ.get("IMG2PROMPT_SKIP_INSTALL", "0") != "1":

    # accelerate is needed for device_map / low-vram loading
    if not launch.is_installed("accelerate"):
        launch.run_pip("install accelerate", "accelerate (Image2Prompt)")

    # transformers >= 4.45 is required for Qwen2-VL support.
    # Forge Neo usually ships a recent version already, so we only
    # upgrade when strictly necessary.
    try:
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("4.45.0"):
            launch.run_pip(
                "install -U \"transformers>=4.45.0\"",
                "transformers >= 4.45 (Image2Prompt / Qwen2-VL)",
            )
    except Exception:
        if not launch.is_installed("transformers"):
            launch.run_pip(
                "install \"transformers>=4.45.0\"",
                "transformers (Image2Prompt)",
            )

    # Florence-2 needs einops and timm
    for pkg in ("einops", "timm"):
        if not launch.is_installed(pkg):
            launch.run_pip(f"install {pkg}", f"{pkg} (Image2Prompt / Florence-2)")
