"""Display metadata for upscalers while preserving their internal names."""

from __future__ import annotations

from collections.abc import Iterable


UPSCALER_DESCRIPTIONS_ZH_TW = {
    "RealESRGAN_x4plus": ("R-ESRGAN 4x+", "真實照片／通用影像"),
    "RealESRGAN_x4plus_anime_6B": ("R-ESRGAN 4x+ Anime6B", "動畫／插畫／二次元"),
    "4x-AnimeSharp": ("4x-AnimeSharp", "動畫／插畫／線條強化"),
    "4x-UltraSharp": ("4x-UltraSharp", "高銳利／細節強化"),
    "4x_foolhardy_Remacri": ("4x foolhardy Remacri", "通用／自然細節"),
}


def upscaler_display_name(name: str) -> str:
    display_name, description = UPSCALER_DESCRIPTIONS_ZH_TW.get(name, (name, None))
    return f"{display_name}（{description}）" if description else display_name


def upscaler_choices(upscalers: Iterable) -> list[tuple[str, str]]:
    """Return Gradio (display label, internal value) choices."""
    return [(upscaler_display_name(item.name), item.name) for item in upscalers]


def named_upscaler_choices(names: Iterable[str]) -> list[tuple[str, str]]:
    return [(upscaler_display_name(name), name) for name in names]
