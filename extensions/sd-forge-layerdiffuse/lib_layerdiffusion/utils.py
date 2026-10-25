# Forge Neo zh-TW integration modified 2026-10-23; see UPSTREAM.md.
import numpy as np
from lib_layerdiffusion.enums import ResizeMode
from backend import memory_management
import cv2
import torch


def forge_clip_encode(sd_model, text):
    if text is None:
        return None

    cond = sd_model.get_learned_conditioning([text])
    return cond.to(memory_management.get_torch_device())


def rgba2rgbfp32(x):
    rgb = x[..., :3].astype(np.float32) / 255.0
    a = x[..., 3:4].astype(np.float32) / 255.0
    return 0.5 + (rgb - 0.5) * a


def to255unit8(x):
    return (x * 255.0).clip(0, 255).astype(np.uint8)


def safe_numpy(x):
    # A very safe method to make sure that Apple/Mac works
    y = x

    # below is very boring but do not change these. If you change these Apple or Mac may fail.
    y = y.copy()
    y = np.ascontiguousarray(y)
    y = y.copy()
    return y


def high_quality_resize(x, size):
    if x.shape[0] != size[1] or x.shape[1] != size[0]:
        if (size[0] * size[1]) < (x.shape[0] * x.shape[1]):
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LANCZOS4

        y = cv2.resize(x, size, interpolation=interpolation)
    else:
        y = x
    return y


def crop_and_resize_image(detected_map, resize_mode, h, w):
    if resize_mode == ResizeMode.RESIZE:
        detected_map = high_quality_resize(detected_map, (w, h))
        detected_map = safe_numpy(detected_map)
        return detected_map

    old_h, old_w, _ = detected_map.shape
    old_w = float(old_w)
    old_h = float(old_h)
    k0 = float(h) / old_h
    k1 = float(w) / old_w

    safeint = lambda x: int(np.round(x))

    if resize_mode == ResizeMode.RESIZE_AND_FILL:
        k = min(k0, k1)
        borders = np.concatenate([detected_map[0, :, :], detected_map[-1, :, :], detected_map[:, 0, :], detected_map[:, -1, :]], axis=0)
        high_quality_border_color = np.median(borders, axis=0).astype(detected_map.dtype)
        high_quality_background = np.tile(high_quality_border_color[None, None], [h, w, 1])
        detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (h - new_h) // 2)
        pad_w = max(0, (w - new_w) // 2)
        high_quality_background[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = detected_map
        detected_map = high_quality_background
        detected_map = safe_numpy(detected_map)
        return detected_map
    else:
        k = max(k0, k1)
        detected_map = high_quality_resize(detected_map, (safeint(old_w * k), safeint(old_h * k)))
        new_h, new_w, _ = detected_map.shape
        pad_h = max(0, (new_h - h) // 2)
        pad_w = max(0, (new_w - w) // 2)
        detected_map = detected_map[pad_h:pad_h+h, pad_w:pad_w+w]
        detected_map = safe_numpy(detected_map)
        return detected_map


def pytorch_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y


def load_frozen_patcher_compat(unet, filename, state_dict, strength):
    from modules_forge.packages.comfy import weight_adapter

    lora_adapter_cls = next((a for a in weight_adapter.adapters if a.name == "lora"), None)
    if lora_adapter_cls is None:
        raise RuntimeError("目前 Forge Neo 缺少 LoRAAdapter，請更新測試分支。")
    if not state_dict:
        raise ValueError("LayerDiffuse 權重內容為空。")
    model_state = unet.model.state_dict()
    patch_dict = {}
    for key, weight in state_dict.items():
        fields = key.split("::")
        if len(fields) != 3:
            raise ValueError(f"不支援的圖層權重鍵：{key}")
        model_key, patch_type, weight_index = fields
        if model_key not in model_state:
            raise ValueError(f"圖層權重與模型不相容，找不到：{model_key}")
        if patch_type not in ("lora", "diff", "set"):
            raise ValueError(f"不支援的圖層修補類型：{patch_type}")
        index = int(weight_index)
        if index < 0 or index >= 16:
            raise ValueError(f"圖層權重索引超出範圍：{key}")
        if model_key in patch_dict and patch_type not in patch_dict[model_key]:
            raise ValueError(f"同一模型鍵包含多種修補，無法安全合併：{model_key}")
        values = patch_dict.setdefault(model_key, {}).setdefault(patch_type, [None] * 16)
        values[index] = weight

    patch_flat = {}
    for model_key, groups in patch_dict.items():
        for patch_type, values in groups.items():
            if patch_type == "lora":
                up, down, alpha, mid = values[:4]
                if up is None or down is None or any(v is not None for v in values[4:]):
                    raise ValueError(f"圖層 LoRA 權重格式不完整或不支援：{model_key}")
                patch_flat[model_key] = lora_adapter_cls(
                    loaded_keys=set(), weights=(up, down, alpha, mid, None, None))
            else:
                if values[0] is None or any(v is not None for v in values[1:]):
                    raise ValueError(f"圖層差異權重格式不支援：{model_key}")
                weight = values[0]
                original_shape = tuple(model_state[model_key].shape)
                shape = tuple(weight.shape)
                if shape != original_shape:
                    if (patch_type != "diff" or len(shape) != 4 or
                            shape[0] != original_shape[0] or shape[2:] != original_shape[2:] or
                            shape[1] <= original_shape[1]):
                        raise ValueError(f"圖層權重形狀不相容：{model_key}")
                patch_flat[model_key] = (("diff", (weight, {"pad_weight": shape != original_shape}))
                                         if patch_type == "diff" else ("set", (weight,)))
    applied = unet.add_patches(filename=filename, patches=patch_flat,
                               strength_patch=float(strength), strength_model=1.0)
    if set(applied) != set(patch_flat):
        raise ValueError("部分 LayerDiffuse 權重未套用，已停止生成。")


def unload_model_clones_compat(model):
    """
    Replacement for memory_management.unload_model_clones(), removed in
    Forge Neo. Unloads any currently loaded model that wraps the same
    underlying model instance. Crucially, we must call detach() (which
    internally calls unpatch_model()) BEFORE removing it from Forge's
    tracking list, otherwise any object patch (like AttentionSharingUnit)
    applied by a previous LayerDiffuse SD1.5 generation stays permanently
    stuck on the shared model.
    """
    from backend import memory_management

    to_unload = [m for m in memory_management.current_loaded_models if model.is_clone(m.model)]
    for m in to_unload:
        m.model.detach(unpatch_all=True)
        memory_management.unload_model(m.model)
