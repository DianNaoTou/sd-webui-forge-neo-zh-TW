"""Request-local Forge Neo integration; no package installation or global model flags.

Modified by DianNaoTou integration, 2026-10-23, based on LayerDiffuse Fix 3.
"""
import functools


def fail_closed(function):
    @functools.wraps(function)
    def wrapped(self, p, *args, **kwargs):
        from modules import scripts
        try:
            previous = getattr(p, '_layerdiffuse_error', None)
            if previous:
                raise RuntimeError(previous)
            return function(self, p, *args, **kwargs)
        except Exception as error:
            p._layerdiffuse_error = str(error)
            raise scripts.ScriptAbortError(f'LayerDiffuse：{error}') from error
    return wrapped


def validate_request(p, method, fg_image, bg_image, blend_image):
    """Fail before downloading weights or patching a model."""
    family = 'sd15' if method.name.endswith('_SD15') else 'sdxl'
    flag = 'is_sd1' if family == 'sd15' else 'is_sdxl'
    if not getattr(p.sd_model, flag, False):
        raise ValueError('所選方法與目前模型不符；請選擇對應的 SD1.5 或 SDXL 方法。')
    if getattr(p.sd_model, 'is_inpaint', False):
        raise ValueError('測試版僅支援一般 SD1.5／SDXL 模型，請勿使用專用 Inpainting 模型。')
    frames = 3 if method.name == 'JOINT_SD15' else (2 if method.name in ('FG_TO_BG_SD15', 'BG_TO_FG_SD15') else 1)
    if not isinstance(p.batch_size, int) or p.batch_size < 1 or p.batch_size % frames:
        raise ValueError(f'此方法的批次大小必須是 {frames} 的正整數倍。')
    required = {
        'FG_TO_BG_SD15': (fg_image,), 'BG_TO_FG_SD15': (bg_image,),
        'FG_TO_BLEND': (fg_image,), 'BG_TO_BLEND': (bg_image,),
        'FG_BLEND_TO_BG': (fg_image, blend_image),
        'BG_BLEND_TO_FG': (bg_image, blend_image),
    }
    if any(image is None for image in required.get(method.name, ())):
        raise ValueError('此方法需要的前景、背景或合成圖片尚未提供。')
    for image in (fg_image, bg_image, blend_image):
        if image is not None and (getattr(image, 'ndim', 0) != 3 or image.shape[2] != 4):
            raise ValueError('輸入圖片必須是 RGBA 格式，請重新載入圖片。')
    for script in getattr(getattr(p, 'scripts', None), 'alwayson_scripts', ()):
        # Use ADetailer's own enable predicate, including API-provided arguments.
        if hasattr(script, 'is_ad_enabled') and not getattr(p, '_ad_disabled', False):
            args = p.script_args[script.args_from:script.args_to]
            if script.is_ad_enabled(*args):
                raise ValueError('LayerDiffuse 測試版暫不支援與 ADetailer 同時啟用；請先關閉其中一項。')
    if family == 'sd15':
        unet = p.sd_model.forge_objects.unet
        options = unet.model_options.get('transformer_options', {})
        if options.get('patches_replace') or options.get('patches'):
            raise ValueError('SD1.5 Attention 模式暫不支援與其他 Attention 掛接插件共同使用。')
        if method.name != 'FG_ONLY_ATTN_SD15' and getattr(p, 'image_mask', None) is not None:
            raise ValueError('SD1.5 多圖模式暫不支援遮罩局部重繪。')
    for name in ('add_patches', 'add_object_patch', 'add_conditioning_modifier', 'detach'):
        if not callable(getattr(p.sd_model.forge_objects.unet, name, None)):
            raise ValueError(f'目前 Forge Neo 缺少必要介面 {name}，請更新 neo-test。')
    return family


def attach_concat(conditions, tensor, condition_class):
    """Preserve other plugins' conditions without mutating their dictionaries."""
    if conditions is None:
        return None
    result = []
    for item in conditions:
        entry = dict(item)
        entry['model_conds'] = dict(item['model_conds'])
        entry['model_conds']['c_concat'] = condition_class(tensor)
        result.append(entry)
    return result


def match_batch(tensor, reference):
    if tuple(tensor.shape[2:]) != tuple(reference.shape[2:]):
        raise ValueError('圖層條件尺寸與採樣尺寸不符，已停止生成。')
    if tensor.shape[0] == reference.shape[0]:
        return tensor.to(reference)
    if tensor.shape[0] == 1:
        return tensor.repeat(reference.shape[0], 1, 1, 1).to(reference)
    raise ValueError('圖層條件批次與採樣批次不符，已停止生成。')


def get_transparent_codec(p, family, role, loader):
    cache = getattr(p, '_layerdiffuse_codecs', None)
    if cache is None:
        cache = p._layerdiffuse_codecs = {}
    key = (family, role)
    if key not in cache:
        cache[key] = loader()
    return cache[key]


def record_patch(p, target, attribute, original, patched):
    records = getattr(p, '_layerdiffuse_patches', None)
    if records is None:
        records = p._layerdiffuse_patches = []
    records.append((target, attribute, original, patched))


def cleanup_request(p):
    from backend import memory_management
    from lib_layerdiffusion.utils import unload_model_clones_compat
    failures = []
    # Drop the request's references even when cleanup of one model fails.
    records = p.__dict__.pop('_layerdiffuse_patches', [])
    codecs = p.__dict__.pop('_layerdiffuse_codecs', {})
    for target, attribute, original, patched in reversed(records):
        try:
            patcher = patched.patcher if attribute == 'vae' else patched
            unload_model_clones_compat(patcher)
        except Exception as error:
            failures.append(error)
        finally:
            if getattr(target, attribute) is patched:
                setattr(target, attribute, original)
    for codec in codecs.values():
        try:
            codec.model.detach(unpatch_all=True)
            memory_management.unload_model(codec.model)
        except Exception as error:
            failures.append(error)
    p.__dict__.pop('_layerdiffuse_error', None)
    if failures:
        raise RuntimeError('LayerDiffuse 清理失敗：' + '; '.join(map(str, failures)))
