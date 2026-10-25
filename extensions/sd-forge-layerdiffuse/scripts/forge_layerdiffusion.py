# Forge Neo zh-TW integration modified 2026-10-23; see UPSTREAM.md.
import gradio as gr
import os
import torch
import numpy as np

from modules import extensions

# Two LayerDiffuse installations share Python module names; refuse ambiguous imports.
_extension_root = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))
for _extension in extensions.active():
    if (os.path.realpath(_extension.path) != _extension_root and
            os.path.isfile(os.path.join(_extension.path, 'scripts', 'forge_layerdiffusion.py'))):
        raise ImportError('偵測到其他已啟用的 LayerDiffuse，請先在擴充功能頁停用舊版並重新啟動。')

from lib_layerdiffusion.utils import rgba2rgbfp32, to255unit8, crop_and_resize_image, forge_clip_encode, load_frozen_patcher_compat
from modules import scripts
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img
from lib_layerdiffusion.enums import ResizeMode
from enum import Enum
from modules.paths import models_path
from backend import utils, memory_management

from lib_layerdiffusion.models import TransparentVAEDecoder, TransparentVAEEncoder
from backend.sampling.sampling_function import sampling_prepare
from lib_layerdiffusion.downloads import load_file_from_url, REVISION
from lib_layerdiffusion.attention_sharing import AttentionSharingPatcher
from modules_forge.forge_canvas.canvas import ForgeCanvas
from modules import images
from PIL import Image, ImageOps
from backend.sampling.condition import Condition
from lib_layerdiffusion.compat import (fail_closed, validate_request, attach_concat, match_batch,
    get_transparent_codec, record_patch, cleanup_request)


def is_model_loaded(model):
    return any(model == m.model for m in memory_management.current_loaded_models)


layer_model_root = os.path.join(models_path, 'layer_model')
# Directory creation and downloads are deferred until the feature is enabled.


METHOD_LABELS = {
    'FG_ONLY_ATTN_SD15': '(SD1.5) Only Generate Transparent Image (Attention Injection)',
    'FG_TO_BG_SD15': '(SD1.5) From Foreground to Background (need batch size 2)',
    'BG_TO_FG_SD15': '(SD1.5) From Background to Foreground (need batch size 2)',
    'JOINT_SD15': '(SD1.5) Generate Everything Together (need batch size 3)',
    'FG_ONLY_ATTN': '(SDXL) Only Generate Transparent Image (Attention Injection)',
    'FG_ONLY_CONV': '(SDXL) Only Generate Transparent Image (Conv Injection)',
    'FG_TO_BLEND': '(SDXL) From Foreground to Blending',
    'FG_BLEND_TO_BG': '(SDXL) From Foreground and Blending to Background',
    'BG_TO_BLEND': '(SDXL) From Background to Blending',
    'BG_BLEND_TO_FG': '(SDXL) From Background and Blending to Foreground',
}
RESIZE_LABELS = {'RESIZE': 'Just Resize', 'CROP_AND_RESIZE': 'Crop and Resize', 'RESIZE_AND_FILL': 'Resize and Fill'}


class LayerMethod(Enum):
    FG_ONLY_ATTN_SD15 = "(SD1.5) Only Generate Transparent Image (Attention Injection)"
    FG_TO_BG_SD15 = "(SD1.5) From Foreground to Background (need batch size 2)"
    BG_TO_FG_SD15 = "(SD1.5) From Background to Foreground (need batch size 2)"
    JOINT_SD15 = "(SD1.5) Generate Everything Together (need batch size 3)"
    FG_ONLY_ATTN = "(SDXL) Only Generate Transparent Image (Attention Injection)"
    FG_ONLY_CONV = "(SDXL) Only Generate Transparent Image (Conv Injection)"
    FG_TO_BLEND = "(SDXL) From Foreground to Blending"
    FG_BLEND_TO_BG = "(SDXL) From Foreground and Blending to Background"
    BG_TO_BLEND = "(SDXL) From Background to Blending"
    BG_BLEND_TO_FG = "(SDXL) From Background and Blending to Foreground"


def load_layer_model_state_dict(filename):
    return utils.load_torch_file(filename, safe_load=True)


class LayerDiffusionForForge(scripts.Script):
    # Legacy beta API identifier, not a user-facing translated label.
    api_aliases = ("LayerDiffuse 圖層透明生成",)

    def title(self):
        return "LayerDiffuse"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            gr.Markdown('Experimental: supports standard SD1.5 / SDXL models; disabled by default. Required weights download on first use. Disable ADetailer and use PNG output. SD1.5 weight must be 1.')
            enabled = gr.Checkbox(label='Enable', value=False)
            method = gr.Dropdown(choices=[(METHOD_LABELS[e.name], e.value) for e in LayerMethod], value=LayerMethod.FG_ONLY_ATTN.value, label="Generation Method", type='value')
            gr.HTML('</br>')  # some strange gradio problems

            with gr.Row():
                with gr.Column(visible=False) as fg_col:
                    gr.Markdown('Foreground Image')
                    fg_image = ForgeCanvas(numpy=True, no_scribbles=True, height=300).background
                with gr.Column(visible=False) as bg_col:
                    gr.Markdown('Background Image')
                    bg_image = ForgeCanvas(numpy=True, no_scribbles=True, height=300).background
                with gr.Column(visible=False) as blend_col:
                    gr.Markdown('Blended Image')
                    blend_image = ForgeCanvas(numpy=True, no_scribbles=True, height=300).background

            gr.HTML('</br>')  # some strange gradio problems

            with gr.Row():
                weight = gr.Slider(label="Weight (SDXL)", value=1.0, minimum=0.0, maximum=2.0, step=0.001)
                ending_step = gr.Slider(label="Layer Effect Stop Fraction", value=1.0, minimum=0.0, maximum=1.0)

            fg_additional_prompt = gr.Textbox(value="", placeholder="Additional Foreground Prompt", visible=False, label='Additional Foreground Prompt')
            bg_additional_prompt = gr.Textbox(value="", placeholder="Additional Background Prompt", visible=False, label='Additional Background Prompt')
            blend_additional_prompt = gr.Textbox(value="", placeholder="Additional Blended Image Prompt", visible=False, label='Additional Blended Image Prompt')

            resize_mode = gr.Radio(choices=[(RESIZE_LABELS[e.name], e.value) for e in ResizeMode], value=ResizeMode.CROP_AND_RESIZE.value, label="Image Resize Mode", type='value', visible=False)
            output_origin = gr.Checkbox(label='Output Original Image in img2img (upstream reserved option)', value=False, visible=False)

        def method_changed(m):
            m = LayerMethod(m)

            if m == LayerMethod.FG_TO_BG_SD15:
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False, value=''), gr.update(visible=True), gr.update(visible=True)

            if m == LayerMethod.BG_TO_FG_SD15:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False, value=''), gr.update(visible=True)

            if m == LayerMethod.JOINT_SD15:
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

            if m == LayerMethod.FG_TO_BLEND:
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False, value=''), gr.update(visible=False, value=''), gr.update(visible=False, value='')

            if m == LayerMethod.BG_TO_BLEND:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False, value=''), gr.update(visible=False, value=''), gr.update(visible=False, value='')

            if m == LayerMethod.BG_BLEND_TO_FG:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False, value=''), gr.update(visible=False, value=''), gr.update(visible=False, value='')

            if m == LayerMethod.FG_BLEND_TO_BG:
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False, value=''), gr.update(visible=False, value=''), gr.update(visible=False, value='')

            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, value=''), gr.update(visible=False, value=''), gr.update(visible=False, value='')

        method.change(method_changed, inputs=method, outputs=[fg_col, bg_col, blend_col, resize_mode, fg_additional_prompt, bg_additional_prompt, blend_additional_prompt], show_progress=False, queue=False)

        return enabled, method, weight, ending_step, fg_image, bg_image, blend_image, resize_mode, output_origin, fg_additional_prompt, bg_additional_prompt, blend_additional_prompt

    @fail_closed
    def process(self, p, *script_args):
        if not script_args or not script_args[0]:
            return
        method = LayerMethod(script_args[1])
        if method.name.endswith('_SD15') and script_args[2] != 1.0:
            raise ValueError('SD1.5 Attention 方法目前只支援權重 1，請將權重設為 1。')
        validate_request(p, method, *script_args[4:7])

    def cleanup_processing(self, p):
        cleanup_request(p)

    @fail_closed
    def process_before_every_sampling(self, p: StableDiffusionProcessing, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        enabled, method, weight, ending_step, fg_image, bg_image, blend_image, resize_mode, output_origin, fg_additional_prompt, bg_additional_prompt, blend_additional_prompt = script_args

        if not enabled:
            return

        p.extra_generation_params.update(dict(
            layerdiffusion_enabled=enabled,
            layerdiffusion_method=method,
            layerdiffusion_weight=weight,
            layerdiffusion_ending_step=ending_step,
            layerdiffusion_fg_image=fg_image is not None,
            layerdiffusion_bg_image=bg_image is not None,
            layerdiffusion_blend_image=blend_image is not None,
            layerdiffusion_resize_mode=resize_mode,
            layerdiffusion_fg_additional_prompt=fg_additional_prompt,
            layerdiffusion_bg_additional_prompt=bg_additional_prompt,
            layerdiffusion_blend_additional_prompt=blend_additional_prompt,
        ))

        B, C, H, W = kwargs['noise'].shape  # latent_shape
        height = H * 8
        width = W * 8
        batch_size = p.batch_size

        method = LayerMethod(method)
        if method.name.endswith('_SD15') and weight != 1.0:
            raise ValueError('SD1.5 Attention 方法目前只支援權重 1，請將權重設為 1。')
        validate_request(p, method, fg_image, bg_image, blend_image)
        print(f'[LayerDiffuse] {METHOD_LABELS[method.name]}')

        resize_mode = ResizeMode(resize_mode)
        fg_image = crop_and_resize_image(rgba2rgbfp32(fg_image), resize_mode, height, width) if fg_image is not None else None
        bg_image = crop_and_resize_image(rgba2rgbfp32(bg_image), resize_mode, height, width) if bg_image is not None else None
        blend_image = crop_and_resize_image(rgba2rgbfp32(blend_image), resize_mode, height, width) if blend_image is not None else None

        original_unet = p.sd_model.forge_objects.unet.clone()
        unet = p.sd_model.forge_objects.unet.clone()
        vae = p.sd_model.forge_objects.vae
        clip = p.sd_model.forge_objects.clip

        if method in [LayerMethod.FG_TO_BLEND, LayerMethod.FG_BLEND_TO_BG, LayerMethod.BG_TO_BLEND, LayerMethod.BG_BLEND_TO_FG]:
            if fg_image is not None:
                fg_image = vae.encode(torch.from_numpy(np.ascontiguousarray(fg_image[None].copy())))
                fg_image = vae.first_stage_model.process_in(fg_image)

            if bg_image is not None:
                bg_image = vae.encode(torch.from_numpy(np.ascontiguousarray(bg_image[None].copy())))
                bg_image = vae.first_stage_model.process_in(bg_image)

            if blend_image is not None:
                blend_image = vae.encode(torch.from_numpy(np.ascontiguousarray(blend_image[None].copy())))
                blend_image = vae.first_stage_model.process_in(blend_image)

        if method in [LayerMethod.FG_TO_BG_SD15, LayerMethod.BG_TO_FG_SD15]:
            if fg_image is not None:
                fg_image = torch.from_numpy(np.ascontiguousarray(fg_image[None].copy())).movedim(-1, 1)

            if bg_image is not None:
                bg_image = torch.from_numpy(np.ascontiguousarray(bg_image[None].copy())).movedim(-1, 1)

            if blend_image is not None:
                blend_image = torch.from_numpy(np.ascontiguousarray(blend_image[None].copy())).movedim(-1, 1)

        if method == LayerMethod.FG_ONLY_ATTN_SD15:
            model_path = load_file_from_url(
                url=f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{REVISION}/layer_sd15_transparent_attn.safetensors',
                model_dir=layer_model_root,
                file_name='layer_sd15_transparent_attn.safetensors'
            )
            layer_lora_model = load_layer_model_state_dict(model_path)
            patcher = AttentionSharingPatcher(unet, frames=1, use_control=False)
            patcher.load_state_dict(layer_lora_model, strict=True)

        original_prompt = p.prompts[0]
        fg_additional_prompt = fg_additional_prompt or ''
        bg_additional_prompt = bg_additional_prompt or ''
        blend_additional_prompt = blend_additional_prompt or ''

        fg_additional_prompt = fg_additional_prompt + ', ' + original_prompt if fg_additional_prompt != '' else None
        bg_additional_prompt = bg_additional_prompt + ', ' + original_prompt if bg_additional_prompt != '' else None
        blend_additional_prompt = blend_additional_prompt + ', ' + original_prompt if blend_additional_prompt != '' else None

        fg_cond = forge_clip_encode(p.sd_model, fg_additional_prompt)
        bg_cond = forge_clip_encode(p.sd_model, bg_additional_prompt)
        blend_cond = forge_clip_encode(p.sd_model, blend_additional_prompt)

        if method == LayerMethod.JOINT_SD15:
            unet.set_transformer_option('cond_overwrite', [fg_cond, bg_cond, blend_cond])
            model_path = load_file_from_url(
                url=f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{REVISION}/layer_sd15_joint.safetensors',
                model_dir=layer_model_root,
                file_name='layer_sd15_joint.safetensors'
            )
            layer_lora_model = load_layer_model_state_dict(model_path)
            patcher = AttentionSharingPatcher(unet, frames=3, use_control=False)
            patcher.load_state_dict(layer_lora_model, strict=True)

        if method == LayerMethod.FG_TO_BG_SD15:
            unet.set_transformer_option('cond_overwrite', [bg_cond, blend_cond])
            model_path = load_file_from_url(
                url=f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{REVISION}/layer_sd15_fg2bg.safetensors',
                model_dir=layer_model_root,
                file_name='layer_sd15_fg2bg.safetensors'
            )
            layer_lora_model = load_layer_model_state_dict(model_path)
            patcher = AttentionSharingPatcher(unet, frames=2, use_control=True)
            patcher.load_state_dict(layer_lora_model, strict=True)
            patcher.set_control(fg_image)

        if method == LayerMethod.BG_TO_FG_SD15:
            unet.set_transformer_option('cond_overwrite', [fg_cond, blend_cond])
            model_path = load_file_from_url(
                url=f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{REVISION}/layer_sd15_bg2fg.safetensors',
                model_dir=layer_model_root,
                file_name='layer_sd15_bg2fg.safetensors'
            )
            layer_lora_model = load_layer_model_state_dict(model_path)
            patcher = AttentionSharingPatcher(unet, frames=2, use_control=True)
            patcher.load_state_dict(layer_lora_model, strict=True)
            patcher.set_control(bg_image)

        if method == LayerMethod.FG_ONLY_ATTN:
            model_path = load_file_from_url(
                url=f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{REVISION}/layer_xl_transparent_attn.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_transparent_attn.safetensors'
            )
            layer_lora_model = load_layer_model_state_dict(model_path)
            load_frozen_patcher_compat(unet,'layer_xl_transparent_attn.safetensors', layer_lora_model, weight)

        if method == LayerMethod.FG_ONLY_CONV:
            model_path = load_file_from_url(
                url=f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{REVISION}/layer_xl_transparent_conv.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_transparent_conv.safetensors'
            )
            layer_lora_model = load_layer_model_state_dict(model_path)
            load_frozen_patcher_compat(unet,'layer_xl_transparent_conv.safetensors', layer_lora_model, weight)

        if method == LayerMethod.BG_TO_BLEND:
            model_path = load_file_from_url(
                url=f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{REVISION}/layer_xl_bg2ble.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_bg2ble.safetensors'
            )
            unet.extra_concat_condition = bg_image
            layer_lora_model = load_layer_model_state_dict(model_path)
            load_frozen_patcher_compat(unet,'layer_xl_bg2ble.safetensors', layer_lora_model, weight)

        if method == LayerMethod.FG_TO_BLEND:
            model_path = load_file_from_url(
                url=f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{REVISION}/layer_xl_fg2ble.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_fg2ble.safetensors'
            )
            unet.extra_concat_condition = fg_image
            layer_lora_model = load_layer_model_state_dict(model_path)
            load_frozen_patcher_compat(unet,'layer_xl_fg2ble.safetensors', layer_lora_model, weight)

        if method == LayerMethod.BG_BLEND_TO_FG:
            model_path = load_file_from_url(
                url=f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{REVISION}/layer_xl_bgble2fg.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_bgble2fg.safetensors'
            )
            unet.extra_concat_condition = torch.cat([bg_image, blend_image], dim=1)
            layer_lora_model = load_layer_model_state_dict(model_path)
            load_frozen_patcher_compat(unet,'layer_xl_bgble2fg.safetensors', layer_lora_model, weight)

        if method == LayerMethod.FG_BLEND_TO_BG:
            model_path = load_file_from_url(
                url=f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{REVISION}/layer_xl_fgble2bg.safetensors',
                model_dir=layer_model_root,
                file_name='layer_xl_fgble2bg.safetensors'
            )
            unet.extra_concat_condition = torch.cat([fg_image, blend_image], dim=1)
            layer_lora_model = load_layer_model_state_dict(model_path)
            load_frozen_patcher_compat(unet,'layer_xl_fgble2bg.safetensors', layer_lora_model, weight)

        sigma_end = unet.model.predictor.percent_to_sigma(ending_step)

        def remove_concat(cond):
            if cond is None:
                return None
            result = []
            for item in cond:
                entry = dict(item)
                entry['model_conds'] = dict(item['model_conds'])
                entry['model_conds'].pop('c_concat', None)
                result.append(entry)
            return result

        def conditioning_modifier(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
            if timestep[0].item() < sigma_end:
                if not is_model_loaded(original_unet):
                    sampling_prepare(original_unet, x)
                target_model = original_unet.model
                cond = remove_concat(cond)
                uncond = remove_concat(uncond)
            else:
                target_model = model
                if unet.extra_concat_condition is not None:
                    concat = match_batch(unet.extra_concat_condition, x)
                    cond = attach_concat(cond, concat, Condition)
                    uncond = attach_concat(uncond, concat, Condition)

            return target_model, x, timestep, uncond, cond, cond_scale, model_options, seed

        unet.add_conditioning_modifier(conditioning_modifier)

        record_patch(p, p.sd_model.forge_objects, 'unet', p.sd_model.forge_objects.unet, unet)
        p.sd_model.forge_objects.unet = unet
        return

    @fail_closed
    def postprocess_image_after_composite(self, p: StableDiffusionProcessing, pp, *script_args, **kwargs):


        enabled, method, weight, ending_step, fg_image, bg_image, blend_image, resize_mode, output_origin, fg_additional_prompt, bg_additional_prompt, blend_additional_prompt = script_args
        if not enabled:
            return


        mod_number = 1
        method = LayerMethod(method)
        need_process = False

        if method in [LayerMethod.FG_ONLY_ATTN, LayerMethod.FG_ONLY_CONV, LayerMethod.BG_BLEND_TO_FG]:
            need_process = True
            def load_codec():
                model_path = load_file_from_url(
                    url=f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{REVISION}/vae_transparent_decoder.safetensors',
                    model_dir=layer_model_root,
                    file_name='vae_transparent_decoder.safetensors'
                )
                return TransparentVAEDecoder(utils.load_torch_file(model_path))
            vae_transparent_decoder = get_transparent_codec(p, 'sdxl', 'decoder', load_codec)

        if method in [LayerMethod.FG_ONLY_ATTN_SD15, LayerMethod.JOINT_SD15, LayerMethod.BG_TO_FG_SD15]:
            need_process = True
            def load_codec():
                model_path = load_file_from_url(
                    url=f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{REVISION}/layer_sd15_vae_transparent_decoder.safetensors',
                    model_dir=layer_model_root,
                    file_name='layer_sd15_vae_transparent_decoder.safetensors'
                )
                return TransparentVAEDecoder(utils.load_torch_file(model_path))
            vae_transparent_decoder = get_transparent_codec(p, 'sd15', 'decoder', load_codec)
            if method == LayerMethod.JOINT_SD15:
                mod_number = 3
            if method == LayerMethod.BG_TO_FG_SD15:
                mod_number = 2

        if not need_process:
            return

        i = pp.index

        if i % mod_number == 0:
            latent = p.latents_after_sampling[i]
            pixel = p.pixels_after_sampling[i]

            lC, lH, lW = latent.shape
            if lH != pixel.height // 8 or lW != pixel.width // 8:
                print('[LayerDiffuse] 圖片尺寸已變更，改用零潛在張量解碼透明圖。')
                latent = torch.zeros((lC, pixel.height // 8, pixel.width // 8)).to(latent)

            png, vis = vae_transparent_decoder.decode(latent, pixel)
            pp.image = png
            p.extra_result_images.append(vis)
        return

    @fail_closed
    def before_process_init_images(self, p: StableDiffusionProcessingImg2Img, pp, *script_args, **kwargs):


        enabled, method, weight, ending_step, fg_image, bg_image, blend_image, resize_mode, output_origin, fg_additional_prompt, bg_additional_prompt, blend_additional_prompt = script_args
        if not enabled:
            return

        method = LayerMethod(method)
        need_process = False

        if method in [LayerMethod.FG_ONLY_ATTN, LayerMethod.FG_ONLY_CONV, LayerMethod.BG_BLEND_TO_FG]:
            need_process = True
            def load_codec():
                model_path = load_file_from_url(
                    url=f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{REVISION}/vae_transparent_encoder.safetensors',
                    model_dir=layer_model_root,
                    file_name='vae_transparent_encoder.safetensors'
                )
                return TransparentVAEEncoder(utils.load_torch_file(model_path))
            vae_transparent_encoder = get_transparent_codec(p, 'sdxl', 'encoder', load_codec)

        if method in [LayerMethod.FG_ONLY_ATTN_SD15, LayerMethod.JOINT_SD15, LayerMethod.BG_TO_FG_SD15]:
            need_process = True
            def load_codec():
                model_path = load_file_from_url(
                    url=f'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/{REVISION}/layer_sd15_vae_transparent_encoder.safetensors',
                    model_dir=layer_model_root,
                    file_name='layer_sd15_vae_transparent_encoder.safetensors'
                )
                return TransparentVAEEncoder(utils.load_torch_file(model_path))
            vae_transparent_encoder = get_transparent_codec(p, 'sd15', 'encoder', load_codec)

        if not need_process:
            return

        input_png_raw = p.init_images[0].convert("RGBA")
        input_png_bg_grey = images.flatten(input_png_raw, (127, 127, 127)).convert('RGBA')
        p.init_images = [input_png_bg_grey]

        crop_region = pp['crop_region']
        image = input_png_raw

        if crop_region is None and p.resize_mode != 3:
            image = images.resize_image(p.resize_mode, image, p.width, p.height, force_RGBA=True)

        if crop_region is not None:
            image = image.crop(crop_region)
            image = images.resize_image(2, image, p.width, p.height, force_RGBA=True)

        latent_offset = vae_transparent_encoder.encode(image)

        original_vae = p.sd_model.forge_objects.vae
        vae = original_vae.clone()

        def vae_regulation(posterior):
            z = posterior.mean + posterior.std * latent_offset.to(posterior.mean)
            return z

        vae.patcher.set_model_vae_regulation(vae_regulation)

        record_patch(p, p.sd_model.forge_objects, 'vae', original_vae, vae)
        p.sd_model.forge_objects.vae = vae
        return
