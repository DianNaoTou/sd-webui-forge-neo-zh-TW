"""
Image2Prompt — extra tab for SD WebUI Forge Neo.

Upload an image and generate a text prompt from it using a vision-language
encoder (Qwen2-VL / Qwen2.5-VL or Florence-2). The generated prompt can be
sent directly to txt2img / img2img.
"""

from modules.localization import translate as ui_text

import gc
import gradio as gr
import torch

from modules import script_callbacks, shared, devices

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    'Qwen2-VL-2B-Instruct (recommended, ~5GB)': {
        "repo": "Qwen/Qwen2-VL-2B-Instruct",
        "type": "qwen2vl",
    },
    'Qwen2.5-VL-3B-Instruct (better quality, ~7GB)': {
        "repo": "Qwen/Qwen2.5-VL-3B-Instruct",
        "type": "qwen25vl",
    },
    'Qwen2-VL-7B-Instruct (best quality, ~16GB)': {
        "repo": "Qwen/Qwen2-VL-7B-Instruct",
        "type": "qwen2vl",
    },
    'Florence-2-base (lightweight, ~1GB)': {
        "repo": "microsoft/Florence-2-base",
        "type": "florence2",
    },
    "Florence-2-large (~3GB)": {
        "repo": "microsoft/Florence-2-large",
        "type": "florence2",
    },
}

STYLES = {
    'SD Prompt / Natural Language': (
        "Describe this image as a Stable Diffusion prompt. Output a single "
        "comma-separated prompt describing: subject, appearance, clothing, "
        "pose, background, lighting, camera angle, art style and quality "
        "modifiers. Output ONLY the prompt, no explanations, no quotes."
    ),
    'Danbooru Tags': (
        "Describe this image using booru-style tags. Output a single line of "
        "lowercase comma-separated tags (e.g. '1girl, long hair, ...'), "
        "covering subject, features, clothing, pose, background and style. "
        "Output ONLY the tags."
    ),
    'Detailed Description': (
        "Describe this image in extreme detail in one paragraph: subject, "
        "colors, textures, composition, lighting, mood and style. "
        "Output ONLY the description."
    ),
    'Short Description': (
        "Describe this image in one short sentence. Output ONLY the sentence."
    ),
}

# Legacy selector identifiers remain accepted by callback/API callers.
LEGACY_LABELS = {'Qwen2-VL-2B-Instruct (推薦, ~5GB)': 'Qwen2-VL-2B-Instruct (recommended, ~5GB)', 'Qwen2.5-VL-3B-Instruct (效果較佳, ~7GB)': 'Qwen2.5-VL-3B-Instruct (better quality, ~7GB)', 'Qwen2-VL-7B-Instruct (最佳效果, ~16GB)': 'Qwen2-VL-7B-Instruct (best quality, ~16GB)', 'Florence-2-base (輕量, ~1GB)': 'Florence-2-base (lightweight, ~1GB)', 'SD 提示詞／自然語言': 'SD Prompt / Natural Language', 'Danbooru 標籤格式': 'Danbooru Tags', '詳細描述': 'Detailed Description', '簡短描述': 'Short Description'}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_state = {"model": None, "processor": None, "key": None}


def _free_model():
    _state["model"] = None
    _state["processor"] = None
    _state["key"] = None
    gc.collect()
    devices.torch_gc()


def _get_device_dtype(force_cpu: bool):
    if force_cpu or not torch.cuda.is_available():
        return "cpu", torch.float32
    return "cuda", torch.float16


def _load_model(model_label: str, force_cpu: bool):
    info = MODELS[model_label]
    key = (info["repo"], force_cpu)
    if _state["key"] == key and _state["model"] is not None:
        return _state["model"], _state["processor"]

    _free_model()
    device, dtype = _get_device_dtype(force_cpu)
    print(f"[Image2Prompt] Loading {info['repo']} on {device} ({dtype}) ...")

    from transformers import AutoProcessor

    if info["type"] in ("qwen2vl", "qwen25vl"):
        if info["type"] == "qwen2vl":
            from transformers import Qwen2VLForConditionalGeneration as VLModel
        else:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
            except ImportError:
                raise RuntimeError(
                    "Qwen2.5-VL requires transformers >= 4.49. "
                    "Update with: pip install -U transformers"
                )
        model = VLModel.from_pretrained(
            info["repo"], torch_dtype=dtype, device_map=device
        )
        processor = AutoProcessor.from_pretrained(info["repo"])
    else:  # florence2
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            info["repo"], torch_dtype=dtype, trust_remote_code=True
        ).to(device)
        processor = AutoProcessor.from_pretrained(
            info["repo"], trust_remote_code=True
        )

    model.eval()
    _state.update(model=model, processor=processor, key=key)
    print("[Image2Prompt] Model ready.")
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_qwen(model, processor, image, instruction, max_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=False,
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    result = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return result.strip()


def _run_florence(model, processor, image, max_tokens):
    task = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=task, images=image, return_tensors="pt").to(
        model.device, model.dtype if model.dtype != torch.float32 else None
    )
    with torch.inference_mode():
        out_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=int(max_tokens),
            num_beams=3,
            do_sample=False,
        )
    text = processor.batch_decode(out_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=task, image_size=(image.width, image.height)
    )
    return parsed.get(task, "").strip()


def generate_prompt(image, model_label, style_label, custom_instruction,
                    max_tokens, force_cpu, unload_after):
    model_label = LEGACY_LABELS.get(model_label, model_label)
    style_label = LEGACY_LABELS.get(style_label, style_label)
    if image is None:
        return "", ui_text('Please upload an image first.')

    try:
        image = image.convert("RGB")
        model, processor = _load_model(model_label, force_cpu)

        instruction = (custom_instruction or "").strip() or STYLES[style_label]

        if MODELS[model_label]["type"] == "florence2":
            result = _run_florence(model, processor, image, max_tokens)
        else:
            result = _run_qwen(model, processor, image, instruction, max_tokens)

        # cleanup: single line, no wrapping quotes
        result = result.strip().strip('"').strip()

        status = ui_text('Prompt generated with {0}', model_label.split(' (')[0])
        if unload_after:
            _free_model()
            status += ui_text(' — Model unloaded from VRAM.')
        return result, status

    except Exception as e:
        _free_model()
        return "", ui_text('Error: {0}', e)


def unload_clicked():
    _free_model()
    return ui_text('Model unloaded and VRAM released.')


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _send_to(tab: str):
    """JS to switch tab; the prompt value is passed through the fn output."""
    return f"(x) => {{ switch_to_{tab}(); return x; }}"


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown('## Image2Prompt — Generate prompts from images')

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(
                    label='Input Image',
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=420,
                )
                model_dd = gr.Dropdown(
                    label='Vision-language Model',
                    choices=list(MODELS.keys()),
                    value=list(MODELS.keys())[0],
                )
                style_dd = gr.Dropdown(
                    label='Prompt Format',
                    choices=list(STYLES.keys()),
                    value=list(STYLES.keys())[0],
                )
                with gr.Accordion('Advanced Options', open=False):
                    custom_tb = gr.Textbox(
                        label='Custom Instruction (overrides prompt format, Qwen only)',
                        placeholder='E.g. Describe only the clothing in booru tags...',
                        lines=2,
                    )
                    max_tokens = gr.Slider(
                        label='Maximum Length (Tokens)',
                        minimum=32, maximum=1024, value=256, step=16,
                    )
                    force_cpu = gr.Checkbox(label='Force CPU (slower, uses no VRAM)', value=False)
                    unload_after = gr.Checkbox(
                        label='Unload Model After Generation (free VRAM)',
                        value=False,
                    )

                generate_btn = gr.Button('Generate Prompt', variant="primary")
                unload_btn = gr.Button('Unload Model / Free VRAM')

            with gr.Column(scale=1):
                prompt_out = gr.Textbox(
                    label='Generated Prompt',
                    lines=10,
                    show_copy_button=True,
                    interactive=True,
                )
                status = gr.Markdown("")
                with gr.Row():
                    send_t2i = gr.Button('Send to txt2img')
                    send_i2i = gr.Button('Send to img2img')

        generate_btn.click(
            fn=generate_prompt,
            inputs=[image, model_dd, style_dd, custom_tb, max_tokens,
                    force_cpu, unload_after],
            outputs=[prompt_out, status],
        )
        unload_btn.click(fn=unload_clicked, inputs=[], outputs=[status])

        # Send-to buttons: copy prompt into the main prompt boxes and switch tab
        try:
            from modules.ui import switch_values_symbol  # noqa: F401  (presence check)
        except Exception:
            pass

        def _register_send(btn, target_tab):
            btn.click(
                fn=lambda x: x,
                inputs=[prompt_out],
                outputs=[prompt_out],
                js=(
                    "(x) => {"
                    f"  const tab = '{target_tab}';"
                    "  const app = gradioApp();"
                    "  const ta = app.querySelector(`#${tab}_prompt textarea`);"
                    "  if (ta) {"
                    "    ta.value = x;"
                    "    ta.dispatchEvent(new Event('input', {bubbles: true}));"
                    "  }"
                    f"  switch_to_{target_tab}();"
                    "  return x;"
                    "}"
                ),
            )

        try:
            _register_send(send_t2i, "txt2img")
            _register_send(send_i2i, "img2img")
        except TypeError:
            # Older Gradio: `js` kwarg named `_js`
            send_t2i.click(fn=lambda x: x, inputs=[prompt_out], outputs=[prompt_out])
            send_i2i.click(fn=lambda x: x, inputs=[prompt_out], outputs=[prompt_out])

    return [(ui, "Image2Prompt", "image2prompt_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
