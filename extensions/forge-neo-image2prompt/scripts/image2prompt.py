"""
Image2Prompt — extra tab for SD WebUI Forge Neo.

Upload an image and generate a text prompt from it using a vision-language
encoder (Qwen2-VL / Qwen2.5-VL or Florence-2). The generated prompt can be
sent directly to txt2img / img2img.
"""

import gc
import gradio as gr
import torch

from modules import script_callbacks, shared, devices

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "Qwen2-VL-2B-Instruct (推薦, ~5GB)": {
        "repo": "Qwen/Qwen2-VL-2B-Instruct",
        "type": "qwen2vl",
    },
    "Qwen2.5-VL-3B-Instruct (效果較佳, ~7GB)": {
        "repo": "Qwen/Qwen2.5-VL-3B-Instruct",
        "type": "qwen25vl",
    },
    "Qwen2-VL-7B-Instruct (最佳效果, ~16GB)": {
        "repo": "Qwen/Qwen2-VL-7B-Instruct",
        "type": "qwen2vl",
    },
    "Florence-2-base (輕量, ~1GB)": {
        "repo": "microsoft/Florence-2-base",
        "type": "florence2",
    },
    "Florence-2-large (~3GB)": {
        "repo": "microsoft/Florence-2-large",
        "type": "florence2",
    },
}

STYLES = {
    "SD 提示詞／自然語言": (
        "Describe this image as a Stable Diffusion prompt. Output a single "
        "comma-separated prompt describing: subject, appearance, clothing, "
        "pose, background, lighting, camera angle, art style and quality "
        "modifiers. Output ONLY the prompt, no explanations, no quotes."
    ),
    "Danbooru 標籤格式": (
        "Describe this image using booru-style tags. Output a single line of "
        "lowercase comma-separated tags (e.g. '1girl, long hair, ...'), "
        "covering subject, features, clothing, pose, background and style. "
        "Output ONLY the tags."
    ),
    "詳細描述": (
        "Describe this image in extreme detail in one paragraph: subject, "
        "colors, textures, composition, lighting, mood and style. "
        "Output ONLY the description."
    ),
    "簡短描述": (
        "Describe this image in one short sentence. Output ONLY the sentence."
    ),
}

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
                    "Qwen2.5-VL richiede transformers >= 4.49. "
                    "Aggiorna con: pip install -U transformers"
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
    if image is None:
        return "", "⚠️ 請先上傳圖片。"

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

        status = f"✅ 產生的提示詞 con {model_label.split(' (')[0]}"
        if unload_after:
            _free_model()
            status += " — 模型已從顯存卸載。"
        return result, status

    except Exception as e:
        _free_model()
        return "", f"❌ 錯誤： {e}"


def unload_clicked():
    _free_model()
    return "🧹 模型已卸載，顯存已釋放。"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def _send_to(tab: str):
    """JS to switch tab; the prompt value is passed through the fn output."""
    return f"(x) => {{ switch_to_{tab}(); return x; }}"


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui:
        gr.Markdown("## 🖼️ → 📝 Image2Prompt — 從圖片產生提示詞")

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(
                    label="輸入圖片",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=420,
                )
                model_dd = gr.Dropdown(
                    label="辨識模型（視覺語言）",
                    choices=list(MODELS.keys()),
                    value=list(MODELS.keys())[0],
                )
                style_dd = gr.Dropdown(
                    label="提示詞格式",
                    choices=list(STYLES.keys()),
                    value=list(STYLES.keys())[0],
                )
                with gr.Accordion("進階選項", open=False):
                    custom_tb = gr.Textbox(
                        label="自訂指令（覆蓋提示詞格式，僅限 Qwen）",
                        placeholder="Es: Describe only the clothing in booru tags...",
                        lines=2,
                    )
                    max_tokens = gr.Slider(
                        label="最大長度（Token）",
                        minimum=32, maximum=1024, value=256, step=16,
                    )
                    force_cpu = gr.Checkbox(label="強制使用 CPU（較慢，不占用顯存）", value=False)
                    unload_after = gr.Checkbox(
                        label="每次產生後卸載模型（釋放顯存）",
                        value=False,
                    )

                generate_btn = gr.Button("🚀 產生提示詞", variant="primary")
                unload_btn = gr.Button("🧹 卸載模型／釋放顯存")

            with gr.Column(scale=1):
                prompt_out = gr.Textbox(
                    label="產生的提示詞",
                    lines=10,
                    show_copy_button=True,
                    interactive=True,
                )
                status = gr.Markdown("")
                with gr.Row():
                    send_t2i = gr.Button("📤 傳送至文生圖")
                    send_i2i = gr.Button("📤 傳送至圖生圖")

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
                    "  const btns = app.querySelectorAll('#tabs > .tab-nav > button, #tabs > div > button');"
                    "  for (const b of btns) {"
                    "    const t = b.textContent.trim().toLowerCase();"
                    "    if ((tab === 'txt2img' && t === 'txt2img') || (tab === 'img2img' && t === 'img2img')) { b.click(); break; }"
                    "  }"
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
