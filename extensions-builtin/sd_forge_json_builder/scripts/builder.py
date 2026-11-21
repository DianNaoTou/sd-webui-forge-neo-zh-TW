import gradio as gr
from modules import shared, script_callbacks


def builder_ui():
    return [(gr.Blocks(), "JSON Builder", "sd-forge-json-builder")]


def json_builder_api(demo, app):
    @app.get("/internal/json_builder_prompt_styles")
    async def get_prompt_styles():
        return [{"name": name, "prompt": style.prompt, "negative_prompt": style.negative_prompt} for name, style in shared.prompt_styles.styles.items()]


script_callbacks.on_ui_tabs(builder_ui)
script_callbacks.on_app_started(json_builder_api)
