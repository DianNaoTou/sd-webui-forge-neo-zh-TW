import gradio as gr
from modules import shared
from modules.options import OptionDiv


def on_ui_settings():
    section = ("civitai_model_downloader", 'Civitai Model Downloader')
    options = {
        "civitai_api_key": shared.OptionInfo(
            "",
            'Civitai API Key',
            gr.Textbox,
            {"interactive": True, "type": "text"},
            section=section,
        ).info('Required to download models that need authorization.'),
        "civitai_preferred_domain": shared.OptionInfo(
            "civitai.red",
            'Preferred Civitai Domain',
            gr.Radio,
            {"choices": ["civitai.com", "civitai.red"], "interactive": True},
            section=section,
        ).info('Choose the domain used for Civitai API requests and page links.'),
        "sep00": OptionDiv(),
        "civitai_card_button_open_url": shared.OptionInfo(
            True,
            'Show Open URL button on model cards',
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ).info('Open the model page on Civitai.'),
        "civitai_card_button_delete": shared.OptionInfo(
            True,
            'Show Delete button on model cards',
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ).info('Delete the model and related files.'),
        "sep01": OptionDiv(),
        "civitai_enable_folder_selector": shared.OptionInfo(
            False,
            'Enable destination folder selection',
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ).info('Choose a folder for each download; otherwise use the model type automatically.'),
        "civitai_folder_lycoris": shared.OptionInfo(
            "Lora",
            'LyCORIS Model Folder',
            gr.Textbox,
            {"interactive": True, "type": "text"},
            section=section,
        ).info('Default: Lora'),
        "civitai_folder_locon": shared.OptionInfo(
            "Lora",
            'LoCon Model Folder',
            gr.Textbox,
            {"interactive": True, "type": "text"},
            section=section,
        ).info('Default: Lora'),
        "sep03": OptionDiv(),
        "civitai_disable_card_description": shared.OptionInfo(
            True,
            'Hide model card descriptions',
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ).info('Hide model card descriptions. Restart required.'),
        "civitai_show_model_title_on_card": shared.OptionInfo(
            False,
            'Display model title instead of filename',
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ).info('Display the metadata title on cards; search still uses filenames. Restart required.'),
    }
    for opt_name, opt_info in options.items():
        # Ensure OptionDiv has a section attribute set
        if (
            type(opt_info).__name__ == "OptionDiv"
            and getattr(opt_info, "section", None) is None
        ):
            opt_info.section = section
        shared.opts.add_option(opt_name, opt_info)
