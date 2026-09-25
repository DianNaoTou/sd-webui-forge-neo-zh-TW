import gradio as gr
from modules import shared
from modules.options import OptionDiv


def on_ui_settings():
    section = ("civitai_model_downloader", "Civitai 模型下載器")
    options = {
        "civitai_api_key": shared.OptionInfo(
            "",
            "Civitai API 金鑰",
            gr.Textbox,
            {"interactive": True, "type": "text"},
            section=section,
        ).info("用於下載需要授權的模型。"),
        "civitai_preferred_domain": shared.OptionInfo(
            "civitai.red",
            "偏好的 Civitai 網域",
            gr.Radio,
            {"choices": ["civitai.com", "civitai.red"], "interactive": True},
            section=section,
        ).info("選擇 Civitai API 與頁面連結使用的網域。"),
        "sep00": OptionDiv(),
        "civitai_card_button_open_url": shared.OptionInfo(
            True,
            "在模型卡片顯示「開啟網址」按鈕",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ).info("開啟模型在 Civitai 的頁面。"),
        "civitai_card_button_delete": shared.OptionInfo(
            True,
            "在模型卡片顯示「刪除」按鈕",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ).info("刪除模型的相關檔案。"),
        "sep01": OptionDiv(),
        "civitai_enable_folder_selector": shared.OptionInfo(
            False,
            "啟用目的地資料夾選擇",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ).info("啟用後可為每次下載選擇資料夾；停用時依模型類型自動存放。"),
        "civitai_folder_lycoris": shared.OptionInfo(
            "Lora",
            "LyCORIS 模型資料夾",
            gr.Textbox,
            {"interactive": True, "type": "text"},
            section=section,
        ).info("預設：Lora"),
        "civitai_folder_locon": shared.OptionInfo(
            "Lora",
            "LoCon 模型資料夾",
            gr.Textbox,
            {"interactive": True, "type": "text"},
            section=section,
        ).info("預設：Lora"),
        "sep03": OptionDiv(),
        "civitai_disable_card_description": shared.OptionInfo(
            True,
            "隱藏模型卡片描述",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ).info("隱藏模型卡片描述。需重新啟動才會生效。"),
        "civitai_show_model_title_on_card": shared.OptionInfo(
            False,
            "以模型標題取代檔名顯示",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ).info("以中繼資料中的模型標題顯示卡片，搜尋仍使用檔名。需重新啟動才會生效。"),
    }
    for opt_name, opt_info in options.items():
        # Ensure OptionDiv has a section attribute set
        if (
            type(opt_info).__name__ == "OptionDiv"
            and getattr(opt_info, "section", None) is None
        ):
            opt_info.section = section
        shared.opts.add_option(opt_name, opt_info)
