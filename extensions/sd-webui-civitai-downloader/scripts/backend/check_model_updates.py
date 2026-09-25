import os
import requests
import json
from .utils import get_model_folders, get_civitai_api_key, get_civitai_domains
from .process_control import is_running, set_running, clear_running, cancel_process, is_cancelled, get_type

def get_latest_model_info(model_id, api_key=None):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_exc = None
    got_404 = False
    for domain in get_civitai_domains():
        try:
            resp = requests.get(f"https://{domain}/api/v1/models/{model_id}", headers=headers)
            if resp.status_code == 404:
                got_404 = True
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            continue
    if got_404 and last_exc is None:
        raise ValueError(f"Model {model_id} not found on Civitai (404)")
    if last_exc:
        raise last_exc
    return None

def cancel_check_model_updates():
    cancel_process()

def check_model_updates():
    if is_running():
        yield f"另一個作業正在執行：{get_type()}"
        return
    set_running('updates')
    try:
        MODEL_FOLDERS = get_model_folders()
        skip_types = {"Controlnet", "Upscaler", "VAE"}
        files_to_check = []
        # Deduplicate folders to avoid scanning the same folder multiple times
        unique_folders = set()
        for model_type, folder in MODEL_FOLDERS.items():
            if model_type in skip_types:
                continue
            abs_folder = os.path.abspath(folder)
            if not os.path.exists(abs_folder):
                continue
            unique_folders.add(abs_folder)
        # Only process files with .metadata.json
        for abs_folder in unique_folders:
            for root, dirs, files in os.walk(abs_folder):
                for file in files:
                    if not (file.lower().endswith(('.safetensors', '.ckpt', '.pt'))):
                        continue
                    base = os.path.splitext(file)[0]
                    metadata_path = os.path.join(root, base + '.metadata.json')
                    if not os.path.exists(metadata_path):
                        continue
                    files_to_check.append((root, file, metadata_path))
        total = len(files_to_check)
        if total == 0:
            yield "沒有可檢查更新的模型。"
            return
        api_key = get_civitai_api_key()
        updates = []
        errors = []
        for idx, (_, file, metadata_path) in enumerate(files_to_check, 1):
            if is_cancelled():
                yield '\n\n'.join(updates + errors + [f"已取消；檢查 {idx-1}/{total} 個檔案。"])
                return
            base = os.path.splitext(file)[0]
            status = f"[{idx}/{total}] 檢查中：{file}"
            yield '\n\n'.join(updates + errors + [status])
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                # Try standard Civitai format first
                model_id = meta.get('id')
                current_version_id = None
                if 'modelVersions' in meta and meta['modelVersions']:
                    mv = meta['modelVersions'][0]
                    current_version_id = mv.get('id')
                # If not found, try Civitai Helper format
                if (not model_id or not current_version_id) and 'civitai' in meta:
                    civ = meta['civitai']
                    model_id = civ.get('modelId')
                    current_version_id = civ.get('id')
                if not model_id or not current_version_id:
                    msg = f"無法判斷模型 ID 或版本：{file}"
                    errors.append(msg)
                    yield '\n\n'.join(updates + errors)
                    continue
                # Get latest model info from Civitai
                latest_info = get_latest_model_info(model_id, api_key=api_key)
                latest_versions = latest_info.get('modelVersions', [])
                if not latest_versions:
                    msg = f"找不到模型版本：{model_id} ({file})"
                    errors.append(msg)
                    yield msg
                    continue
                latest_version = latest_versions[0]
                latest_version_id = latest_version.get('id')
                if str(current_version_id) == str(latest_version_id):
                    # Up to date
                    continue
                # New version available (Markdown link)
                model_name = latest_info.get('name', f'Model {model_id}')
                domain = get_civitai_domains()[0]
                url = f"https://{domain}/models/{model_id}?modelVersionId={latest_version_id}"
                update_msg = f"{model_name} 有新版本：[[在瀏覽器開啟]]({url})"
                updates.append(update_msg)
                yield '\n\n'.join(updates + errors)
            except Exception as e:
                msg = f"檢查失敗：{file}: {str(e)}"
                errors.append(msg)
                yield '\n\n'.join(updates + errors)
        # Final summary
        if not updates and not errors:
            yield "所有模型都已是最新版本。"
        elif updates:
            yield '\n\n'.join(updates + errors + [f"檢查完成，{len(updates)} 個模型有可用更新。"])
        elif errors and not updates:
            yield '\n\n'.join(errors + [f"檢查完成，沒有更新；發生 {len(errors)} 個錯誤。"])
    finally:
        clear_running()
