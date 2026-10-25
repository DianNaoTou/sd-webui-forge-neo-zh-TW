from modules.localization import translate as ui_text
import os
import hashlib
import requests
from .utils import get_model_folders, get_civitai_api_key, get_civitai_domains, save_preview_and_metadata
from .process_control import is_running, set_running, clear_running, cancel_process, is_cancelled, get_type

def sha256_of_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def get_model_info_by_hash(file_hash, api_key=None):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    last_exc = None
    for domain in get_civitai_domains():
        try:
            resp = requests.get(f"https://{domain}/api/v1/model-versions/by-hash/{file_hash}", headers=headers)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_exc = e
            continue
    return None  # Not found on any domain

def cancel_check_missing_info():
    cancel_process()

def check_missing_info():
    if is_running():
        yield ui_text('Another operation is running: {0}', get_type())
        return
    set_running('missing_info')
    try:
        MODEL_FOLDERS = get_model_folders()
        summary = []
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
        # First pass: build a list of files with any missing info
        for abs_folder in unique_folders:
            for root, dirs, files in os.walk(abs_folder):
                for file in files:
                    if not (file.lower().endswith(('.safetensors', '.ckpt', '.pt'))):
                        continue
                    base = os.path.splitext(file)[0]
                    metadata_path = os.path.join(root, base + '.metadata.json')
                    preview_found = False
                    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                        if os.path.exists(os.path.join(root, base + f'.preview{ext}')):
                            preview_found = True
                            break
                    missing = []
                    if not os.path.exists(metadata_path):
                        missing.append(ui_text('Metadata'))
                    if not preview_found:
                        missing.append(ui_text('Preview'))
                    if missing:
                        files_to_check.append((root, file, missing))
        total = len(files_to_check)
        if total == 0:
            yield ui_text('All models have metadata and previews.')
            return
        # Second pass: process each file only once
        for idx, (root, file, missing) in enumerate(files_to_check, 1):
            if is_cancelled():
                yield ui_text('Cancelled; processed {0}/{1} files.', idx - 1, total)
                return
            processed = idx
            status = ui_text('[{0}/{1}] Processing: {2} (missing: {3})', processed, total, file, ', '.join(missing))
            print(status)
            yield '\n'.join(summary + [status])
            try:
                base = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                hash_path = os.path.join(root, base + '.sha256')
                # Use cached hash if available, else calculate and save
                if os.path.exists(hash_path):
                    with open(hash_path, 'r', encoding='utf-8') as hf:
                        file_hash = hf.read().strip()
                else:
                    file_hash = sha256_of_file(file_path)
                    with open(hash_path, 'w', encoding='utf-8') as hf:
                        hf.write(file_hash)
                api_key = get_civitai_api_key()
                model_version_info = get_model_info_by_hash(file_hash, api_key=api_key)
                if not model_version_info:
                    msg = ui_text('Skipped: {0} ({1}) - no Civitai model matches the SHA256', file, ', '.join(missing))
                    print(msg)
                    summary.append(msg)
                    yield '\n'.join(summary + [f"[{processed}/{total}] ..."])
                    continue
                # Break circular reference before saving
                if 'model' in model_version_info:
                    del model_version_info['model']
                model_info = model_version_info.get('model', {})
                model_info['modelVersions'] = [model_version_info]
                # Ensure model id is present in metadata
                model_info['id'] = model_version_info.get('modelId')
                preview_url = None
                if model_version_info.get('images'):
                    preview_url = model_version_info['images'][0]['url']
                save_preview_and_metadata(root, file, model_info, preview_url, model_version_info)
                msg = ui_text('Updated: {0} ({1})', file, ', '.join(missing))
                print(msg)
                summary.append(msg)
                yield '\n\n'.join(summary + [f"[{processed}/{total}] ..."])
            except Exception as e:
                msg = ui_text('Processing failed: {0} ({1}) - {2}', file, ', '.join(missing), str(e))
                print(msg)
                summary.append(msg)
                yield '\n\n'.join(summary + [f"[{processed}/{total}] ..."])
        yield '\n\n'.join(summary)
    finally:
        clear_running()
