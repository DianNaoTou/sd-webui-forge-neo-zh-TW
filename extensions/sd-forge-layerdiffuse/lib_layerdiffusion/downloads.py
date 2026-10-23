"""Only selected LayerDiffusion weights are downloaded, pinned and SHA-256 checked."""
import functools
import hashlib
import json
from pathlib import Path


MANIFEST = json.loads(Path(__file__).with_name('weights.json').read_text(encoding='utf-8'))
REVISION = MANIFEST['revision']


@functools.lru_cache(maxsize=32)
def checked_digest(path, size, mtime_ns, ctime_ns):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_file_from_url(*, url, model_dir, file_name):
    from modules.util import load_file_from_url as forge_download

    if file_name not in MANIFEST['files']:
        raise ValueError('未列入清單的 LayerDiffuse 權重，已拒絕下載。')
    expected_url = f"https://huggingface.co/{MANIFEST['repository']}/resolve/{REVISION}/{file_name}"
    if url != expected_url:
        raise ValueError('LayerDiffuse 權重來源或版本不符，已拒絕下載。')
    entry = MANIFEST['files'][file_name]
    path = forge_download(url, model_dir=model_dir, file_name=file_name, hash_prefix=entry['sha256'])
    stat = Path(path).stat()
    if stat.st_size != entry['size'] or checked_digest(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != entry['sha256']:
        raise ValueError(f'圖層權重驗證失敗：{file_name}。原檔已保留，請先移開該檔再重試。')
    return path
