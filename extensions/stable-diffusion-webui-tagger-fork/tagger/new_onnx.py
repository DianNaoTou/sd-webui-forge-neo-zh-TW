"""Lazy, pinned ONNX adapters; no downloads or session creation at import time.

PixAI preprocessing follows DeepGHS' exported preprocess.json. Camie follows
Camais03's onnx_inference.py. Existing WD14 adapters are deliberately separate.
"""
import csv
import json
from pathlib import Path
from threading import RLock

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from modules import shared
from tagger import interrogator as legacy


class TaggerModelError(RuntimeError):
    """An actionable failure confined to the selected new model."""


def normalize_image(image, size, mean, std, *, letterbox=False):
    image = image.convert('RGB')
    if letterbox:
        width, height = image.size
        ratio = width / height
        if ratio > 1:
            dimensions = (size, max(1, int(size / ratio)))
        else:
            dimensions = (max(1, int(size * ratio)), size)
        image = image.resize(dimensions, Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (size, size), (124, 116, 104))
        canvas.paste(image, ((size - dimensions[0]) // 2,
                            (size - dimensions[1]) // 2))
        image = canvas
    else:
        image = image.resize((size, size), Image.Resampling.BILINEAR, reducing_gap=1.0)
    values = np.asarray(image, dtype=np.float32) / np.float32(255)
    values = (values - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])


class LazyOnnxInterrogator(legacy.Interrogator):
    """Reuse the legacy UI, filters, batch loop and unload controls."""
    require_submit = True
    output_name = ''
    size = 0
    count = 0

    def __init__(self, name, repo_id, revision, files):
        super().__init__(name)
        self.repo_id, self.revision, self.files = repo_id, revision, files
        # The inherited legacy accelerated batch method is TensorFlow-specific.
        # Route both batch modes through the existing per-image ONNX batch loop.
        self.run_mode = 2
        self._lock = RLock()

    def _download(self, filename):
        cache = getattr(shared.opts, 'tagger_hf_cache_dir', legacy.Its.hf_cache)
        options = dict(repo_id=self.repo_id, revision=self.revision,
                       filename=filename, cache_dir=cache)
        try:
            return hf_hub_download(**options, local_files_only=True)
        except FileNotFoundError:
            pass
        except Exception as exc:
            # HF versions differ in their local-cache-miss exception hierarchy.
            from huggingface_hub.errors import LocalEntryNotFoundError
            if not isinstance(exc, LocalEntryNotFoundError):
                raise
        print(f'WD14：首次使用 {self.name}，正在下載 {filename}；大型權重可能需要數分鐘。')
        return hf_hub_download(**options)

    def _metadata(self, paths):
        raise NotImplementedError

    def _session(self, model_path):
        ort = legacy.get_onnxrt()
        available = ort.get_available_providers()
        providers = [p for p in legacy.onnxrt_providers if p in available]
        if 'CPUExecutionProvider' not in providers:
            providers.append('CPUExecutionProvider')
        try:
            return ort.InferenceSession(model_path, providers=providers)
        except Exception:
            if providers == ['CPUExecutionProvider']:
                raise
            print(f'WD14：{self.name} GPU 載入失敗，改用 CPU 推論。')
            return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def load(self):
        with self._lock:
            if self.model is not None:
                return
            try:
                # Small metadata is checked before downloading the large weights.
                paths = {f: self._download(f) for f in self.files[1:]}
                tags = self._metadata(paths)
                session = self._session(self._download(self.files[0]))
                inputs = session.get_inputs()
                if len(inputs) != 1 or inputs[0].type != 'tensor(float)':
                    raise ValueError('不支援的模型輸入格式')
                if list(inputs[0].shape) != [1, 3, self.size, self.size]:
                    # Exporters can leave the batch dimension symbolic.
                    shape = inputs[0].shape
                    if len(shape) != 4 or shape[0] not in (1, None, 'batch_size', 'batch') or list(shape[1:]) != [3, self.size, self.size]:
                        raise ValueError(f'模型輸入尺寸不符：{shape}')
                output = next((o for o in session.get_outputs() if o.name == self.output_name), None)
                if output is None or len(output.shape) != 2 or output.shape[1] != self.count:
                    raise ValueError('模型輸出名稱或標籤數量不符')
                # Publish the session only after metadata and contracts succeed.
                self.tags, self.model = tags, session
                print(f'WD14：已載入 {self.name}，執行後端：{session.get_providers()}')
            except Exception as exc:
                self.model, self.tags = None, None
                raise TaggerModelError(
                    f'{self.name} 下載或載入失敗（{type(exc).__name__}）。'
                    '請檢查網路、磁碟空間與 Hugging Face 存取權限後重試；'
                    '已下載檔案會保留於既有快取，也可切換其他模型。'
                ) from exc

    def interrogate(self, image):
        with self._lock:
            self.load()
            try:
                values = self.preprocess(image)
                scores = np.asarray(self.model.run(
                    [self.output_name], {self.model.get_inputs()[0].name: values})[0])
                if scores.shape != (1, self.count) or not np.isfinite(scores).all():
                    raise ValueError('辨識輸出尺寸不符或含有非有限數值')
                probabilities = self.probabilities(scores[0])
                ratings, tags = {}, {}
                for (name, category), value in zip(self.tags, probabilities):
                    (ratings if category == 'rating' else tags)[name] = float(value)
                return ratings, tags
            except Exception as exc:
                self.unload()
                raise TaggerModelError(
                    f'{self.name} 辨識失敗（{type(exc).__name__}）；'
                    '模型已卸載，可重試或切換其他模型。'
                ) from exc

    def unload(self):
        with self._lock:
            return super().unload()

    @staticmethod
    def probabilities(scores):
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError('模型信心度超出範圍')
        return scores


class PixAIInterrogator(LazyOnnxInterrogator):
    output_name, size, count = 'prediction', 448, 13461

    def __init__(self):
        super().__init__('PixAI Tagger v0.9', 'deepghs/pixai-tagger-v0.9-onnx',
                         'd8cf666911a2c3d10d586d7823259192313c7eb7',
                         ('model.onnx', 'preprocess.json', 'selected_tags.csv'))

    def _metadata(self, paths):
        stages = json.loads(Path(paths['preprocess.json']).read_text(encoding='utf-8'))['stages']
        expected = [('resize', [448, 448]), ('to_tensor', None), ('normalize', None)]
        if [(s['type'], s.get('size')) for s in stages] != expected or stages[0]['interpolation'] != 'bilinear' or stages[2]['mean'] != [0.5]*3 or stages[2]['std'] != [0.5]*3:
            raise ValueError('PixAI 前處理設定不符')
        with open(paths['selected_tags.csv'], encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != self.count or [int(r['id']) for r in rows] != list(range(self.count)):
            raise ValueError('PixAI 標籤索引不完整')
        if any(r['category'] not in ('0', '4') for r in rows):
            raise ValueError('PixAI 標籤分類不符')
        return [(r['name'], 'character' if r['category'] == '4' else 'general') for r in rows]

    def preprocess(self, image):
        # DeepGHS load_image(force_background='white') composites alpha first.
        rgba = image.convert('RGBA')
        background = Image.new('RGBA', rgba.size, 'white')
        image = Image.alpha_composite(background, rgba).convert('RGB')
        return normalize_image(image, self.size, [0.5]*3, [0.5]*3)


class CamieInterrogator(LazyOnnxInterrogator):
    output_name, size, count = 'refined_predictions', 512, 70527

    def __init__(self):
        super().__init__('Camie Tagger v2', 'Camais03/camie-tagger-v2',
                         '7d40c1b85b86ab4f607b2caf26b1b50c99db743e',
                         ('camie-tagger-v2.onnx', 'camie-tagger-v2-metadata.json'))

    def _metadata(self, paths):
        metadata = json.loads(Path(paths[self.files[1]]).read_text(encoding='utf-8'))
        data = metadata['dataset_info']
        mapping = data['tag_mapping']
        indices, categories = mapping['idx_to_tag'], mapping['tag_to_category']
        if metadata['model_info']['img_size'] != self.size or data['total_tags'] != self.count or set(indices) != {str(i) for i in range(self.count)}:
            raise ValueError('Camie 標籤索引或尺寸不符')
        allowed = {'general', 'character', 'copyright', 'artist', 'meta', 'year', 'rating'}
        tags = [(indices[str(i)], categories[indices[str(i)]]) for i in range(self.count)]
        if any(category not in allowed for _, category in tags):
            raise ValueError('Camie 標籤分類不符')
        return tags

    def preprocess(self, image):
        return normalize_image(image, self.size, [0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225], letterbox=True)

    @staticmethod
    def probabilities(scores):
        # Stable sigmoid: Camie's output is logits, not WD14 probabilities.
        return np.exp(-np.logaddexp(0, -scores))
