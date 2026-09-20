import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Optional, TypeVar

from PIL import Image, ImageDraw
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

from filelock import FileLock
from tqdm import tqdm

from modules.shared import cmd_opts

from ..utils import NUM, print

URL = TypeVar("URL", bound=str)

no_huggingface: bool = getattr(cmd_opts, "ad_no_huggingface", False)


@dataclass
class PredictOutput:
    bboxes: list[tuple[NUM]] = field(default_factory=tuple)
    masks: list[Image.Image] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    preview: Optional[Image.Image] = None


def _scan_models(path: Path) -> list[Path]:
    return [
        obj
        for obj in path.rglob("*")
        if (obj.is_file() and obj.stat().st_size > 0 and obj.suffix in (".pt", ".tflite", ".task"))
    ]


def _download_model(url: URL, filename: os.PathLike):
    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    # A second WebUI process must not delete an active download's partial file.
    with FileLock(str(target) + ".lock", timeout=120):
        for stale in target.parent.glob(target.name + ".*.partial"):
            stale.unlink()
            print(f'Removed interrupted download: {stale.name}')
        if target.is_file() and target.stat().st_size > 0:
            return
        print(f'Downloading model: {target.name}')
        temporary = None
        try:
            request = Request(str(url), headers={"User-Agent": "Forge-Neo-ADetailer"})
            # Bound connection/read stalls, including before the first byte.
            with urlopen(request, timeout=30) as response:
                total = int(response.headers.get("Content-Length", "0")) or None
                with NamedTemporaryFile(dir=target.parent, prefix=target.name + ".",
                                        suffix=".partial", delete=False) as output:
                    temporary = Path(output.name)
                    received = 0
                    with tqdm(total=total, desc=target.name, unit="B", unit_scale=True,
                              mininterval=1.0) as progress:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                            received += len(chunk)
                            progress.update(len(chunk))
                    if not received or (total is not None and received != total):
                        raise OSError(f"Incomplete download: {received}/{total} bytes")
                    output.flush()
                    os.fsync(output.fileno())
            os.replace(temporary, target)
            print(f'Download complete: {target.name} ({received:,} bytes)')
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _download(folder: os.PathLike, names: dict[str, URL]):
    # One named progress bar at a time stays readable in Windows terminals.
    for file, url in names.items():
        try:
            _download_model(url, os.path.join(folder, file))
        except Exception as exc:
            print(f'Download failed: {file}: {type(exc).__name__}: {exc}. '
                  'The next launch will retry this model.')


def get_models(ad_dir: str, *extra_dirs: str) -> dict[str, os.PathLike]:

    TO_DOWNLOAD: Final[dict[str, URL]] = {
        # https://huggingface.co/Bingsu/adetailer
        "face_yolov8n.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt?download=true",
        "face_yolov8s.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt?download=true",
        "hand_yolov8n.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt?download=true",
        "hand_yolov8s.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt?download=true",
        "person_yolov8n-seg.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8n-seg.pt?download=true",
        "person_yolov8s-seg.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8s-seg.pt?download=true",
        # https://github.com/ultralytics/assets
        "yolov8x-worldv2.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-worldv2.pt",
        # https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector#models
        "mediapipe_face_short.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        "mediapipe_face_full.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite",
        # https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models
        "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    }

    if not no_huggingface:
        print("Loading Models...")
        _download(ad_dir, TO_DOWNLOAD)

    models: dict[str, os.PathLike] = {}
    model_paths: list[Path] = []

    for _dir in (ad_dir, *extra_dirs):
        if not os.path.isdir(_dir):
            continue
        model_paths.extend(_scan_models(Path(_dir)))

    for path in model_paths:
        if path.name in models:
            continue
        models[path.name] = str(path)

    return models


# region BBox / Mask


def create_mask_from_bbox(
    bboxes: list[tuple[int, int, int, int]], shape: tuple[int, int]
) -> list[Image.Image]:
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def create_bbox_from_mask(
    masks: list[Image.Image], shape: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    bboxes = []
    for mask in masks:
        mask = mask.resize(shape)
        bbox = mask.getbbox()
        if bbox is not None:
            bboxes.append(list(bbox))
    return bboxes
