"""Client-frame aligned multimodal preprocessing for Qwen-style pipelines.

This module adds a client payload path for pre-extracted video frames.  The key
alignment rule is: do not call the original ``fetch_image`` for each frame.
Frames are converted to uint8 RGB tensors, stacked as ``(T, C, H, W)``, padded
to ``FRAME_FACTOR`` if needed, and then passed through the same video-level
resize logic used by the original ``fetch_video`` implementation.
"""

from __future__ import annotations

import importlib
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence
from urllib.parse import unquote, urlparse


SAMPLE_RATE = 16000
SPATIAL_MERGE_SIZE = 2
FRAME_FACTOR = 2
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768
MODEL_SEQ_LEN = 32768


def _import_required(module_name: str, install_hint: str | None = None) -> Any:
    """Import an optional runtime dependency with an actionable error."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        hint = f" Install/enable {install_hint}." if install_hint else ""
        raise ImportError(f"Required dependency '{module_name}' is not available.{hint}") from exc


def _load_original_vision_module() -> Any | None:
    """Return the original Qwen vision module if it is importable."""
    candidates = (
        "qwen_vl_utils.vision_process",
        "qwen_omni_utils.vision_process",
        "qwen_omni_utils",
        "vision_process",
    )
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def _original_attr(name: str, default: Any = None) -> Any:
    module = _load_original_vision_module()
    if module is None:
        return default
    return getattr(module, name, default)


def _get_constant(name: str, default: Any) -> Any:
    return _original_attr(name, default)


def _image_factor(image_patch_size: int) -> int:
    spatial_merge_size = int(_get_constant("SPATIAL_MERGE_SIZE", SPATIAL_MERGE_SIZE))
    return int(image_patch_size) * spatial_merge_size


def _ceil_by_factor(value: int, factor: int) -> int:
    return int(math.ceil(value / factor) * factor)


def _smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Call original smart_resize when available, otherwise use a compatible fallback."""
    smart_resize = _original_attr("smart_resize")
    if smart_resize is not None:
        return smart_resize(height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)

    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got height={height}, width={width}")

    # Fallback mirrors the usual Qwen rule: preserve aspect ratio, constrain
    # pixels, and round dimensions to the model's spatial factor.
    current_pixels = height * width
    scale = 1.0
    if current_pixels > max_pixels:
        scale = math.sqrt(max_pixels / current_pixels)
    elif current_pixels < min_pixels:
        scale = math.sqrt(min_pixels / current_pixels)

    resized_height = max(factor, int(round(height * scale / factor)) * factor)
    resized_width = max(factor, int(round(width * scale / factor)) * factor)

    if resized_height * resized_width > max_pixels:
        scale = math.sqrt(max_pixels / (resized_height * resized_width))
        resized_height = max(factor, int(math.floor(resized_height * scale / factor)) * factor)
        resized_width = max(factor, int(math.floor(resized_width * scale / factor)) * factor)
    if resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (resized_height * resized_width))
        resized_height = max(factor, int(math.ceil(resized_height * scale / factor)) * factor)
        resized_width = max(factor, int(math.ceil(resized_width * scale / factor)) * factor)

    return int(resized_height), int(resized_width)


def _is_pil_image(value: Any) -> bool:
    try:
        image_module = importlib.import_module("PIL.Image")
    except ImportError:
        return False
    return isinstance(value, image_module.Image)


def _as_numpy_uint8_rgb_array(frame: Any) -> Any:
    """Convert a single frame to an RGB uint8 HWC numpy array without resizing."""
    np = _import_required("numpy", "numpy")
    torch = _import_required("torch", "torch")

    if _is_pil_image(frame):
        return np.asarray(frame.convert("RGB"), dtype=np.uint8)

    if isinstance(frame, torch.Tensor):
        array = frame.detach().cpu().numpy()
    elif isinstance(frame, np.ndarray):
        array = frame
    else:
        raise TypeError(
            "Each video frame must be PIL.Image, numpy.ndarray, or torch.Tensor; "
            f"got {type(frame)!r}"
        )

    if array.ndim != 3:
        raise ValueError(f"Each frame must be 3-dimensional HWC or CHW, got shape={array.shape}")

    # Accept HWC and CHW.  Ambiguous 3x3x3 frames are treated as HWC.
    if array.shape[-1] in (1, 3, 4):
        hwc = array
    elif array.shape[0] in (1, 3, 4):
        hwc = np.transpose(array, (1, 2, 0))
    else:
        raise ValueError(
            "Frame shape must be HWC or CHW with 1, 3, or 4 channels; "
            f"got shape={array.shape}"
        )

    if np.issubdtype(hwc.dtype, np.floating):
        # User explicitly requested float frames to be clamped to 0..255.
        hwc = np.clip(hwc, 0, 255).astype(np.uint8)
    elif hwc.dtype != np.uint8:
        hwc = np.clip(hwc, 0, 255).astype(np.uint8)

    channels = hwc.shape[-1]
    if channels == 1:
        hwc = np.repeat(hwc, 3, axis=-1)
    elif channels == 4:
        hwc = hwc[..., :3]
    elif channels != 3:
        raise ValueError(f"Frame must have 1, 3, or 4 channels, got {channels}")

    return np.ascontiguousarray(hwc)


def _split_video_batch(video: Any) -> list[Any]:
    """Normalize list/tuple/TCHW/THWC input into a list of frame-like objects."""
    np = _import_required("numpy", "numpy")
    torch = _import_required("torch", "torch")

    if isinstance(video, (list, tuple)):
        return list(video)

    if isinstance(video, torch.Tensor):
        if video.ndim == 3:
            return [video]
        if video.ndim != 4:
            raise ValueError(f"Video tensor must be CHW/HWC/TCHW/THWC, got shape={tuple(video.shape)}")
        return [video[i] for i in range(video.shape[0])]

    if isinstance(video, np.ndarray):
        if video.ndim == 3:
            return [video]
        if video.ndim != 4:
            raise ValueError(f"Video ndarray must be CHW/HWC/TCHW/THWC, got shape={video.shape}")
        return [video[i] for i in range(video.shape[0])]

    raise TypeError(
        "Client video payload must be list/tuple of frames, numpy.ndarray, or torch.Tensor; "
        f"got {type(video)!r}"
    )


def frames_to_video_tensor_no_fetch_image(
    frames: Sequence[Any] | Any,
    *,
    pad_to_frame_factor: bool = True,
) -> Any:
    """Convert client frames to a uint8 ``(T, C, H, W)`` torch tensor without resizing.

    PIL frames are converted to RGB.  NumPy arrays and tensors may be HWC, CHW,
    THWC, or TCHW.  Floating point inputs are clamped to ``[0, 255]`` and cast
    to uint8.  All frames must have the same spatial size after conversion.
    """
    np = _import_required("numpy", "numpy")
    torch = _import_required("torch", "torch")

    frame_list = _split_video_batch(frames)
    if not frame_list:
        raise ValueError("Client video payload contains no frames.")

    chw_tensors = []
    expected_shape: tuple[int, int, int] | None = None
    for index, frame in enumerate(frame_list):
        hwc = _as_numpy_uint8_rgb_array(frame)
        chw = np.transpose(hwc, (2, 0, 1))
        tensor = torch.from_numpy(np.ascontiguousarray(chw))
        shape = tuple(int(dim) for dim in tensor.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                "All video frames must have the same shape before video-level resize; "
                f"frame 0 has {expected_shape}, frame {index} has {shape}."
            )
        chw_tensors.append(tensor)

    frame_factor = int(_get_constant("FRAME_FACTOR", FRAME_FACTOR))
    if pad_to_frame_factor and len(chw_tensors) % frame_factor != 0:
        target_len = _ceil_by_factor(len(chw_tensors), frame_factor)
        chw_tensors.extend([chw_tensors[-1].clone() for _ in range(target_len - len(chw_tensors))])

    return torch.stack(chw_tensors, dim=0)


def _resolve_video_resize_shape(ele: Mapping[str, Any], video: Any, image_patch_size: int) -> tuple[int, int]:
    nframes, _channels, height, width = video.shape
    image_factor = _image_factor(image_patch_size)

    video_min_token_num = int(_get_constant("VIDEO_MIN_TOKEN_NUM", VIDEO_MIN_TOKEN_NUM))
    video_max_token_num = int(_get_constant("VIDEO_MAX_TOKEN_NUM", VIDEO_MAX_TOKEN_NUM))
    model_seq_len = int(_get_constant("MODEL_SEQ_LEN", MODEL_SEQ_LEN))
    video_frame_min_pixels = video_min_token_num * image_factor * image_factor
    video_frame_max_pixels = video_max_token_num * image_factor * image_factor
    video_total_pixels = int(
        _get_constant("VIDEO_TOTAL_PIXELS", model_seq_len * image_factor * image_factor * 0.9)
    )

    min_pixels = int(ele.get("min_pixels", video_frame_min_pixels))
    total_pixels = int(ele.get("total_pixels", video_total_pixels))
    default_max_pixels = max(
        min(video_frame_max_pixels, total_pixels / nframes * int(_get_constant("FRAME_FACTOR", FRAME_FACTOR))),
        int(min_pixels * 1.05),
    )
    max_pixels = int(ele.get("max_pixels", default_max_pixels))

    if "resized_height" in ele and "resized_width" in ele:
        resized_height = int(ele["resized_height"])
        resized_width = int(ele["resized_width"])
        return _smart_resize(
            resized_height,
            resized_width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    return _smart_resize(
        int(height),
        int(width),
        factor=image_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def _resize_video_tensor_bicubic(video: Any, resized_height: int, resized_width: int) -> Any:
    """Apply the original video-level torchvision bicubic resize."""
    transforms = _import_required("torchvision.transforms", "torchvision")
    functional = _import_required("torchvision.transforms.functional", "torchvision")
    interpolation_mode = getattr(transforms, "InterpolationMode", getattr(functional, "InterpolationMode"))
    return functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=interpolation_mode.BICUBIC,
        antialias=True,
    ).float()


def fetch_video_from_frames_aligned(
    ele: Mapping[str, Any],
    image_patch_size: int = 14,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
) -> Any:
    """Fetch client-supplied frames using the aligned no-``fetch_image`` path.

    The returned tensor and optional sample FPS / metadata are compatible with
    the original ``fetch_video`` contract.  Only one resize is performed, after
    frames have been stacked as video.
    """
    if "video" not in ele:
        raise KeyError("Video element must contain a 'video' field.")

    torch = _import_required("torch", "torch")
    video = frames_to_video_tensor_no_fetch_image(ele["video"], pad_to_frame_factor=True)
    if video.ndim != 4:
        raise ValueError(f"Aligned video tensor must be TCHW, got shape={tuple(video.shape)}")
    if video.shape[1] != 3:
        raise ValueError(f"Aligned video tensor must have 3 RGB channels, got shape={tuple(video.shape)}")

    resized_height, resized_width = _resolve_video_resize_shape(ele, video, image_patch_size)
    video = _resize_video_tensor_bicubic(video, resized_height, resized_width)

    sample_fps = float(ele.get("sample_fps", 2.0))
    metadata = {
        "fps": ele.get("raw_fps", ele.get("fps")),
        "frames_indices": ele.get("frames_indices"),
        "total_num_frames": ele.get("total_num_frames"),
        "video_backend": ele.get("video_backend", "client_frames"),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}

    outputs: list[Any] = [video]
    if return_video_sample_fps:
        outputs.append(sample_fps)
    if return_video_metadata:
        outputs.append(metadata)
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def _local_path_from_uri(path_or_uri: str | os.PathLike[str]) -> str:
    text = os.fspath(path_or_uri)
    parsed = urlparse(text)
    if parsed.scheme == "file":
        if parsed.netloc and parsed.netloc not in ("localhost", ""):
            raise ValueError(f"Only local file:// audio URIs are supported, got {text!r}")
        return unquote(parsed.path.lstrip("/") if os.name == "nt" and parsed.path.startswith("/") else parsed.path)
    if parsed.scheme and len(parsed.scheme) > 1:
        raise ValueError(f"Only local paths and file:// URIs are supported for audio, got {text!r}")
    return text


def _to_mono_audio(audio: Any, *, downmix: bool) -> Any:
    np = _import_required("numpy", "numpy")
    array = np.asarray(audio)
    if array.ndim == 1:
        return array.astype(np.float32, copy=False)
    if array.ndim == 2:
        # Accept both (channels, samples) and (samples, channels).
        channel_axis = 0 if array.shape[0] <= 8 and array.shape[0] < array.shape[1] else 1
        channels = array.shape[channel_axis]
        if channels == 1:
            return np.squeeze(array, axis=channel_axis).astype(np.float32, copy=False)
        if not downmix:
            raise ValueError(
                "Audio numpy.ndarray must be mono. Pass downmix_to_mono=True to average channels."
            )
        return np.mean(array, axis=channel_axis).astype(np.float32, copy=False)
    raise ValueError(f"Audio array must be 1D mono or 2D channels/samples, got shape={array.shape}")


def _resample_audio_array(audio: Any, source_sample_rate: int, target_sample_rate: int) -> Any:
    if int(source_sample_rate) == int(target_sample_rate):
        return audio
    librosa = _import_required("librosa", "librosa")
    return librosa.resample(audio, orig_sr=int(source_sample_rate), target_sr=int(target_sample_rate))


def load_audio_aligned(
    audio: str | os.PathLike[str] | Any,
    *,
    sample_rate: int = SAMPLE_RATE,
    source_sample_rate: int | None = None,
    downmix_to_mono: bool = False,
) -> Any:
    """Load a client audio payload as mono float32 numpy array at ``sample_rate``.

    ``audio`` may be a local path, ``file://`` URI, or numpy array.  NumPy input
    must be mono unless ``downmix_to_mono=True`` is provided.  When numpy input
    needs resampling, ``source_sample_rate`` must be supplied.
    """
    np = _import_required("numpy", "numpy")

    if isinstance(audio, (str, os.PathLike)):
        path = _local_path_from_uri(audio)
        if not Path(path).exists():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        librosa = _import_required("librosa", "librosa")
        loaded, _sr = librosa.load(path, sr=int(sample_rate), mono=True)
        return np.asarray(loaded, dtype=np.float32)

    if isinstance(audio, np.ndarray):
        mono = _to_mono_audio(audio, downmix=downmix_to_mono)
        if source_sample_rate is not None:
            mono = _resample_audio_array(mono, source_sample_rate, sample_rate)
        return np.asarray(mono, dtype=np.float32)

    raise TypeError(
        "Audio payload must be a local path, file:// URI, or numpy.ndarray; "
        f"got {type(audio)!r}"
    )


def _iter_content_items(conversations: Sequence[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for message in conversations:
        content = message.get("content", [])
        if isinstance(content, Mapping):
            yield content
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping):
                    yield item


def _is_client_frame_video(video_value: Any) -> bool:
    if isinstance(video_value, (list, tuple)):
        return True
    try:
        np = importlib.import_module("numpy")
        if isinstance(video_value, np.ndarray):
            return True
    except ImportError:
        pass
    try:
        torch = importlib.import_module("torch")
        if isinstance(video_value, torch.Tensor):
            return True
    except ImportError:
        pass
    return False


def process_mm_info_client_aligned(
    conversations: Sequence[Mapping[str, Any]],
    image_patch_size: int = 14,
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    sample_rate: int = SAMPLE_RATE,
    load_audio_from_video: bool = False,
) -> tuple[Any, ...]:
    """Process image/video/audio inputs with an aligned client-frame video path.

    Images and string video paths are delegated to the original Qwen functions
    when available.  Client frame videos are handled by
    ``fetch_video_from_frames_aligned`` and therefore skip per-frame
    ``fetch_image`` resizing.  When ``load_audio_from_video=True``, an ``audio``
    field attached to a video item is loaded as the audio track for that video;
    this is useful for matching Qwen3-Omni's ``use_audio_in_video=True`` call
    shape without adding a separate audio item to the chat template.
    """
    image_inputs: list[Any] = []
    video_inputs: list[Any] = []
    audio_inputs: list[Any] = []
    video_sample_fps: list[float] = []
    video_metadata: list[dict[str, Any]] = []

    original_fetch_image = _original_attr("fetch_image")
    original_fetch_video = _original_attr("fetch_video")

    def call_original_fetch_image(item: Mapping[str, Any]) -> Any:
        assert original_fetch_image is not None
        try:
            return original_fetch_image(dict(item), image_patch_size=image_patch_size)
        except TypeError:
            return original_fetch_image(dict(item), _image_factor(image_patch_size))

    for item in _iter_content_items(conversations):
        item_type = item.get("type")
        if item_type == "image" or "image" in item:
            if original_fetch_image is None:
                raise ImportError("Original fetch_image is required to process image inputs.")
            image_inputs.append(call_original_fetch_image(item))
            continue

        if item_type == "video" or "video" in item:
            video_value = item.get("video")
            if _is_client_frame_video(video_value):
                result = fetch_video_from_frames_aligned(
                    item,
                    image_patch_size=image_patch_size,
                    return_video_sample_fps=True,
                    return_video_metadata=True,
                )
            else:
                if original_fetch_video is None:
                    raise ImportError("Original fetch_video is required to process string video paths.")
                result = original_fetch_video(
                    dict(item),
                    image_patch_size=image_patch_size,
                    return_video_sample_fps=True,
                    return_video_metadata=True,
                )
            video, sample_fps_value, metadata = result
            video_inputs.append(video)
            video_sample_fps.append(float(sample_fps_value))
            video_metadata.append(dict(metadata or {}))
            if load_audio_from_video and "audio" in item:
                audio_inputs.append(load_audio_aligned(item["audio"], sample_rate=sample_rate))
            continue

        if item_type == "audio" or "audio" in item:
            audio_inputs.append(load_audio_aligned(item["audio"], sample_rate=sample_rate))

    outputs: list[Any] = [image_inputs or None, video_inputs or None, audio_inputs or None]
    if return_video_kwargs:
        outputs.append({"fps": video_sample_fps})
    if return_video_metadata:
        outputs.append(video_metadata)
    return tuple(outputs)


def process_mm_info_client_aligned_qwen3_omni(
    conversations: Sequence[Mapping[str, Any]],
    image_patch_size: int = 14,
    use_audio_in_video: bool = True,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[Any, Any, Any]:
    """Return client-aligned inputs in the official Qwen3-Omni order.

    The official Transformers example uses::

        audios, images, videos = process_mm_info(...)

    This adapter keeps that call shape while using the aligned client-frame
    video path from ``process_mm_info_client_aligned``.  ``use_audio_in_video``
    is accepted for API symmetry with ``qwen_omni_utils.process_mm_info``; in
    the client-frame deployment path the audio is expected to be provided as an
    explicit ``{"type": "audio", "audio": ...}`` item, or already embedded in
    the caller's preprocessing pipeline.
    """
    del use_audio_in_video

    images, videos, audios = process_mm_info_client_aligned(
        conversations,
        image_patch_size=image_patch_size,
        return_video_kwargs=False,
        return_video_metadata=False,
        sample_rate=sample_rate,
        load_audio_from_video=use_audio_in_video,
    )
    return audios, images, videos
