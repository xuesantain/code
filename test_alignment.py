"""Alignment test helpers for direct video path vs client frame+audio payload."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

from mm_aligned import (
    SAMPLE_RATE,
    _import_required,
    _load_original_vision_module,
    fetch_video_from_frames_aligned,
    load_audio_aligned,
)


def _require_original_vision_module() -> Any:
    module = _load_original_vision_module()
    if module is None:
        raise ImportError(
            "Could not import the original Qwen vision module. Expected one of "
            "qwen_vl_utils.vision_process, qwen_omni_utils.vision_process, "
            "qwen_omni_utils, or vision_process."
        )
    return module


def _call_original_video_backend(ele: Mapping[str, Any]) -> tuple[Any, dict[str, Any], float]:
    """Call the original video reader backend and normalize its return shape."""
    module = _require_original_vision_module()
    backend_name = module.get_video_reader_backend()
    backend = module.VIDEO_READER_BACKENDS[backend_name]
    result = backend(dict(ele))

    if not isinstance(result, tuple):
        raise TypeError(f"Original video backend returned {type(result)!r}, expected tuple.")
    if len(result) == 3:
        raw_video, metadata, sample_fps = result
    elif len(result) == 2:
        raw_video, sample_fps = result
        metadata = {}
    else:
        raise ValueError(f"Original video backend returned {len(result)} values, expected 2 or 3.")

    metadata = dict(metadata or {})
    metadata.setdefault("video_backend", backend_name)
    return raw_video, metadata, float(sample_fps)


def _normalize_fetch_video_result(result: Any) -> tuple[Any, float, dict[str, Any]]:
    if not isinstance(result, tuple):
        return result, math.nan, {}
    if len(result) == 3:
        video, sample_fps, metadata = result
        return video, float(sample_fps), dict(metadata or {})
    if len(result) == 2:
        video, sample_fps = result
        return video, float(sample_fps), {}
    raise ValueError(f"fetch_video returned {len(result)} values, expected 1, 2, or 3.")


def _tensor_to_uint8_tchw(video: Any) -> Any:
    torch = _import_required("torch", "torch")
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor video, got {type(video)!r}")
    if video.ndim != 4:
        raise ValueError(f"Expected TCHW video tensor, got shape={tuple(video.shape)}")
    if video.dtype.is_floating_point:
        return video.detach().cpu().clamp(0, 255).to(torch.uint8)
    return video.detach().cpu().clamp(0, 255).to(torch.uint8)


def tensor_video_to_pil_frames(video: Any) -> list[Any]:
    """Convert a ``(T, C, H, W)`` tensor to RGB PIL frames without JPEG loss."""
    np = _import_required("numpy", "numpy")
    image_module = _import_required("PIL.Image", "Pillow")
    tensor = _tensor_to_uint8_tchw(video)
    if tensor.shape[1] != 3:
        raise ValueError(f"Expected RGB TCHW tensor, got shape={tuple(tensor.shape)}")
    frames = []
    for frame in tensor:
        hwc = frame.permute(1, 2, 0).contiguous().numpy()
        frames.append(image_module.fromarray(np.asarray(hwc, dtype=np.uint8), mode="RGB"))
    return frames


def tensor_video_to_numpy_frames(video: Any) -> list[Any]:
    """Convert a ``(T, C, H, W)`` tensor to RGB uint8 HWC numpy frames."""
    tensor = _tensor_to_uint8_tchw(video)
    return [frame.permute(1, 2, 0).contiguous().numpy() for frame in tensor]


def tensor_video_to_tensor_frames(video: Any) -> list[Any]:
    """Convert a ``(T, C, H, W)`` tensor to a list of CHW uint8 tensor frames."""
    tensor = _tensor_to_uint8_tchw(video)
    return [frame.clone() for frame in tensor]


def extract_audio_from_video_like_original(
    video_path: str | Path,
    ele: Mapping[str, Any] | None = None,
    *,
    output_audio_path: str | Path | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> Any:
    """Extract audio using the original ``use_audio_in_video=True`` timing rule.

    ``video_start`` is mapped to librosa ``offset``.  ``video_end`` defines
    ``duration = video_end - video_start`` when provided.
    """
    np = _import_required("numpy", "numpy")
    librosa = _import_required("librosa", "librosa")

    ele = dict(ele or {})
    video_start = float(ele.get("video_start", 0.0) or 0.0)
    video_end = ele.get("video_end")
    duration = None
    if video_end is not None:
        duration = max(0.0, float(video_end) - video_start)

    audio, _sr = librosa.load(
        str(video_path),
        sr=int(sample_rate),
        mono=True,
        offset=video_start,
        duration=duration,
    )
    audio = np.asarray(audio, dtype=np.float32)

    if output_audio_path is not None:
        soundfile = _import_required("soundfile", "soundfile")
        output_path = Path(output_audio_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(str(output_path), audio, int(sample_rate), subtype="FLOAT")

    return audio


def simulate_client_payload_from_video(
    video_path: str | Path,
    ele: Mapping[str, Any] | None = None,
    output_audio_path: str | Path | None = None,
    image_patch_size: int = 14,
    sample_rate: int = SAMPLE_RATE,
    client_frame_format: str = "pil",
) -> tuple[list[Any], str | None, dict[str, Any], float, Any]:
    """Create client frames and wav audio from a local video for alignment tests.

    Video frames are not sampled independently here.  They come directly from
    ``VIDEO_READER_BACKENDS[get_video_reader_backend()](ele)``, i.e. the same
    pre-resize frames used by the original direct video path.
    """
    del image_patch_size  # Sampling is owned by the original backend.

    video_path = Path(video_path)
    video_ele = {"type": "video", "video": str(video_path)}
    video_ele.update(dict(ele or {}))

    raw_video, metadata, sample_fps = _call_original_video_backend(video_ele)
    metadata = dict(metadata or {})
    metadata.setdefault("video_backend", _require_original_vision_module().get_video_reader_backend())

    if client_frame_format == "pil":
        frames = tensor_video_to_pil_frames(raw_video)
    elif client_frame_format == "tensor":
        frames = tensor_video_to_tensor_frames(raw_video)
    elif client_frame_format == "numpy":
        frames = tensor_video_to_numpy_frames(raw_video)
    else:
        raise ValueError("client_frame_format must be one of: 'pil', 'tensor', 'numpy'.")

    wav_path = str(output_audio_path) if output_audio_path is not None else None
    audio = extract_audio_from_video_like_original(
        video_path,
        video_ele,
        output_audio_path=wav_path,
        sample_rate=sample_rate,
    )

    return frames, wav_path, metadata, sample_fps, audio


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _float_equal(left: float, right: float, atol: float) -> bool:
    return abs(float(left) - float(right)) <= atol


def _list_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    try:
        return list(left) == list(right)
    except TypeError:
        return left == right


def _audio_diff(left: Any, right: Any) -> tuple[float | None, float | None]:
    np = _import_required("numpy", "numpy")
    if len(left) != len(right):
        return None, None
    if len(left) == 0:
        return 0.0, 0.0
    diff = np.abs(np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32))
    return float(np.max(diff)), float(np.mean(diff))


def _video_diff(left: Any, right: Any) -> tuple[float | None, float | None]:
    torch = _import_required("torch", "torch")
    if _shape_tuple(left) != _shape_tuple(right):
        return None, None
    if left.numel() == 0:
        return 0.0, 0.0
    diff = torch.abs(left.detach().cpu().float() - right.detach().cpu().float())
    return float(torch.max(diff).item()), float(torch.mean(diff).item())


def compare_direct_video_vs_client_payload(
    video_path: str | Path,
    ele: Mapping[str, Any] | None = None,
    image_patch_size: int = 14,
    sample_rate: int = SAMPLE_RATE,
    atol: float = 1e-4,
    client_frame_format: str = "pil",
) -> dict[str, Any]:
    """Compare original direct-video preprocessing with simulated client payload."""
    module = _require_original_vision_module()
    video_path = Path(video_path)
    base_ele = {"type": "video", "video": str(video_path)}
    base_ele.update(dict(ele or {}))

    direct_result = module.fetch_video(
        dict(base_ele),
        image_patch_size=image_patch_size,
        return_video_sample_fps=True,
        return_video_metadata=True,
    )
    direct_video, direct_sample_fps, direct_metadata = _normalize_fetch_video_result(direct_result)
    direct_audio = extract_audio_from_video_like_original(video_path, base_ele, sample_rate=sample_rate)

    with tempfile.TemporaryDirectory(prefix="mm_aligned_") as tmpdir:
        wav_path = Path(tmpdir) / "client_audio.wav"
        frames, saved_wav_path, client_metadata, client_sample_fps, client_audio = simulate_client_payload_from_video(
            video_path,
            base_ele,
            output_audio_path=wav_path,
            image_patch_size=image_patch_size,
            sample_rate=sample_rate,
            client_frame_format=client_frame_format,
        )
        client_ele = {
            "type": "video",
            "video": frames,
            "sample_fps": client_sample_fps,
            "raw_fps": client_metadata.get("fps"),
            "frames_indices": client_metadata.get("frames_indices"),
            "total_num_frames": client_metadata.get("total_num_frames"),
            "video_backend": client_metadata.get("video_backend"),
        }
        for key in (
            "min_pixels",
            "max_pixels",
            "total_pixels",
            "resized_height",
            "resized_width",
        ):
            if key in base_ele:
                client_ele[key] = base_ele[key]

        client_result = fetch_video_from_frames_aligned(
            client_ele,
            image_patch_size=image_patch_size,
            return_video_sample_fps=True,
            return_video_metadata=True,
        )
        client_video, client_sample_fps_2, client_metadata_2 = _normalize_fetch_video_result(client_result)
        client_audio_loaded = load_audio_aligned(saved_wav_path, sample_rate=sample_rate)

    direct_shape = _shape_tuple(direct_video)
    client_shape = _shape_tuple(client_video)
    video_max_diff, video_mean_diff = _video_diff(direct_video, client_video)
    audio_max_diff, audio_mean_diff = _audio_diff(direct_audio, client_audio_loaded)

    direct_frames_indices = direct_metadata.get("frames_indices")
    client_frames_indices = client_metadata_2.get("frames_indices", client_metadata.get("frames_indices"))
    sample_fps_equal = _float_equal(direct_sample_fps, client_sample_fps_2, atol)
    frames_indices_equal = _list_equal(direct_frames_indices, client_frames_indices)
    audio_length_equal = len(direct_audio) == len(client_audio_loaded)

    passed = (
        direct_shape == client_shape
        and sample_fps_equal
        and frames_indices_equal
        and video_max_diff is not None
        and video_max_diff <= atol
        and audio_length_equal
        and audio_max_diff is not None
        and audio_max_diff <= atol
    )

    return {
        "video_shape_equal": direct_shape == client_shape,
        "direct_video_shape": direct_shape,
        "client_video_shape": client_shape,
        "sample_fps_equal": sample_fps_equal,
        "direct_sample_fps": direct_sample_fps,
        "client_sample_fps": client_sample_fps_2,
        "frames_indices_equal": frames_indices_equal,
        "direct_frames_indices": direct_frames_indices,
        "client_frames_indices": client_frames_indices,
        "video_max_abs_diff": video_max_diff,
        "video_mean_abs_diff": video_mean_diff,
        "audio_length_equal": audio_length_equal,
        "direct_audio_length": int(len(direct_audio)),
        "client_audio_length": int(len(client_audio_loaded)),
        "audio_max_abs_diff": audio_max_diff,
        "audio_mean_abs_diff": audio_mean_diff,
        "client_frame_format": client_frame_format,
        "video_backend": client_metadata.get("video_backend"),
        "passed": passed,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direct video preprocessing with client frames+wav.")
    parser.add_argument("--video", required=True, help="Path to a local video file.")
    parser.add_argument("--fps", type=float, default=None, help="Optional fps field passed to original Qwen backend.")
    parser.add_argument("--video-start", type=float, default=None, help="Optional video_start/audio offset in seconds.")
    parser.add_argument("--video-end", type=float, default=None, help="Optional video_end in seconds.")
    parser.add_argument("--image-patch-size", type=int, default=14)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--client-frame-format",
        choices=("pil", "tensor", "numpy"),
        default="pil",
        help="How to represent simulated client frames.",
    )
    args = parser.parse_args()

    ele: dict[str, Any] = {}
    if args.fps is not None:
        ele["fps"] = args.fps
    if args.video_start is not None:
        ele["video_start"] = args.video_start
    if args.video_end is not None:
        ele["video_end"] = args.video_end

    report = compare_direct_video_vs_client_payload(
        args.video,
        ele=ele,
        image_patch_size=args.image_patch_size,
        sample_rate=args.sample_rate,
        atol=args.atol,
        client_frame_format=args.client_frame_format,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
