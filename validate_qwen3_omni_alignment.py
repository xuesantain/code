"""Validate client frames+audio against the official Qwen3-Omni service shape.

This script mirrors the official Transformers example at preprocessing level:

    text = processor.apply_chat_template(...)
    audios, images, videos = process_mm_info(...)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, ...)

The direct path uses ``qwen_omni_utils.process_mm_info`` with a local video
path.  The client path uses pre-extracted frames plus a wav audio file and the
aligned server-side code from ``mm_aligned.py``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from mm_aligned import (
    SAMPLE_RATE,
    _import_required,
    process_mm_info_client_aligned_qwen3_omni,
)
from test_alignment import (
    _audio_diff,
    _shape_tuple,
    _video_diff,
    simulate_client_payload_from_video,
)


DEFAULT_PROMPT = "What can you see and hear? Answer in one short sentence."


def _first_or_none(values: Any) -> Any:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        return values[0] if values else None
    return values


def _sequence_len(values: Any) -> int:
    if values is None:
        return 0
    if isinstance(values, (list, tuple)):
        return len(values)
    return 1


def _build_direct_conversation(video_path: str | Path, prompt: str, ele: Mapping[str, Any]) -> list[dict[str, Any]]:
    video_item = {"type": "video", "video": str(video_path)}
    video_item.update(dict(ele))
    return [{"role": "user", "content": [video_item, {"type": "text", "text": prompt}]}]


def _build_client_conversation(
    frames: Sequence[Any],
    wav_path: str | Path,
    prompt: str,
    metadata: Mapping[str, Any],
    sample_fps: float,
    ele: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build a client conversation with audio attached to the video item.

    Keeping audio on the video item lets ``apply_chat_template`` see the same
    high-level content structure as the direct ``use_audio_in_video=True`` path:
    one video item plus one text item.
    """
    video_item: dict[str, Any] = {
        "type": "video",
        "video": list(frames),
        "audio": str(wav_path),
        "sample_fps": float(sample_fps),
        "raw_fps": metadata.get("fps"),
        "frames_indices": metadata.get("frames_indices"),
        "total_num_frames": metadata.get("total_num_frames"),
        "video_backend": metadata.get("video_backend"),
    }
    for key in (
        "min_pixels",
        "max_pixels",
        "total_pixels",
        "resized_height",
        "resized_width",
        "video_start",
        "video_end",
    ):
        if key in ele:
            video_item[key] = ele[key]
    return [{"role": "user", "content": [video_item, {"type": "text", "text": prompt}]}]


def _load_official_process_mm_info() -> Any:
    try:
        module = _import_required("qwen_omni_utils", "qwen-omni-utils")
        return module.process_mm_info
    except AttributeError as exc:
        raise ImportError("qwen_omni_utils is importable but does not expose process_mm_info.") from exc


def _compare_preprocess_outputs(
    direct_outputs: tuple[Any, Any, Any],
    client_outputs: tuple[Any, Any, Any],
    *,
    atol: float,
) -> dict[str, Any]:
    direct_audios, direct_images, direct_videos = direct_outputs
    client_audios, client_images, client_videos = client_outputs

    direct_video = _first_or_none(direct_videos)
    client_video = _first_or_none(client_videos)
    direct_audio = _first_or_none(direct_audios)
    client_audio = _first_or_none(client_audios)

    video_max_diff, video_mean_diff = (None, None)
    if direct_video is not None and client_video is not None:
        video_max_diff, video_mean_diff = _video_diff(direct_video, client_video)

    audio_max_diff, audio_mean_diff = (None, None)
    if direct_audio is not None and client_audio is not None:
        audio_max_diff, audio_mean_diff = _audio_diff(direct_audio, client_audio)

    direct_video_shape = _shape_tuple(direct_video)
    client_video_shape = _shape_tuple(client_video)
    direct_audio_len = None if direct_audio is None else int(len(direct_audio))
    client_audio_len = None if client_audio is None else int(len(client_audio))

    passed = (
        _sequence_len(direct_images) == _sequence_len(client_images)
        and _sequence_len(direct_videos) == _sequence_len(client_videos)
        and _sequence_len(direct_audios) == _sequence_len(client_audios)
        and direct_video_shape == client_video_shape
        and video_max_diff is not None
        and video_max_diff <= atol
        and direct_audio_len == client_audio_len
        and audio_max_diff is not None
        and audio_max_diff <= atol
    )

    return {
        "direct_audio_count": _sequence_len(direct_audios),
        "client_audio_count": _sequence_len(client_audios),
        "direct_image_count": _sequence_len(direct_images),
        "client_image_count": _sequence_len(client_images),
        "direct_video_count": _sequence_len(direct_videos),
        "client_video_count": _sequence_len(client_videos),
        "video_shape_equal": direct_video_shape == client_video_shape,
        "direct_video_shape": direct_video_shape,
        "client_video_shape": client_video_shape,
        "video_max_abs_diff": video_max_diff,
        "video_mean_abs_diff": video_mean_diff,
        "audio_length_equal": direct_audio_len == client_audio_len,
        "direct_audio_length": direct_audio_len,
        "client_audio_length": client_audio_len,
        "audio_max_abs_diff": audio_max_diff,
        "audio_mean_abs_diff": audio_mean_diff,
        "passed": passed,
    }


def _tensor_summary(value: Any) -> dict[str, Any]:
    torch = _import_required("torch", "torch")
    if isinstance(value, torch.Tensor):
        return {"shape": tuple(int(dim) for dim in value.shape), "dtype": str(value.dtype)}
    return {"type": type(value).__name__}


def _compare_processor_inputs(direct_inputs: Mapping[str, Any], client_inputs: Mapping[str, Any]) -> dict[str, Any]:
    torch = _import_required("torch", "torch")
    common_keys = sorted(set(direct_inputs.keys()) & set(client_inputs.keys()))
    key_reports: dict[str, Any] = {}
    all_equal = True

    for key in common_keys:
        left = direct_inputs[key]
        right = client_inputs[key]
        report = {
            "direct": _tensor_summary(left),
            "client": _tensor_summary(right),
            "shape_equal": _shape_tuple(left) == _shape_tuple(right),
        }
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and left.shape == right.shape:
            if left.dtype.is_floating_point or right.dtype.is_floating_point:
                diff = torch.abs(left.detach().cpu().float() - right.detach().cpu().float())
                report["max_abs_diff"] = float(torch.max(diff).item()) if diff.numel() else 0.0
                report["mean_abs_diff"] = float(torch.mean(diff).item()) if diff.numel() else 0.0
                report["exact_equal"] = bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
            else:
                report["exact_equal"] = bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
        else:
            report["exact_equal"] = left == right

        all_equal = all_equal and bool(report["shape_equal"]) and bool(report.get("exact_equal", False))
        key_reports[key] = report

    return {
        "direct_only_keys": sorted(set(direct_inputs.keys()) - set(client_inputs.keys())),
        "client_only_keys": sorted(set(client_inputs.keys()) - set(direct_inputs.keys())),
        "common_keys": common_keys,
        "keys": key_reports,
        "exact_passed": all_equal,
    }


def compare_official_qwen3_omni_service_alignment(
    video_path: str | Path,
    *,
    model_path: str | None = None,
    prompt: str = DEFAULT_PROMPT,
    ele: Mapping[str, Any] | None = None,
    image_patch_size: int = 14,
    sample_rate: int = SAMPLE_RATE,
    atol: float = 1e-4,
    client_frame_format: str = "tensor",
    run_processor: bool = False,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Compare official direct-video preprocessing with aligned client payload."""
    process_mm_info = _load_official_process_mm_info()
    base_ele = dict(ele or {})
    video_path = Path(video_path)

    direct_conversation = _build_direct_conversation(video_path, prompt, base_ele)

    with tempfile.TemporaryDirectory(prefix="qwen3_omni_aligned_") as tmpdir:
        wav_path = Path(tmpdir) / "client_audio.wav"
        frames, saved_wav_path, metadata, sample_fps, _client_audio = simulate_client_payload_from_video(
            video_path,
            base_ele,
            output_audio_path=wav_path,
            image_patch_size=image_patch_size,
            sample_rate=sample_rate,
            client_frame_format=client_frame_format,
        )
        client_conversation = _build_client_conversation(
            frames,
            saved_wav_path or wav_path,
            prompt,
            metadata,
            sample_fps,
            base_ele,
        )

        direct_outputs = process_mm_info(direct_conversation, use_audio_in_video=True)
        client_outputs = process_mm_info_client_aligned_qwen3_omni(
            client_conversation,
            image_patch_size=image_patch_size,
            use_audio_in_video=True,
            sample_rate=sample_rate,
        )

        report: dict[str, Any] = {
            "client_frame_format": client_frame_format,
            "video_backend": metadata.get("video_backend"),
            "sample_fps": sample_fps,
            "frames_indices": metadata.get("frames_indices"),
            "preprocess": _compare_preprocess_outputs(direct_outputs, client_outputs, atol=atol),
        }

        if run_processor:
            if not model_path:
                raise ValueError("--model-path is required when --run-processor is used.")
            transformers = _import_required("transformers", "transformers")
            processor = transformers.Qwen3OmniMoeProcessor.from_pretrained(
                model_path,
                local_files_only=local_files_only,
            )

            direct_text = processor.apply_chat_template(
                direct_conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            client_text = processor.apply_chat_template(
                client_conversation,
                add_generation_prompt=True,
                tokenize=False,
            )

            direct_audios, direct_images, direct_videos = direct_outputs
            client_audios, client_images, client_videos = client_outputs
            direct_inputs = processor(
                text=direct_text,
                audio=direct_audios,
                images=direct_images,
                videos=direct_videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            )
            client_inputs = processor(
                text=client_text,
                audio=client_audios,
                images=client_images,
                videos=client_videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True,
            )
            report["processor"] = {
                "chat_template_equal": direct_text == client_text,
                "direct_text": direct_text,
                "client_text": client_text,
                "inputs": _compare_processor_inputs(direct_inputs, client_inputs),
            }

    report["passed"] = bool(report["preprocess"]["passed"]) and (
        not run_processor or bool(report.get("processor", {}).get("chat_template_equal"))
    )
    return report


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate client frames+audio against official Qwen3-Omni preprocessing."
    )
    parser.add_argument("--video", required=True, help="Path to a local mp4/video file.")
    parser.add_argument("--model-path", default=None, help="Model or local processor path for --run-processor.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--video-start", type=float, default=None)
    parser.add_argument("--video-end", type=float, default=None)
    parser.add_argument("--image-patch-size", type=int, default=14)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--client-frame-format", choices=("pil", "tensor", "numpy"), default="tensor")
    parser.add_argument("--run-processor", action="store_true", help="Also compare Qwen3OmniMoeProcessor outputs.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download processor files when --run-processor is used.",
    )
    args = parser.parse_args()

    ele: dict[str, Any] = {}
    if args.fps is not None:
        ele["fps"] = args.fps
    if args.video_start is not None:
        ele["video_start"] = args.video_start
    if args.video_end is not None:
        ele["video_end"] = args.video_end

    report = compare_official_qwen3_omni_service_alignment(
        args.video,
        model_path=args.model_path,
        prompt=args.prompt,
        ele=ele,
        image_patch_size=args.image_patch_size,
        sample_rate=args.sample_rate,
        atol=args.atol,
        client_frame_format=args.client_frame_format,
        run_processor=args.run_processor,
        local_files_only=args.local_files_only,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

