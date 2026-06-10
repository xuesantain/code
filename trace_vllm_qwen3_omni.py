"""Capture vLLM Qwen3-Omni text-generation traces.

This uses vLLM's Python offline API because it exposes token ids and per-token
logprobs more directly than an OpenAI-compatible service endpoint.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any

from trace_utils import dump_json, dump_jsonl, load_json, summarize_value


DEFAULT_TOPK = 20
DEFAULT_MAX_NEW_TOKENS = 128


def _load_conversation(path: str | Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Conversation file must contain a JSON list.")
    return data


def _decode_one(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _value_from_logprob_info(info: Any, name: str, default: Any = None) -> Any:
    if isinstance(info, dict):
        return info.get(name, default)
    return getattr(info, name, default)


def _logprob_item_to_dict(token_id: int, info: Any, tokenizer: Any) -> dict[str, Any]:
    token = _value_from_logprob_info(info, "decoded_token")
    if token is None:
        token = _decode_one(tokenizer, token_id)
    logprob = _value_from_logprob_info(info, "logprob")
    return {
        "token_id": int(token_id),
        "token": token,
        "logprob": None if logprob is None else float(logprob),
    }


def _extract_vllm_decode_rows(completion: Any, tokenizer: Any) -> list[dict[str, Any]]:
    token_ids = [int(token_id) for token_id in list(getattr(completion, "token_ids", []) or [])]
    step_logprobs = getattr(completion, "logprobs", None) or []
    rows: list[dict[str, Any]] = []

    for step, token_id in enumerate(token_ids):
        logprob_dict = step_logprobs[step] if step < len(step_logprobs) and step_logprobs[step] else {}
        chosen_info = logprob_dict.get(token_id) if isinstance(logprob_dict, dict) else None
        chosen_logprob_value = None if chosen_info is None else _value_from_logprob_info(chosen_info, "logprob")
        chosen_logprob = None if chosen_logprob_value is None else float(chosen_logprob_value)
        chosen_token = (
            _decode_one(tokenizer, token_id)
            if chosen_info is None
            else _value_from_logprob_info(chosen_info, "decoded_token") or _decode_one(tokenizer, token_id)
        )

        topk: list[dict[str, Any]] = []
        if isinstance(logprob_dict, dict):
            topk = [
                _logprob_item_to_dict(int(tid), info, tokenizer)
                for tid, info in logprob_dict.items()
                if info is not None
            ]
            topk = [item for item in topk if item["logprob"] is not None]
            topk.sort(key=lambda item: item["logprob"], reverse=True)

        rows.append(
            {
                "backend": "vllm",
                "step": int(step),
                "chosen_token_id": int(token_id),
                "chosen_token": chosen_token,
                "chosen_logprob": chosen_logprob,
                "topk": topk,
            }
        )

    return rows


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _load_processor(transformers: Any, model_path: str, args: argparse.Namespace) -> Any:
    processor_cls = getattr(transformers, "Qwen3OmniMoeProcessor", None)
    if processor_cls is None:
        processor_cls = getattr(transformers, "AutoProcessor", None)
    if processor_cls is None:
        raise ImportError("transformers does not expose Qwen3OmniMoeProcessor or AutoProcessor.")
    return processor_cls.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )


def capture_vllm_trace(args: argparse.Namespace) -> dict[str, Any]:
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    import transformers
    from qwen_omni_utils import process_mm_info
    from vllm import LLM, SamplingParams

    output_dir = Path(args.output_dir)
    conversation = _load_conversation(args.conversation_file)
    processor = _load_processor(transformers, args.model_path, args)

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=args.use_audio_in_video)
    multi_modal_data: dict[str, Any] = {}
    if images is not None:
        multi_modal_data["image"] = images
    if videos is not None:
        multi_modal_data["video"] = videos
    if audios is not None:
        multi_modal_data["audio"] = audios

    prompt_ids = processor.tokenizer(text, add_special_tokens=False).input_ids
    preprocess_trace = {
        "backend": "vllm",
        "model": args.model_path,
        "use_audio_in_video": bool(args.use_audio_in_video),
        "chat_template_text": text,
        "prompt_input_ids": prompt_ids,
        "raw_mm": {
            "audios": summarize_value(audios),
            "images": summarize_value(images),
            "videos": summarize_value(videos),
        },
        "multi_modal_data": summarize_value(multi_modal_data),
        "mm_processor_kwargs": {"use_audio_in_video": bool(args.use_audio_in_video)},
    }
    dump_json(output_dir / "vllm_preprocess_trace.json", preprocess_trace)

    tensor_parallel_size = args.tensor_parallel_size
    if tensor_parallel_size is None:
        tensor_parallel_size = max(1, torch.cuda.device_count())

    llm_kwargs = {
        "model": args.model_path,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "limit_mm_per_prompt": {
            "image": args.limit_images,
            "video": args.limit_videos,
            "audio": args.limit_audios,
        },
        "max_num_seqs": 1,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
    }
    llm = LLM(**_filter_supported_kwargs(LLM, llm_kwargs))
    vllm_input = {
        "prompt": text,
        "multi_modal_data": multi_modal_data,
        "mm_processor_kwargs": {"use_audio_in_video": bool(args.use_audio_in_video)},
    }
    sampling_kwargs = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_tokens": args.max_new_tokens,
        "logprobs": args.top_k,
        "prompt_logprobs": args.top_k if args.prompt_logprobs else None,
        "seed": args.seed,
    }
    sampling_params = SamplingParams(**_filter_supported_kwargs(SamplingParams, sampling_kwargs))

    outputs = llm.generate([vllm_input], sampling_params=sampling_params)
    request_output = outputs[0]
    completion = request_output.outputs[0]
    rows = _extract_vllm_decode_rows(completion, processor.tokenizer)

    dump_jsonl(output_dir / "vllm_decode_steps.jsonl", rows)
    output = {
        "backend": "vllm",
        "generated_text": getattr(completion, "text", ""),
        "token_ids": list(getattr(completion, "token_ids", []) or []),
        "prompt_token_ids": getattr(request_output, "prompt_token_ids", None),
    }
    dump_json(output_dir / "vllm_output.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Save vLLM Qwen3-Omni Thinker generation trace.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--conversation-file", required=True, help="JSON file containing the conversation list.")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--limit-images", type=int, default=3)
    parser.add_argument("--limit-videos", type=int, default=3)
    parser.add_argument("--limit-audios", type=int, default=3)
    parser.add_argument("--prompt-logprobs", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--use-audio-in-video", dest="use_audio_in_video", action="store_true", default=True)
    parser.add_argument("--no-use-audio-in-video", dest="use_audio_in_video", action="store_false")
    args = parser.parse_args()

    output = capture_vllm_trace(args)
    print(output["generated_text"])


if __name__ == "__main__":
    main()
