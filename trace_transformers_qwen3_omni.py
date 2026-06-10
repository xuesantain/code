"""Capture Transformers Qwen3-Omni text-generation traces.

This script saves:

* ``hf_preprocess_trace.json``: chat template, prompt ids, multimodal summaries.
* ``hf_decode_steps.jsonl``: one row per generated token with top-k logprobs.
* ``hf_output.json``: decoded generated text.

It focuses on the Thinker text path and does not request audio generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from trace_utils import dump_json, dump_jsonl, load_json, summarize_value


DEFAULT_TOPK = 20
DEFAULT_MAX_NEW_TOKENS = 128


def _load_conversation(path: str | Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Conversation file must contain a JSON list.")
    return data


def _load_model(transformers: Any, model_path: str, args: argparse.Namespace) -> Any:
    model_cls = getattr(transformers, "Qwen3OmniMoeForConditionalGeneration", None)
    if model_cls is None:
        raise ImportError("transformers does not expose Qwen3OmniMoeForConditionalGeneration.")

    kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if args.dtype:
        kwargs["dtype"] = args.dtype

    try:
        return model_cls.from_pretrained(model_path, **kwargs)
    except TypeError:
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return model_cls.from_pretrained(model_path, **kwargs)


def _decode_one(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _build_decode_rows(
    *,
    backend: str,
    tokenizer: Any,
    token_ids: list[int],
    scores: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step, token_id in enumerate(token_ids):
        step_scores = scores[step][0].float()
        logprobs = torch.log_softmax(step_scores, dim=-1)
        chosen_logprob = float(logprobs[int(token_id)].item())
        top = torch.topk(logprobs, k=min(int(top_k), int(logprobs.numel())))
        topk = [
            {
                "token_id": int(tid),
                "token": _decode_one(tokenizer, int(tid)),
                "logprob": float(lp),
            }
            for lp, tid in zip(top.values.detach().cpu().tolist(), top.indices.detach().cpu().tolist())
        ]
        rows.append(
            {
                "backend": backend,
                "step": int(step),
                "chosen_token_id": int(token_id),
                "chosen_token": _decode_one(tokenizer, int(token_id)),
                "chosen_logprob": chosen_logprob,
                "topk": topk,
            }
        )
    return rows


def capture_transformers_trace(args: argparse.Namespace) -> dict[str, Any]:
    import transformers
    from qwen_omni_utils import process_mm_info

    output_dir = Path(args.output_dir)
    conversation = _load_conversation(args.conversation_file)
    processor = transformers.Qwen3OmniMoeProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model = _load_model(transformers, args.model_path, args)
    model.eval()

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=args.use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=args.use_audio_in_video,
    )
    inputs = inputs.to(model.device)

    prompt_len = int(inputs["input_ids"].shape[1])
    preprocess_trace = {
        "backend": "transformers",
        "model": args.model_path,
        "use_audio_in_video": bool(args.use_audio_in_video),
        "chat_template_text": text,
        "raw_mm": {
            "audios": summarize_value(audios),
            "images": summarize_value(images),
            "videos": summarize_value(videos),
        },
        "processor_inputs": summarize_value(dict(inputs)),
        "input_ids": inputs["input_ids"][0].detach().cpu().tolist(),
    }
    dump_json(output_dir / "hf_preprocess_trace.json", preprocess_trace)

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            return_audio=False,
            use_audio_in_video=args.use_audio_in_video,
            thinker_return_dict_in_generate=True,
            thinker_output_scores=True,
            thinker_max_new_tokens=args.max_new_tokens,
            thinker_do_sample=False,
        )

    new_token_ids = gen_out.sequences[0, prompt_len:].detach().cpu().tolist()
    rows = _build_decode_rows(
        backend="transformers",
        tokenizer=processor.tokenizer,
        token_ids=[int(token_id) for token_id in new_token_ids],
        scores=gen_out.scores,
        top_k=args.top_k,
    )
    decoded = processor.batch_decode(
        gen_out.sequences[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    dump_jsonl(output_dir / "hf_decode_steps.jsonl", rows)
    output = {"backend": "transformers", "generated_text": decoded, "token_ids": new_token_ids}
    dump_json(output_dir / "hf_output.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Save Transformers Qwen3-Omni Thinker generation trace.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--conversation-file", required=True, help="JSON file containing the conversation list.")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--use-audio-in-video", dest="use_audio_in_video", action="store_true", default=True)
    parser.add_argument("--no-use-audio-in-video", dest="use_audio_in_video", action="store_false")
    args = parser.parse_args()

    output = capture_transformers_trace(args)
    print(output["generated_text"])


if __name__ == "__main__":
    main()
