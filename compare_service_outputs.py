"""Offline diff for Transformers and vLLM trace files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from trace_utils import dump_json, load_json, load_jsonl


def _prompt_ids(trace: dict[str, Any]) -> list[int] | None:
    value = trace.get("input_ids", trace.get("prompt_input_ids"))
    if value is None:
        return None
    return [int(token_id) for token_id in value]


def _topk_to_map(row: dict[str, Any]) -> dict[int, float]:
    result: dict[int, float] = {}
    for item in row.get("topk", []) or []:
        token_id = item.get("token_id")
        logprob = item.get("logprob")
        if token_id is not None and logprob is not None:
            result[int(token_id)] = float(logprob)
    return result


def _compare_summary(left: Any, right: Any, path: str, diffs: list[dict[str, Any]]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left.keys()) | set(right.keys())):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in left:
                diffs.append({"path": child_path, "kind": "missing_left", "right": right[key]})
            elif key not in right:
                diffs.append({"path": child_path, "kind": "missing_right", "left": left[key]})
            elif key in {"type", "len", "shape", "dtype", "hash"} and left[key] != right[key]:
                diffs.append({"path": child_path, "kind": "value_diff", "left": left[key], "right": right[key]})
            elif key not in {"repr"}:
                _compare_summary(left[key], right[key], child_path, diffs)
        return

    if isinstance(left, list) and isinstance(right, list):
        for index in range(min(len(left), len(right))):
            _compare_summary(left[index], right[index], f"{path}[{index}]", diffs)
        if len(left) != len(right):
            diffs.append({"path": path, "kind": "list_len_diff", "left": len(left), "right": len(right)})
        return

    if left != right:
        diffs.append({"path": path, "kind": "value_diff", "left": left, "right": right})


def _compare_preprocess(hf: dict[str, Any], vllm: dict[str, Any]) -> dict[str, Any]:
    hf_ids = _prompt_ids(hf)
    vllm_ids = _prompt_ids(vllm)
    raw_mm_diffs: list[dict[str, Any]] = []
    _compare_summary(hf.get("raw_mm"), vllm.get("raw_mm"), "raw_mm", raw_mm_diffs)

    first_prompt_id_diff = None
    if hf_ids is not None and vllm_ids is not None:
        for index, (left, right) in enumerate(zip(hf_ids, vllm_ids)):
            if left != right:
                first_prompt_id_diff = {"index": index, "hf": left, "vllm": right}
                break
        if first_prompt_id_diff is None and len(hf_ids) != len(vllm_ids):
            first_prompt_id_diff = {"index": min(len(hf_ids), len(vllm_ids)), "hf_len": len(hf_ids), "vllm_len": len(vllm_ids)}

    return {
        "chat_template_equal": hf.get("chat_template_text") == vllm.get("chat_template_text"),
        "hf_prompt_len": None if hf_ids is None else len(hf_ids),
        "vllm_prompt_len": None if vllm_ids is None else len(vllm_ids),
        "prompt_ids_equal": hf_ids == vllm_ids if hf_ids is not None and vllm_ids is not None else None,
        "first_prompt_id_diff": first_prompt_id_diff,
        "raw_mm_equal": len(raw_mm_diffs) == 0,
        "raw_mm_diffs": raw_mm_diffs[:50],
        "raw_mm_diff_count": len(raw_mm_diffs),
    }


def _step_topk_diff(hf_row: dict[str, Any], vllm_row: dict[str, Any]) -> dict[str, Any]:
    hf_top = _topk_to_map(hf_row)
    vllm_top = _topk_to_map(vllm_row)
    common = sorted(set(hf_top.keys()) & set(vllm_top.keys()))
    diffs = [
        {
            "token_id": token_id,
            "hf_logprob": hf_top[token_id],
            "vllm_logprob": vllm_top[token_id],
            "abs_diff": abs(hf_top[token_id] - vllm_top[token_id]),
        }
        for token_id in common
    ]
    diffs.sort(key=lambda item: item["abs_diff"], reverse=True)
    return {
        "common_topk_count": len(common),
        "hf_only_topk": sorted(set(hf_top.keys()) - set(vllm_top.keys()))[:20],
        "vllm_only_topk": sorted(set(vllm_top.keys()) - set(hf_top.keys()))[:20],
        "largest_common_logprob_diffs": diffs[:10],
    }


def _compare_decode_steps(
    hf_steps: list[dict[str, Any]],
    vllm_steps: list[dict[str, Any]],
    *,
    logprob_atol: float,
) -> dict[str, Any]:
    compared_steps = min(len(hf_steps), len(vllm_steps))
    first_token_divergence = None
    first_logprob_divergence = None
    matching_prefix_steps = 0

    for index in range(compared_steps):
        hf_row = hf_steps[index]
        vllm_row = vllm_steps[index]
        same_token = hf_row.get("chosen_token_id") == vllm_row.get("chosen_token_id")
        if same_token and first_token_divergence is None:
            matching_prefix_steps += 1
        if not same_token and first_token_divergence is None:
            first_token_divergence = {
                "step": index,
                "hf": {
                    "token_id": hf_row.get("chosen_token_id"),
                    "token": hf_row.get("chosen_token"),
                    "logprob": hf_row.get("chosen_logprob"),
                },
                "vllm": {
                    "token_id": vllm_row.get("chosen_token_id"),
                    "token": vllm_row.get("chosen_token"),
                    "logprob": vllm_row.get("chosen_logprob"),
                },
                "topk": _step_topk_diff(hf_row, vllm_row),
            }

        hf_lp = hf_row.get("chosen_logprob")
        vllm_lp = vllm_row.get("chosen_logprob")
        if hf_lp is not None and vllm_lp is not None:
            lp_diff = abs(float(hf_lp) - float(vllm_lp))
            if lp_diff > logprob_atol and first_logprob_divergence is None:
                first_logprob_divergence = {
                    "step": index,
                    "hf_logprob": float(hf_lp),
                    "vllm_logprob": float(vllm_lp),
                    "abs_diff": lp_diff,
                }

    return {
        "hf_step_count": len(hf_steps),
        "vllm_step_count": len(vllm_steps),
        "compared_steps": compared_steps,
        "length_equal": len(hf_steps) == len(vllm_steps),
        "token_ids_equal": first_token_divergence is None and len(hf_steps) == len(vllm_steps),
        "matching_prefix_steps": matching_prefix_steps,
        "first_token_divergence": first_token_divergence,
        "first_logprob_divergence": first_logprob_divergence,
    }


def _maybe_load(path: str | None) -> Any | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return load_json(candidate)


def compare_traces(args: argparse.Namespace) -> dict[str, Any]:
    hf_preprocess = load_json(args.hf_preprocess)
    vllm_preprocess = load_json(args.vllm_preprocess)
    hf_steps = load_jsonl(args.hf_steps)
    vllm_steps = load_jsonl(args.vllm_steps)

    hf_output = _maybe_load(args.hf_output)
    vllm_output = _maybe_load(args.vllm_output)
    output_report = None
    if hf_output is not None and vllm_output is not None:
        output_report = {
            "hf_generated_text": hf_output.get("generated_text"),
            "vllm_generated_text": vllm_output.get("generated_text"),
            "generated_text_equal": hf_output.get("generated_text") == vllm_output.get("generated_text"),
            "token_ids_equal": hf_output.get("token_ids") == vllm_output.get("token_ids"),
        }

    preprocess = _compare_preprocess(hf_preprocess, vllm_preprocess)
    decode = _compare_decode_steps(hf_steps, vllm_steps, logprob_atol=args.logprob_atol)
    passed = (
        bool(preprocess["chat_template_equal"])
        and preprocess["prompt_ids_equal"] is not False
        and bool(preprocess["raw_mm_equal"])
        and bool(decode["token_ids_equal"])
        and (output_report is None or bool(output_report["generated_text_equal"]))
    )

    return {
        "passed": passed,
        "preprocess": preprocess,
        "decode": decode,
        "output": output_report,
    }


def _print_human(report: dict[str, Any]) -> None:
    preprocess = report["preprocess"]
    decode = report["decode"]
    print(f"passed={report['passed']}")
    print(f"chat_template_equal={preprocess['chat_template_equal']}")
    print(f"prompt_ids_equal={preprocess['prompt_ids_equal']}")
    print(f"raw_mm_equal={preprocess['raw_mm_equal']}")
    print(f"token_ids_equal={decode['token_ids_equal']}")
    print(f"matching_prefix_steps={decode['matching_prefix_steps']}")
    if preprocess["first_prompt_id_diff"] is not None:
        print(f"first_prompt_id_diff={preprocess['first_prompt_id_diff']}")
    if decode["first_token_divergence"] is not None:
        print(f"first_token_divergence={decode['first_token_divergence']}")
    if decode["first_logprob_divergence"] is not None:
        print(f"first_logprob_divergence={decode['first_logprob_divergence']}")
    if report.get("output") is not None:
        print(f"generated_text_equal={report['output']['generated_text_equal']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Transformers and vLLM service trace files.")
    parser.add_argument("--hf-preprocess", default="hf_preprocess_trace.json")
    parser.add_argument("--vllm-preprocess", default="vllm_preprocess_trace.json")
    parser.add_argument("--hf-steps", default="hf_decode_steps.jsonl")
    parser.add_argument("--vllm-steps", default="vllm_decode_steps.jsonl")
    parser.add_argument("--hf-output", default="hf_output.json")
    parser.add_argument("--vllm-output", default="vllm_output.json")
    parser.add_argument("--logprob-atol", type=float, default=1e-4)
    parser.add_argument("--output", default=None, help="Optional path for a JSON comparison report.")
    args = parser.parse_args()

    report = compare_traces(args)
    if args.output:
        dump_json(args.output, report)
    _print_human(report)


if __name__ == "__main__":
    main()
