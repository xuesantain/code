"""Utilities for saving comparable model/service traces.

The helpers in this file intentionally store compact summaries for large
arrays/tensors: shape, dtype, and a content hash.  This keeps trace files useful
for offline comparison without dumping every multimodal tensor element.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable


def _optional_import(module_name: str) -> Any | None:
    try:
        return __import__(module_name)
    except ImportError:
        return None


def tensor_hash(value: Any) -> str | None:
    """Return a SHA256 hash for tensor-like values, or ``None`` if unsupported."""
    np = _optional_import("numpy")
    torch = _optional_import("torch")

    if value is None or np is None:
        return None

    if torch is not None and isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
    else:
        try:
            array = np.ascontiguousarray(np.asarray(value))
        except Exception:
            return None

    return hashlib.sha256(array.tobytes()).hexdigest()


def summarize_value(value: Any, *, max_items: int = 5, max_depth: int = 6) -> Any:
    """Summarize nested values for trace files.

    Tensor and ndarray payloads are represented by type, shape, dtype, and hash.
    Lists and dictionaries are traversed with bounded depth and item count.
    """
    np = _optional_import("numpy")
    torch = _optional_import("torch")

    if max_depth < 0:
        return {"type": type(value).__name__, "repr": repr(value)[:300], "truncated": True}

    if torch is not None and isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "hash": tensor_hash(value),
        }

    if np is not None and isinstance(value, np.ndarray):
        return {
            "type": "np.ndarray",
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "hash": tensor_hash(value),
        }

    if isinstance(value, Mapping):
        return {
            str(key): summarize_value(item, max_items=max_items, max_depth=max_depth - 1)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        items = [
            summarize_value(item, max_items=max_items, max_depth=max_depth - 1)
            for item in list(value)[:max_items]
        ]
        return {
            "type": type(value).__name__,
            "len": len(value),
            "items": items,
            "truncated": len(value) > max_items,
        }

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, Path):
        return str(value)

    return {"type": type(value).__name__, "repr": repr(value)[:300]}


def dump_json(path: str | Path, obj: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def dump_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default))
            handle.write("\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def json_default(value: Any) -> Any:
    np = _optional_import("numpy")
    torch = _optional_import("torch")

    if isinstance(value, Path):
        return str(value)

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()

    if isinstance(value, tuple):
        return list(value)

    return repr(value)
