"""JSON persistence helpers for experiment results.

This module provides robust JSON save/load helpers that handle common numpy
objects and non-finite values (NaN/Inf) safely.
"""

import json
import os
from datetime import datetime

import numpy as np


def _to_jsonable(obj):
    """Recursively convert Python/numpy objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if not np.isfinite(val):
            return None
        return val

    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    return obj


def save_results_json(results, name, folder="results", add_timestamp=False, metadata=None):
    """Save experiment results to JSON file.

    Parameters
    ----------
    results : Any
        Results object (typically list/dict) to serialize.
    name : str
        Base filename (with or without .json extension).
    folder : str, optional
        Output folder (default: "results").
    add_timestamp : bool, optional
        If True, append timestamp YYYYMMDD_HHMMSS to filename.
    metadata : dict or None, optional
        Optional metadata to store alongside results.

    Returns
    -------
    str
        Full output path.
    """
    os.makedirs(folder, exist_ok=True)

    base = name[:-5] if name.lower().endswith(".json") else name
    if add_timestamp:
        base = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    filepath = os.path.join(folder, base + ".json")

    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "results": _to_jsonable(results),
    }
    if metadata is not None:
        payload["metadata"] = _to_jsonable(metadata)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)

    return filepath


def load_results_json(name, folder="results", return_metadata=False):
    """Load experiment results from JSON file.

    Supports both wrapped format {"results": ...} and raw JSON content.

    Parameters
    ----------
    name : str
        Filename (with or without .json extension).
    folder : str, optional
        Input folder (default: "results").
    return_metadata : bool, optional
        If True, return tuple (results, metadata).

    Returns
    -------
    Any or tuple[Any, dict | None]
        Loaded results, optionally with metadata.
    """
    filepath = os.path.join(folder, name if name.lower().endswith(".json") else name + ".json")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        results = data["results"]
        metadata = data.get("metadata")
    else:
        results = data
        metadata = None

    if return_metadata:
        return results, metadata

    return results
