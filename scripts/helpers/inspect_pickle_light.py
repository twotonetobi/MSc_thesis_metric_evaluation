#!/usr/bin/env python3
"""
inspect_pickle_light.py

Analyzes a 'light' pickle file and, if possible, also analyzes the corresponding
AUDIO pickle (derived by filename) and writes a nicely formatted combined report.

Caution: Unpickling arbitrary files can execute code. Use only with trusted files.
"""
import argparse
import io
import os
import pickle
import sys
import types
from collections import Counter
from datetime import datetime
from pprint import pformat
from typing import List, Optional

# Optional imports
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore


# ----- Helpers for summarization -----

def is_numpy_array(obj):
    return np is not None and isinstance(obj, np.ndarray)


def is_pandas_dataframe(obj):
    return pd is not None and isinstance(obj, pd.DataFrame)


def is_pandas_series(obj):
    return pd is not None and isinstance(obj, pd.Series)


def safe_len(obj):
    try:
        return len(obj)  # type: ignore
    except Exception:
        return None


def basic_type(obj):
    t = type(obj)
    return f"{t.__module__}.{t.__name__}"


def summarize_scalar(obj):
    return {"type": basic_type(obj), "repr": repr(obj)[:200]}


def summarize_numpy(arr):
    summary = {"type": basic_type(arr), "shape": tuple(arr.shape), "dtype": str(arr.dtype)}
    if arr.size > 0 and np is not None:
        try:
            if np.issubdtype(arr.dtype, np.number):
                summary.update({
                    "min": float(np.nanmin(arr)),
                    "max": float(np.nanmax(arr)),
                    "mean": float(np.nanmean(arr)),
                    "std": float(np.nanstd(arr)),
                })
        except Exception:
            pass
        try:
            flat = arr.ravel()
            preview_count = min(10, flat.size)
            summary["sample_values"] = [repr(x) for x in flat[:preview_count]]
        except Exception:
            pass
    return summary


def summarize_pandas_df(df):
    summary = {
        "type": basic_type(df),
        "shape": (df.shape[0], df.shape[1]),
        "columns": list(df.columns.astype(str)),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
    }
    try:
        summary["head"] = df.head(3).to_dict(orient="list")
    except Exception:
        pass
    return summary


def summarize_pandas_series(s):
    summary = {"type": basic_type(s), "length": int(s.shape[0]), "dtype": str(s.dtype)}
    try:
        summary["head"] = list(map(repr, s.head(5).tolist()))
    except Exception:
        pass
    return summary


def summarize_mapping(d, depth, max_depth):
    out = {
        "type": basic_type(d),
        "length": safe_len(d),
        "key_types": Counter(basic_type(k) for k in d.keys()) if hasattr(d, "keys") else None,
        "value_types": Counter(basic_type(v) for v in d.values()) if hasattr(d, "values") else None,
    }
    items = []
    try:
        for i, (k, v) in enumerate(d.items()):
            if i >= 10:
                break
            items.append({
                "key_type": basic_type(k),
                "key_repr": repr(k)[:200],
                "value_summary": summarize(v, depth + 1, max_depth),
            })
    except Exception:
        pass
    out["sample_items"] = items
    return out


def summarize_sequence(seq, depth, max_depth):
    out = {"type": basic_type(seq), "length": safe_len(seq)}
    elems = []
    try:
        it = iter(seq)
        for i, v in enumerate(it):
            if i >= 10:
                break
            elems.append(summarize(v, depth + 1, max_depth))
    except Exception:
        pass
    out["sample_elements"] = elems
    return out


def summarize_object(obj, depth, max_depth):
    out = {"type": basic_type(obj)}
    attrs = {}
    for name in dir(obj):
        if name.startswith("__") and name.endswith("__"):
            continue
        try:
            val = getattr(obj, name)
            if isinstance(val, (types.FunctionType, types.MethodType)):
                continue
            attrs[name] = basic_type(val)
            if len(attrs) >= 20:
                break
        except Exception:
            continue
    out["attrs_sample_types"] = attrs
    out["repr"] = repr(obj)[:200]
    return out


def summarize(obj, depth=0, max_depth=3):
    if depth > max_depth:
        return {"type": basic_type(obj), "note": "max_depth_reached"}
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return summarize_scalar(obj)
    if is_numpy_array(obj):
        return summarize_numpy(obj)
    if is_pandas_dataframe(obj):
        return summarize_pandas_df(obj)
    if is_pandas_series(obj):
        return summarize_pandas_series(obj)
    if isinstance(obj, dict):
        return summarize_mapping(obj, depth, max_depth)
    if isinstance(obj, (list, tuple, set, frozenset)):
        return summarize_sequence(obj, depth, max_depth)
    return summarize_object(obj, depth, max_depth)


# ----- Heuristic 'tic/tick' detection -----

def detect_tics(obj):
    findings = []

    def add_finding(path, kind, extra=None):
        item = {"path": path, "kind": kind}
        if extra is not None:
            item.update(extra)
        findings.append(item)

    def scan(o, path="$"):
        if is_numpy_array(o):
            arr = o
            try:
                flat = arr.ravel()
                if np is not None:
                    if flat.size >= 2 and np.issubdtype(arr.dtype, np.integer):
                        diffs = np.diff(flat[: min(1000, flat.size)])
                        if np.all(diffs >= 0):
                            add_finding(path, "monotonic_integer_sequence", {
                                "length": int(flat.size),
                                "first_values": flat[:5].tolist(),
                            })
                    uniq = np.unique(flat[: min(10000, flat.size)])
                    try:
                        uniq_int = set(map(int, uniq.tolist()))
                    except Exception:
                        uniq_int = set()
                    if len(uniq) <= 3 and uniq_int.issubset({0, 1}):
                        density = float(np.mean(flat != 0))
                        add_finding(path, "binary_spike_train", {
                            "length": int(flat.size),
                            "density": density,
                        })
            except Exception:
                pass
            return
        if is_pandas_series(o):
            s = o
            try:
                if pd is not None and pd.api.types.is_integer_dtype(s.dtype) and s.shape[0] >= 2:
                    diffs = s.iloc[: min(1000, len(s))].diff().fillna(0)
                    if (diffs >= 0).all():
                        add_finding(path, "monotonic_integer_series", {
                            "length": int(len(s)),
                            "first_values": list(map(int, s.head(5).tolist())),
                        })
            except Exception:
                pass
            return
        if is_pandas_dataframe(o):
            try:
                for col in o.columns:
                    scan(o[col], f"{path}['{col}']")
            except Exception:
                pass
            return
        if isinstance(o, dict):
            for k, v in list(o.items())[:100]:
                k_str = str(k).lower()
                if any(tag in k_str for tag in ["tic", "tick", "time", "ts", "spike"]):
                    add_finding(f"{path}[{repr(k)}]", "name_match")
                scan(v, f"{path}[{repr(k)}]")
            return
        if isinstance(o, (list, tuple)):
            for i, v in enumerate(list(o)[:100]):
                scan(v, f"{path}[{i}]")
            return
        if isinstance(o, (set, frozenset)):
            for i, v in enumerate(list(o)[:100]):
                scan(v, f"{path}{{{i}}}")
            return
        try:
            for name in dir(o):
                if name.startswith("__") and name.endswith("__"):
                    continue
                try:
                    v = getattr(o, name)
                    if isinstance(v, (dict, list, tuple)) or is_numpy_array(v) or is_pandas_series(v) or is_pandas_dataframe(v):
                        if any(tag in name.lower() for tag in ["tic", "tick", "time", "ts", "spike"]):
                            add_finding(f"{path}.{name}", "name_match")
                        scan(v, f"{path}.{name}")
                except Exception:
                    continue
        except Exception:
            pass

    scan(obj)
    return findings


# ----- IO helpers -----

def load_pickle(path):
    with open(path, "rb") as f:
        data = f.read()
    for encoding in (None, "latin1", "bytes"):
        try:
            bio = io.BytesIO(data)
            if encoding is None:
                return pickle.load(bio)
            else:
                return pickle.load(bio, encoding=encoding)  # type: ignore[arg-type]
        except Exception:
            continue
    return pickle.loads(data)


def derive_audio_pickle_path(light_path: str) -> Optional[str]:
    """Heuristic: strip trailing _seedXYZ from basename and look in conformer_osci/audio/"""
    base = os.path.basename(light_path)
    dir_root = os.path.abspath(os.path.join(os.path.dirname(light_path), "../../../.."))
    # Expected audio directory relative to evaluation root
    audio_dir = os.path.join(dir_root, "data", "conformer_osci", "audio")
    name_no_ext = os.path.splitext(base)[0]
    # drop last underscore segment if it starts with 'seed'
    parts = name_no_ext.split("_")
    if parts and parts[-1].lower().startswith("seed"):
        parts = parts[:-1]
    audio_name = "_".join(parts) + ".pkl"
    candidate = os.path.join(audio_dir, audio_name)
    if os.path.exists(candidate):
        return candidate
    return None


def generate_report(section_title: str, pickle_path: str, obj, top_sum, tics: List[dict]) -> str:
    lines: List[str] = []
    lines.append(section_title)
    lines.append(f"File: {pickle_path}")
    try:
        size = os.path.getsize(pickle_path)
        lines.append(f"Size: {size} bytes")
    except Exception:
        pass
    lines.append("")
    lines.append("- Top-level Summary -")
    lines.append(pformat(top_sum, width=120))
    lines.append("")
    lines.append("- Heuristic 'tic/tick' Detection -")
    if tics:
        for i, item in enumerate(tics, 1):
            details = {k: v for k, v in item.items() if k not in {"path", "kind"}}
            details_str = ", ".join(f"{k}={v}" for k, v in details.items())
            lines.append(f"[{i}] path={item.get('path')} kind={item.get('kind')} details={{" + details_str + "}}")
    else:
        lines.append("No tic/tick-like structures detected by heuristics.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze LIGHT pickle and corresponding AUDIO pickle (if found)")
    parser.add_argument("pickle_path", help="Path to the LIGHT .pkl file")
    parser.add_argument("--max-depth", type=int, default=3, help="Max recursion depth for summaries")
    parser.add_argument("--out", default="content_light_pickle.txt", help="Output text file for combined report")
    args = parser.parse_args()

    # Normalize output filename to underscores
    args.out = args.out.replace(" ", "_")

    abs_light = os.path.abspath(args.pickle_path)
    print("=== Light Pickle Inspection ===")
    print(f"File: {abs_light}")
    try:
        size = os.path.getsize(abs_light)
        print(f"Size: {size} bytes")
    except Exception:
        pass
    print(f"Time: {datetime.now().isoformat()}")

    try:
        light_obj = load_pickle(abs_light)
    except Exception as e:
        print("\n[ERROR] Failed to load LIGHT pickle:", repr(e))
        sys.exit(1)

    print("\n--- LIGHT: Top-level Summary ---")
    light_sum = summarize(light_obj, max_depth=args.max_depth)
    print(pformat(light_sum, width=120))
    print("\n--- LIGHT: Heuristic 'tic/tick' Detection ---")
    light_tics = detect_tics(light_obj)
    if light_tics:
        for i, item in enumerate(light_tics, 1):
            details = {k: v for k, v in item.items() if k not in {"path", "kind"}}
            details_str = ", ".join(f"{k}={v}" for k, v in details.items())
            print(f"[{i}] path={item.get('path')} kind={item.get('kind')} details={{" + details_str + "}}")
    else:
        print("No tic/tick-like structures detected by heuristics.")

    # Build LIGHT-only report
    light_report = generate_report("=== LIGHT Pickle Inspection ===", abs_light, light_obj, light_sum, light_tics)
    light_report = "=== LIGHT Report ===\nGenerated: " + datetime.now().isoformat() + "\n\n" + light_report + "Done.\n"

    out_path = os.path.abspath(args.out)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(light_report)
        print(f"\n[INFO] LIGHT report written to: {out_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to write report to {out_path}: {e}")
        sys.exit(2)

    print("\nDone.")


if __name__ == "__main__":
    main()
