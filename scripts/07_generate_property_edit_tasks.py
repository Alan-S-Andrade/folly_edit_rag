#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
FOLLY_EDIT_RAG_ROOT = SCRIPT_DIR.parent
BENCHMARK_REPLICATION_ROOT = FOLLY_EDIT_RAG_ROOT.parent

sys.path.insert(0, str(BENCHMARK_REPLICATION_ROOT))

from run_feature_extraction_end_to_end import (  # noqa: E402
    choose_unit_cmd,
    collect_unit_features,
    list_microbenchmarks,
)
from feature_prompt_guidance import feature_prompt_guidance  # noqa: E402
from feature_schema import (  # noqa: E402
    CANONICAL_FEATURE_COLUMNS,
    canonicalize_dataframe,
    feature_display_name as canonical_feature_display_name,
)
from config.settings import (  # noqa: E402
    CMAKE_BUILD_DIR,
    COMPILE_LOGS_DIR,
    FOLLY_SRC_ROOT,
    OUTPUTS_DIR,
    SUCCESSFUL_EDITS_DIR,
    WDL_BENCH_ROOT,
)

DEFAULT_XLSX = BENCHMARK_REPLICATION_ROOT / "search_engine" / "all_features.xlsx"
DEFAULT_CMAKE = FOLLY_SRC_ROOT / "CMakeLists.txt"
WDL_SOURCES_ROOT = WDL_BENCH_ROOT / "wdl_sources"
FBTHRIFT_SRC_ROOT = WDL_SOURCES_ROOT / "fbthrift"
FBTHRIFT_BUILD_DIR = WDL_BENCH_ROOT / "wdl_build" / "build" / "fbthrift"
TASKS_DIR = FOLLY_EDIT_RAG_ROOT / "data" / "property_edit_tasks"
PROPERTY_RUNS_DIR = OUTPUTS_DIR / "property_edit_runs"
TASKS_DIR.mkdir(parents=True, exist_ok=True)
PROPERTY_RUNS_DIR.mkdir(parents=True, exist_ok=True)

ID_COLUMNS = {"binary", "binary_path", "command", "microbenchmark"}
STATUS_PREFIX = "status."
DEFAULT_TARGET_FEATURES = CANONICAL_FEATURE_COLUMNS
MAGNITUDE_BANDS = {
    "small": (0.10, 0.40),
    "medium": (0.40, 0.90),
    "large": (0.90, None),
}
LEGACY_MAGNITUDE_ORDER = ("minimally", "moderately", "substantially", "significantly")
DEFAULT_SMALL_BINARY_THRESHOLD = 64
DEFAULT_LARGE_BINARY_SAMPLE_CAP = 36
DEFAULT_LARGE_BINARY_SAMPLE_SCALE = 1.25
DEFAULT_MAX_CHILD_ATTEMPTS = 5


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(row)) + "\n")


def slugify(text: str, max_len: int = 96) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    if not slug:
        return "item"
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:10]
    if len(slug) <= max_len:
        return f"{slug}_{digest}"
    keep = max_len - len(digest) - 1
    return f"{slug[:keep].rstrip('_')}_{digest}"


def feature_display_name(feature_name: str) -> str:
    return canonical_feature_display_name(feature_name)


def binary_subject(binary_name: str) -> str:
    subject = binary_name.replace("_benchmark", " benchmark")
    subject = subject.replace("_bench", " bench")
    return subject.replace("_", " ").strip()


def direction_phrase(direction: str) -> str:
    return "increase" if direction == "increase" else "decrease"


def microbenchmark_tokens(name: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|\d+", name.lower())


def lexical_similarity(a: str, b: str) -> float:
    at = set(microbenchmark_tokens(a))
    bt = set(microbenchmark_tokens(b))
    token_score = len(at & bt) / max(len(at | bt), 1)
    prefix_len = len(os.path.commonprefix([a.lower(), b.lower()]))
    return token_score + (prefix_len / max(len(a), len(b), 1))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 or b_norm == 0.0:
        return -1.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def select_target_feature_columns(
    df: pd.DataFrame,
    requested_features: tuple[str, ...],
) -> list[str]:
    selected: list[str] = []
    missing: list[str] = []
    for column in requested_features:
        if column not in df.columns:
            missing.append(column)
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            selected.append(column)
    if missing:
        raise ValueError(f"Requested feature columns not found in XLSX: {missing}")
    return selected


def compute_feature_scale(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return 1.0
    if arr.size == 1:
        return max(abs(float(arr[0])) * 0.05, 1e-6)
    q10, q25, q50, q75, q90 = np.quantile(arr, [0.10, 0.25, 0.50, 0.75, 0.90])
    spread = max(
        float(q90 - q10),
        float(q75 - q25),
        float(np.std(arr)),
        abs(float(q50)) * 0.05,
        1e-6,
    )
    return spread


def format_signed_delta(value: float) -> str:
    return f"{value:+.6g}"


def format_range(lower: float, upper: float | None) -> str:
    if upper is None:
        return f">= {format_signed_delta(lower)}"
    return f"{format_signed_delta(lower)} to {format_signed_delta(upper)}"


def signed_delta_window(
    direction: str,
    lower: float,
    upper: float | None,
    center: float,
) -> tuple[float, float | None, float]:
    if direction == "increase":
        return lower, upper, center
    signed_lower = -upper if upper is not None else None
    signed_upper = -lower
    signed_center = -center
    return (
        signed_lower if signed_lower is not None else signed_upper,
        signed_upper,
        signed_center,
    )


def format_signed_target_window(
    direction: str,
    lower: float,
    upper: float | None,
    center: float,
) -> tuple[str, str]:
    if direction == "increase":
        return format_signed_delta(center), format_range(lower, upper)
    if upper is None:
        return format_signed_delta(-center), f"<= {format_signed_delta(-lower)}"
    signed_lower, signed_upper, signed_center = signed_delta_window(direction, lower, upper, center)
    return format_signed_delta(signed_center), format_range(signed_lower, signed_upper)


def load_successful_magnitude_profiles(
    feature_df: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    successful_root = SUCCESSFUL_EDITS_DIR
    if not successful_root.exists():
        return {}

    global_scales = {
        feature_name: compute_feature_scale(feature_df[feature_name])
        for feature_name in CANONICAL_FEATURE_COLUMNS
        if feature_name in feature_df.columns
    }
    observed_by_feature: dict[str, list[float]] = {}
    for meta_path in successful_root.glob("*/metadata.json"):
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not metadata.get("performance_success", False):
            continue
        feature_name = str(metadata.get("feature_name", ""))
        if feature_name not in global_scales:
            continue
        try:
            observed_delta = abs(float(metadata.get("observed_delta")))
        except Exception:
            continue
        scale = max(float(global_scales[feature_name]), 1e-9)
        observed_by_feature.setdefault(feature_name, []).append(observed_delta / scale)

    learned: dict[str, dict[str, float | None]] = {}
    for feature_name, ratios in observed_by_feature.items():
        arr = np.array(sorted(ratios), dtype=float)
        if arr.size == 1:
            center_medium = float(arr[0])
            center_small = max(center_medium * 0.5, 0.05)
            center_large = max(center_medium * 1.5, center_medium + 1e-6)
        elif arr.size == 2:
            center_small = float(arr[0])
            center_medium = float(np.mean(arr))
            center_large = max(float(arr[1]) * 1.25, center_medium + 1e-6)
        else:
            center_small = float(np.quantile(arr, 1.0 / 3.0))
            center_medium = float(np.quantile(arr, 0.50))
            center_large = float(np.quantile(arr, 2.0 / 3.0))

        lower_small = max(center_small * 0.5, 0.05)
        lower_medium = max((center_small + center_medium) / 2.0, lower_small + 1e-6)
        lower_large = max((center_medium + center_large) / 2.0, lower_medium + 1e-6)
        upper_small = lower_medium
        upper_medium = lower_large
        learned[feature_name] = {
            "small_lower": lower_small,
            "small_upper": upper_small,
            "small_center": center_small,
            "medium_lower": lower_medium,
            "medium_upper": upper_medium,
            "medium_center": center_medium,
            "large_lower": lower_large,
            "large_upper": None,
            "large_center": center_large,
        }
    return learned


def load_feature_table(xlsx_path: Path) -> pd.DataFrame:
    df = canonicalize_dataframe(pd.read_excel(xlsx_path))
    required = {"binary", "microbenchmark"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required XLSX columns: {sorted(missing)}")
    return df


def _normalized_feature_matrix(group: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    if not feature_columns:
        return np.zeros((len(group), 0), dtype=float)
    numeric = group[feature_columns].apply(pd.to_numeric, errors="coerce")
    arr = numeric.to_numpy(dtype=float)
    if arr.size == 0:
        return np.zeros((len(group), 0), dtype=float)

    col_means = np.nanmean(arr, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_rows, nan_cols = np.where(np.isnan(arr))
    if nan_rows.size:
        arr[nan_rows, nan_cols] = np.take(col_means, nan_cols)
    col_stds = np.nanstd(arr, axis=0)
    col_stds = np.where(col_stds > 1e-9, col_stds, 1.0)
    return (arr - col_means) / col_stds


def _microbenchmark_operation_family(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name.lstrip("%").strip())
    if not normalized:
        return "unknown"
    return normalized.split(" ", 1)[0].lower()


def _microbenchmark_container_family(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name.lstrip("%").strip())
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)<", normalized)
    if match:
        return match.group(1).lower()
    parts = normalized.split(" ")
    if len(parts) >= 2:
        return re.sub(r"[^a-zA-Z0-9_]+", "", parts[1]).lower() or "unknown"
    return "unknown"


def _microbenchmark_size_bucket(name: str) -> str:
    match = re.search(r"\[(\d+)\]\s*$", name.strip())
    if not match:
        return "unknown"
    size = int(match.group(1))
    if size <= 16:
        return "tiny"
    if size <= 128:
        return "small"
    if size <= 1024:
        return "medium"
    if size <= 8192:
        return "large"
    return "xlarge"


def _microbenchmark_diversity_key(name: str) -> tuple[str, str, str]:
    return (
        _microbenchmark_operation_family(name),
        _microbenchmark_container_family(name),
        _microbenchmark_size_bucket(name),
    )


def _squared_distance(matrix: np.ndarray, left: int, right: int) -> float:
    if matrix.shape[1] == 0:
        return 0.0
    diff = matrix[left] - matrix[right]
    return float(np.dot(diff, diff))


def _select_representative_position(
    group: pd.DataFrame,
    positions: list[int],
    matrix: np.ndarray,
) -> int:
    if len(positions) == 1 or matrix.shape[1] == 0:
        return min(positions, key=lambda pos: str(group.iloc[pos]["microbenchmark"]))

    centroid = np.mean(matrix[positions], axis=0)
    best_pos = positions[0]
    best_score = float("inf")
    best_name = str(group.iloc[best_pos]["microbenchmark"])
    for pos in positions:
        distance = float(np.dot(matrix[pos] - centroid, matrix[pos] - centroid))
        name = str(group.iloc[pos]["microbenchmark"])
        if distance < best_score - 1e-12 or (abs(distance - best_score) <= 1e-12 and name < best_name):
            best_score = distance
            best_pos = pos
            best_name = name
    return best_pos


def _greedy_diverse_select(
    group: pd.DataFrame,
    matrix: np.ndarray,
    candidate_positions: list[int],
    desired_total: int,
    seed_positions: list[int] | None = None,
) -> list[int]:
    selected = list(dict.fromkeys(seed_positions or []))
    selected_set = set(selected)
    available = [pos for pos in candidate_positions if pos not in selected_set]
    if desired_total <= len(selected):
        return selected[:desired_total]

    while len(selected) < desired_total and available:
        best_pos = available[0]
        best_score = float("-inf")
        best_name = str(group.iloc[best_pos]["microbenchmark"])
        if selected:
            for pos in available:
                score = min(_squared_distance(matrix, pos, chosen) for chosen in selected)
                name = str(group.iloc[pos]["microbenchmark"])
                if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and name < best_name):
                    best_pos = pos
                    best_score = score
                    best_name = name
        else:
            if matrix.shape[1] == 0:
                best_pos = min(available, key=lambda pos: str(group.iloc[pos]["microbenchmark"]))
            else:
                centroid = np.mean(matrix[available], axis=0)
                for pos in available:
                    score = float(np.dot(matrix[pos] - centroid, matrix[pos] - centroid))
                    name = str(group.iloc[pos]["microbenchmark"])
                    if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and name < best_name):
                        best_pos = pos
                        best_score = score
                        best_name = name
        selected.append(best_pos)
        available = [pos for pos in available if pos != best_pos]
    return selected


def _target_sample_size(
    count: int,
    small_binary_threshold: int,
    large_binary_sample_cap: int,
    large_binary_sample_scale: float,
) -> int:
    if count <= small_binary_threshold:
        return count
    scaled_budget = int(math.ceil(math.sqrt(count) * max(large_binary_sample_scale, 0.1)))
    return min(count, max(8, scaled_budget), max(1, large_binary_sample_cap))


def sample_microbenchmarks_for_binary(
    binary_name: str,
    group: pd.DataFrame,
    feature_columns: list[str],
    sample_large_binaries: bool,
    small_binary_threshold: int,
    large_binary_sample_cap: int,
    large_binary_sample_scale: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ordered = group.sort_values("microbenchmark", kind="stable").reset_index()
    total = len(ordered)
    if total == 0:
        return group, {
            "binary": binary_name,
            "mode": "empty",
            "total_microbenchmarks": 0,
            "selected_microbenchmarks": 0,
        }

    target_count = _target_sample_size(
        total,
        small_binary_threshold=small_binary_threshold,
        large_binary_sample_cap=large_binary_sample_cap,
        large_binary_sample_scale=large_binary_sample_scale,
    )
    if (not sample_large_binaries) or target_count >= total:
        selected_names = ordered["microbenchmark"].astype(str).tolist()
        return group, {
            "binary": binary_name,
            "mode": "all",
            "total_microbenchmarks": total,
            "selected_microbenchmarks": total,
            "selection_budget": total,
            "diversity_buckets": len({
                _microbenchmark_diversity_key(name)
                for name in selected_names
            }),
            "selected_microbenchmarks_preview": selected_names[:20],
        }

    matrix = _normalized_feature_matrix(ordered, feature_columns)
    buckets: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for pos, name in enumerate(ordered["microbenchmark"].astype(str).tolist()):
        buckets[_microbenchmark_diversity_key(name)].append(pos)

    representatives = [
        _select_representative_position(ordered, positions, matrix)
        for _, positions in sorted(buckets.items(), key=lambda item: item[0])
    ]
    if len(representatives) > target_count:
        selected_positions = _greedy_diverse_select(
            ordered,
            matrix,
            representatives,
            target_count,
        )
    else:
        selected_positions = _greedy_diverse_select(
            ordered,
            matrix,
            representatives,
            len(representatives),
        )
        selected_positions = _greedy_diverse_select(
            ordered,
            matrix,
            list(range(total)),
            target_count,
            seed_positions=selected_positions,
        )

    selected_positions = sorted(set(selected_positions))
    selected_indices = [int(ordered.iloc[pos]["index"]) for pos in selected_positions]
    sampled_group = group.loc[selected_indices].sort_values("microbenchmark", kind="stable")
    selected_names = sampled_group["microbenchmark"].astype(str).tolist()
    return sampled_group, {
        "binary": binary_name,
        "mode": "diverse_sampled",
        "total_microbenchmarks": total,
        "selected_microbenchmarks": len(selected_names),
        "selection_budget": target_count,
        "diversity_buckets": len(buckets),
        "selected_microbenchmarks_preview": selected_names[:20],
    }


def parse_sources_token_line(text: str) -> list[str]:
    out: list[str] = []
    for token in text.split():
        if token.endswith((".cpp", ".cc", ".cxx", ".c")):
            out.append(token)
    return out


def parse_benchmark_binary_sources(cmake_path: Path) -> dict[str, dict[str, Any]]:
    lines = cmake_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    current_dir = Path("folly")
    result: dict[str, dict[str, Any]] = {}
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        dir_match = re.match(r"^DIRECTORY\s+(.+?)/?\s*$", stripped)
        if dir_match:
            current_dir = Path("folly") / Path(dir_match.group(1).strip())
            i += 1
            continue

        bench_match = re.match(r"^BENCHMARK\s+(\S+)(.*)$", stripped)
        if not bench_match:
            i += 1
            continue

        binary_name = bench_match.group(1)
        stanza: list[str] = [stripped]
        j = i + 1
        while j < len(lines):
            next_stripped = lines[j].strip()
            if re.match(r"^(DIRECTORY|TEST|BENCHMARK)\b", next_stripped):
                break
            stanza.append(next_stripped)
            j += 1

        sources: list[str] = []
        capture_sources = False
        for stanza_line in stanza:
            if not stanza_line or stanza_line.startswith("#"):
                continue
            if "SOURCES" in stanza_line:
                after = stanza_line.split("SOURCES", 1)[1].strip()
                if after:
                    sources.extend(parse_sources_token_line(after))
                    capture_sources = False
                else:
                    capture_sources = True
                continue
            if capture_sources:
                if re.match(r"^(HEADERS|DEPENDS|EXTRA_ARGS|CONTENT_DIR|WINDOWS_DISABLED|APPLE_DISABLED|BROKEN|SLOW|HANGING)\b", stanza_line):
                    capture_sources = False
                    continue
                sources.extend(parse_sources_token_line(stanza_line))

        if sources:
            result[binary_name] = {
                "source_root": FOLLY_SRC_ROOT,
                "build_dir": CMAKE_BUILD_DIR,
                "source_rel_paths": [current_dir / src for src in sources],
            }
        i = j
    result.update(parse_fbthrift_binary_sources())
    return result


def parse_fbthrift_binary_sources() -> dict[str, dict[str, Any]]:
    return {
        "VarintUtilsBench": {
            "source_root": FBTHRIFT_SRC_ROOT,
            "build_dir": FBTHRIFT_BUILD_DIR,
            "source_rel_paths": [Path("thrift/lib/cpp/util/test/VarintUtilsBench.cpp")],
        },
        "ProtocolBench": {
            "source_root": FBTHRIFT_SRC_ROOT,
            "build_dir": FBTHRIFT_BUILD_DIR,
            "source_rel_paths": [Path("thrift/lib/cpp2/test/ProtocolBench.cpp")],
        },
    }


def choose_target_file(
    source_root: Path,
    source_rel_paths: list[Path],
    microbenchmark_name: str,
    text_cache: dict[Path, str],
) -> Path | None:
    if not source_rel_paths:
        return None
    if len(source_rel_paths) == 1:
        return source_rel_paths[0]

    best_path = source_rel_paths[0]
    best_score = float("-inf")
    for rel_path in source_rel_paths:
        abs_path = source_root / rel_path
        text = text_cache.setdefault(
            abs_path,
            abs_path.read_text(encoding="utf-8", errors="ignore") if abs_path.exists() else "",
        )
        score = 0.0
        if microbenchmark_name in text:
            score += 8.0
        score += text.count("BENCHMARK(")
        score += text.count("BENCHMARK_RELATIVE(")
        if "main(" in text:
            score += 0.5
        if "Benchmark" in rel_path.name or "Bench" in rel_path.name:
            score += 1.0
        if rel_path.name == "main.cpp":
            score += 0.25
        if score > best_score:
            best_score = score
            best_path = rel_path
    return best_path


def choose_anchor_names(
    group: pd.DataFrame,
    current_index: int,
    feature_columns: list[str],
    max_anchors: int = 2,
) -> list[str]:
    current = group.loc[current_index]
    others = group[group.index != current_index]
    if others.empty:
        return []

    anchors: list[str] = []
    current_name = str(current["microbenchmark"])

    lexical_name = max(
        (
            (lexical_similarity(current_name, str(row["microbenchmark"])), str(row["microbenchmark"]))
            for _, row in others.iterrows()
        ),
        key=lambda item: item[0],
    )[1]
    if lexical_name != current_name:
        anchors.append(lexical_name)

    current_vec = pd.to_numeric(current[feature_columns], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    scored: list[tuple[float, str]] = []
    for _, row in others.iterrows():
        other_name = str(row["microbenchmark"])
        if other_name in anchors:
            continue
        other_vec = pd.to_numeric(row[feature_columns], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        scored.append((cosine_similarity(current_vec, other_vec), other_name))
    if scored:
        feature_name = max(scored, key=lambda item: item[0])[1]
        if feature_name != current_name and feature_name not in anchors:
            anchors.append(feature_name)

    return anchors[:max_anchors]


def build_task_text(
    binary_name: str,
    target_file: str,
    microbenchmark_name: str,
    anchors: list[str],
    feature_name: str,
    direction: str,
    magnitude: str,
    baseline_value: float,
    target_delta_lower_bound: float,
    target_delta_upper_bound: float | None,
    target_delta_center: float,
) -> str:
    display = feature_display_name(feature_name)
    subject = binary_subject(binary_name)
    signed_center_text, signed_range_text = format_signed_target_window(
        direction,
        target_delta_lower_bound,
        target_delta_upper_bound,
        target_delta_center,
    )
    anchor_clause = ""
    if anchors:
        anchor_clause = " and nearby variants such as " + ", ".join(anchors)
    guidance_lines = feature_prompt_guidance(feature_name, direction)

    lines = [
        f"Add a new {subject} variant to {Path(target_file).name} for the {binary_name} binary.",
        f"Keep the coding style and benchmark registration patterns consistent with the existing {microbenchmark_name} benchmark{anchor_clause}.",
        "Do not delete, rename, or weaken existing benchmarks; preserve the existing benchmark harness and main() unless a build fix absolutely requires a local adjustment.",
        f"The new variant should appear as an additional microbenchmark in --bm_list and should {direction_phrase(direction)} {display} with a {magnitude} change relative to {microbenchmark_name}.",
        f"Target delta for {display}: about {signed_center_text} with acceptable range {signed_range_text}.",
        f"Current baseline for {microbenchmark_name}: {display}={baseline_value:.6g}.",
    ]
    lines.extend(guidance_lines)
    return " ".join(lines)


def build_retrieval_query(
    task_text: str,
    target_file: str,
    binary_name: str,
    microbenchmark_name: str,
    anchors: list[str],
    feature_name: str,
    direction: str,
    magnitude: str,
) -> str:
    anchor_text = ", ".join(anchors) if anchors else "none"
    return f"""You are retrieving context for a Folly benchmark editing task.

Task:
{task_text}

Target file:
{target_file}

Target binary:
{binary_name}

Source microbenchmark:
{microbenchmark_name}

Style anchors:
{anchor_text}

Target metric:
{feature_name}

Requested change:
{direction} / {magnitude}

Retrieve the most useful context for making a minimal, compilable source edit that adds a new benchmark variant.
Prioritize:
- the exact target file
- the benchmark block or helper code closest to {microbenchmark_name}
- sibling benchmark blocks for {anchor_text}
- the matching CMake BENCHMARK mapping for {binary_name}
- nearby helper utilities and registration patterns
- prior successful edits that add new variants in the same file or binary
"""


def task_bounds(
    feature_name: str,
    scale: float,
    magnitude: str,
    learned_profiles: dict[str, dict[str, float | None]],
) -> tuple[float, float | None, float]:
    profile = learned_profiles.get(feature_name)
    if profile:
        return (
            float(profile[f"{magnitude}_lower"]),
            None if profile[f"{magnitude}_upper"] is None else float(profile[f"{magnitude}_upper"]),
            float(profile[f"{magnitude}_center"]),
        )

    lower_factor, upper_factor = MAGNITUDE_BANDS[magnitude]
    center_factor = (lower_factor + (upper_factor if upper_factor is not None else lower_factor * 1.5)) / 2.0
    return (
        lower_factor * scale,
        None if upper_factor is None else upper_factor * scale,
        center_factor * scale,
    )


def build_task_manifest(
    xlsx_path: Path,
    cmake_path: Path,
    requested_features: tuple[str, ...],
    binary_filter: set[str] | None,
    microbenchmark_filter: set[str] | None,
    feature_filter: set[str] | None,
    sample_large_binaries: bool,
    small_binary_threshold: int,
    large_binary_sample_cap: int,
    large_binary_sample_scale: float,
    max_tasks: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df = load_feature_table(xlsx_path)
    feature_columns = select_target_feature_columns(df, requested_features)
    source_map = parse_benchmark_binary_sources(cmake_path)
    learned_profiles = load_successful_magnitude_profiles(df)
    text_cache: dict[Path, str] = {}

    tasks: list[dict[str, Any]] = []
    sampling_summary: list[dict[str, Any]] = []
    for binary_name, group in df.groupby("binary", sort=True):
        if binary_filter and binary_name not in binary_filter:
            continue
        if binary_name not in source_map:
            continue
        if microbenchmark_filter:
            group = group[group["microbenchmark"].astype(str).isin(microbenchmark_filter)]
        if group.empty:
            continue

        group, sample_info = sample_microbenchmarks_for_binary(
            binary_name=binary_name,
            group=group,
            feature_columns=feature_columns,
            sample_large_binaries=sample_large_binaries,
            small_binary_threshold=small_binary_threshold,
            large_binary_sample_cap=large_binary_sample_cap,
            large_binary_sample_scale=large_binary_sample_scale,
        )
        sampling_summary.append(sample_info)

        scales = {
            feature_name: compute_feature_scale(group[feature_name])
            for feature_name in feature_columns
        }

        source_info = source_map[binary_name]
        source_root = Path(source_info["source_root"])
        build_dir = Path(source_info["build_dir"])
        source_rel_paths = list(source_info["source_rel_paths"])
        for index, row in group.iterrows():
            microbenchmark_name = str(row["microbenchmark"])

            target_rel_path = choose_target_file(
                source_root,
                source_rel_paths,
                microbenchmark_name,
                text_cache,
            )
            if target_rel_path is None:
                continue

            anchors = choose_anchor_names(group, index, feature_columns)
            for feature_name in feature_columns:
                if feature_filter and feature_name not in feature_filter:
                    continue
                baseline_value_raw = pd.to_numeric(pd.Series([row[feature_name]]), errors="coerce").iloc[0]
                if pd.isna(baseline_value_raw):
                    continue
                baseline_value = float(baseline_value_raw)
                scale = scales[feature_name]
                for direction in ("increase", "decrease"):
                    for magnitude in MAGNITUDE_BANDS:
                        lower_bound, upper_bound, target_center = task_bounds(
                            feature_name,
                            scale,
                            magnitude,
                            learned_profiles,
                        )
                        task_text = build_task_text(
                            binary_name=binary_name,
                            target_file=str(target_rel_path),
                            microbenchmark_name=microbenchmark_name,
                            anchors=anchors,
                            feature_name=feature_name,
                            direction=direction,
                            magnitude=magnitude,
                            baseline_value=baseline_value,
                            target_delta_lower_bound=lower_bound,
                            target_delta_upper_bound=upper_bound,
                            target_delta_center=target_center,
                        )
                        task_id = slugify(
                            f"{binary_name}_{microbenchmark_name}_{feature_name}_{direction}_{magnitude}"
                        )
                        task = {
                            "task_id": task_id,
                            "task": task_text,
                            "binary": binary_name,
                            "target_file": str(target_rel_path),
                            "source_root": str(source_root),
                            "build_dir": str(build_dir),
                            "source_microbenchmark": microbenchmark_name,
                            "style_anchors": anchors,
                            "feature_name": feature_name,
                            "feature_display_name": feature_display_name(feature_name),
                            "direction": direction,
                            "magnitude": magnitude,
                            "baseline_value": baseline_value,
                            "scale": scale,
                            "target_delta_lower_bound": lower_bound,
                            "target_delta_upper_bound": upper_bound,
                            "target_delta_center": target_center,
                        }
                        tasks.append(task)
                        if max_tasks is not None and len(tasks) >= max_tasks:
                            return tasks, sampling_summary
    return tasks, sampling_summary


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


CHILD_TARGET_ORDER = tuple(
    f"{direction}_{magnitude}"
    for direction in ("increase", "decrease")
    for magnitude in MAGNITUDE_BANDS
)


def child_target_key(direction: str, magnitude: str) -> str:
    return f"{direction}_{magnitude}"


def expected_benchmark_suffix(direction: str, magnitude: str) -> str:
    return f"__wdl_{direction}_{magnitude}"


def build_grouped_job_id(row: dict[str, Any]) -> str:
    return slugify(
        f"{row['binary']}_{row['source_microbenchmark']}_{row['feature_name']}_{row['target_file']}"
    )


def build_group_task_text(group: dict[str, Any]) -> str:
    display = feature_display_name(group["feature_name"])
    child_lines = [
        (
            lambda signed_center_text, signed_range_text: (
                f"{child['direction']} {display} with a {child['magnitude']} change "
                f"(target delta about {signed_center_text}, "
                f"acceptable range {signed_range_text}) "
                f"using benchmark-name suffix {child['expected_suffix']}"
            )
        )(
            *format_signed_target_window(
                str(child["direction"]),
                float(child["target_delta_lower_bound"]),
                child["target_delta_upper_bound"],
                float(child["target_delta_center"]),
            )
        )
        for child in group["child_tasks"].values()
    ]
    anchor_clause = ""
    if group.get("style_anchors"):
        anchor_clause = " and nearby variants such as " + ", ".join(group["style_anchors"])
    return " ".join(
        [
            f"Produce {len(group['child_tasks'])} isolated full-file candidates for {Path(group['target_file']).name} in the {group['binary']} binary.",
            f"Each candidate must add exactly one new variant based on the existing {group['source_microbenchmark']} benchmark{anchor_clause}.",
            "Each candidate must preserve existing benchmarks, benchmark harness structure, and main() unless a build fix absolutely requires a local adjustment.",
            "The child targets cover increase/decrease crossed with explicit numeric delta bands for the same feature.",
            "Each candidate must include the required benchmark-name suffix for its child target so it can be compiled and evaluated independently.",
            "Child targets:",
            "; ".join(child_lines) + ".",
        ]
    )


def group_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["target_file"]),
            str(row["binary"]),
            str(row["source_microbenchmark"]),
            str(row["feature_name"]),
        )
        child_key = child_target_key(str(row["direction"]), str(row["magnitude"]))
        group = grouped.get(key)
        if group is None:
            group = {
                "grouped_job_id": build_grouped_job_id(row),
                "binary": row["binary"],
                "target_file": row["target_file"],
                "source_root": row["source_root"],
                "build_dir": row["build_dir"],
                "source_microbenchmark": row["source_microbenchmark"],
                "style_anchors": list(row.get("style_anchors", [])),
                "feature_name": row["feature_name"],
                "feature_display_name": row["feature_display_name"],
                "child_tasks": {},
            }
            grouped[key] = group

        child = dict(row)
        child["child_key"] = child_key
        child["expected_suffix"] = expected_benchmark_suffix(
            str(row["direction"]),
            str(row["magnitude"]),
        )
        group["child_tasks"][child_key] = child

    jobs = []
    for group in grouped.values():
        available_keys = list(group["child_tasks"].keys())
        preferred_order = [child_key for child_key in CHILD_TARGET_ORDER if child_key in group["child_tasks"]]
        if preferred_order:
            remaining = sorted(child_key for child_key in available_keys if child_key not in preferred_order)
            ordered_keys = preferred_order + remaining
            missing_child_keys = [
                child_key for child_key in CHILD_TARGET_ORDER if child_key not in group["child_tasks"]
            ]
        else:
            ordered_keys = sorted(available_keys)
            missing_child_keys = []
        ordered_children = {
            child_key: group["child_tasks"][child_key]
            for child_key in ordered_keys
        }
        group["child_tasks"] = ordered_children
        group["missing_child_keys"] = missing_child_keys
        group["task"] = build_group_task_text(group)
        jobs.append(group)
    jobs.sort(key=lambda item: str(item["grouped_job_id"]))
    return jobs


def run_cmd_logged(cmd: list[str], log_path: Path, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )
    payload = {
        "cmd": cmd,
        "cwd": str(cwd) if cwd else None,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    write_json(log_path, payload)
    return proc


def status(msg: str) -> None:
    print(msg, flush=True)


def band_success(abs_delta: float, lower: float, upper: float | None, tolerance: float) -> bool:
    if abs_delta + tolerance < lower:
        return False
    if upper is None:
        return True
    return abs_delta - tolerance < upper


def evaluate_candidate_row(task: dict[str, Any], candidate_row: dict[str, Any]) -> dict[str, Any]:
    feature_name = task["feature_name"]
    baseline_value = float(task["baseline_value"])
    candidate_value = candidate_row.get(feature_name)
    if candidate_value is None:
        return {
            "feature_present": False,
            "success": False,
            "reason": f"Candidate row did not contain {feature_name}",
        }

    candidate_value = float(candidate_value)
    observed_delta = candidate_value - baseline_value
    desired_sign = 1.0 if task["direction"] == "increase" else -1.0
    signed_delta = observed_delta * desired_sign
    abs_delta = abs(observed_delta)
    scale = float(task["scale"])
    lower = float(task["target_delta_lower_bound"])
    upper = task["target_delta_upper_bound"]
    upper = None if upper is None else float(upper)
    tolerance = max(scale * 0.05, 1e-9)
    direction_ok = signed_delta >= -tolerance
    magnitude_ok = band_success(abs_delta, lower, upper, tolerance) if direction_ok else False
    target_center = float(task["target_delta_center"])
    center_distance = abs(abs_delta - target_center)
    if direction_ok and magnitude_ok:
        score = 100.0 - (center_distance / max(scale, 1e-9))
    elif direction_ok:
        score = 10.0 - (center_distance / max(scale, 1e-9))
    else:
        score = -100.0 - (center_distance / max(scale, 1e-9))

    return {
        "feature_present": True,
        "success": bool(direction_ok and magnitude_ok),
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
        "observed_delta": observed_delta,
        "observed_abs_delta": abs_delta,
        "direction_success": bool(direction_ok),
        "magnitude_success": bool(magnitude_ok),
        "requested_lower_bound": lower,
        "requested_upper_bound": upper,
        "requested_center": target_center,
        "tolerance": tolerance,
        "score": score,
    }


def evaluate_task_run(
    task: dict[str, Any],
    baseline_microbenchmarks: list[str],
    binary_path: Path,
    perfspect_bin: str,
    perf_events: str,
    amd: bool,
    output_dir: Path,
) -> dict[str, Any]:
    baseline_set = set(baseline_microbenchmarks)
    listed, has_bm_list = list_microbenchmarks(binary_path)
    listed = sorted(set(listed))
    new_microbenchmarks = [name for name in listed if name not in baseline_set]

    evaluation: dict[str, Any] = {
        "binary_path": str(binary_path),
        "has_bm_list": has_bm_list,
        "listed_microbenchmarks": listed,
        "new_microbenchmarks": new_microbenchmarks,
        "candidate_results": [],
        "success": False,
    }
    if not new_microbenchmarks:
        evaluation["reason"] = "No new microbenchmark names appeared in --bm_list after the edit."
        return evaluation

    target_is_intel_pt = str(task["feature_name"]).startswith("intel_pt.")
    skip_intel_pt = amd or not target_is_intel_pt

    best_result: dict[str, Any] | None = None
    for microbenchmark_name in new_microbenchmarks:
        candidate_dir = output_dir / slugify(microbenchmark_name)
        cmd = choose_unit_cmd(binary_path, microbenchmark_name, has_bm_list)
        row = collect_unit_features(
            binary_path=binary_path,
            unit_name=microbenchmark_name,
            cmd=cmd,
            unit_output_dir=candidate_dir,
            perfspect_bin=perfspect_bin,
            perf_events=perf_events,
            skip_intel_pt=skip_intel_pt,
            requested_feature=task["feature_name"],
        )
        candidate_eval = evaluate_candidate_row(task, row)
        result = {
            "microbenchmark": microbenchmark_name,
            "command": cmd,
            "features": row,
            "evaluation": candidate_eval,
        }
        evaluation["candidate_results"].append(result)
        if best_result is None or candidate_eval.get("score", float("-inf")) > best_result["evaluation"].get("score", float("-inf")):
            best_result = result

    evaluation["best_candidate"] = best_result
    evaluation["success"] = bool(best_result and best_result["evaluation"].get("success"))
    return evaluation


def resolve_binary_path(build_dir: Path, binary_name: str, source_root: str) -> Path:
    source_root_lower = str(source_root).lower()
    if "fbthrift" in build_dir.name.lower() or "fbthrift" in source_root_lower:
        return (build_dir / "bin" / binary_name).resolve()
    return (build_dir / binary_name).resolve()


def evaluate_group_child(
    child_task: dict[str, Any],
    baseline_microbenchmarks: list[str],
    binary_path: Path,
    perfspect_bin: str,
    perf_events: str,
    amd: bool,
    output_dir: Path,
) -> dict[str, Any]:
    baseline_set = set(baseline_microbenchmarks)
    listed, has_bm_list = list_microbenchmarks(binary_path)
    listed = sorted(set(listed))
    new_microbenchmarks = [name for name in listed if name not in baseline_set]
    expected_suffix = str(child_task["expected_suffix"])
    matching_new = [name for name in new_microbenchmarks if expected_suffix in name]

    evaluation: dict[str, Any] = {
        "binary_path": str(binary_path),
        "has_bm_list": has_bm_list,
        "listed_microbenchmarks": listed,
        "new_microbenchmarks": new_microbenchmarks,
        "expected_suffix": expected_suffix,
        "matching_new_microbenchmarks": matching_new,
        "success": False,
    }
    if len(new_microbenchmarks) != 1:
        evaluation["reason"] = "Candidate file did not add exactly one new microbenchmark."
        return evaluation
    if len(matching_new) != 1:
        evaluation["reason"] = "New microbenchmark name did not contain the expected deterministic suffix."
        return evaluation

    microbenchmark_name = matching_new[0]
    target_is_intel_pt = str(child_task["feature_name"]).startswith("intel_pt.")
    skip_intel_pt = amd or not target_is_intel_pt
    cmd = choose_unit_cmd(binary_path, microbenchmark_name, has_bm_list)
    row = collect_unit_features(
        binary_path=binary_path,
        unit_name=microbenchmark_name,
        cmd=cmd,
        unit_output_dir=output_dir / slugify(microbenchmark_name),
        perfspect_bin=perfspect_bin,
        perf_events=perf_events,
        skip_intel_pt=skip_intel_pt,
        requested_feature=child_task["feature_name"],
    )
    candidate_eval = evaluate_candidate_row(child_task, row)
    evaluation["candidate_result"] = {
        "microbenchmark": microbenchmark_name,
        "command": cmd,
        "features": row,
        "evaluation": candidate_eval,
    }
    evaluation["success"] = bool(candidate_eval.get("success"))
    if not evaluation["success"]:
        evaluation["reason"] = (
            candidate_eval.get("reason")
            or "Compiled candidate did not satisfy the requested feature delta."
        )
    return evaluation


def _trim_feedback_text(text: str, max_lines: int = 24, max_chars: int = 2400) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    lines = cleaned.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    clipped = "\n".join(lines).strip()
    if len(clipped) > max_chars:
        clipped = clipped[-max_chars:]
    return clipped


def build_child_retry_feedback(
    child_task: dict[str, Any],
    child_summary: dict[str, Any],
) -> str:
    attempts = child_summary.get("attempts") or []
    if not attempts:
        return ""

    last_attempt = attempts[-1]
    compile_record = last_attempt.get("compile_record") or {}
    if not bool(compile_record.get("success")):
        compile_attempts = compile_record.get("compile_attempts") or []
        patch_apply_failed = any(
            attempt.get("patch_apply_success") is False for attempt in compile_attempts
        )
        reached_compile = any("compile_rc" in attempt for attempt in compile_attempts)
        error_blob = _trim_feedback_text(
            "\n".join(
                part
                for part in (
                    str(compile_record.get("second_patch_apply_summary", "")),
                    str(compile_record.get("first_patch_apply_summary", "")),
                    str(compile_record.get("second_patch_apply_stderr", "")),
                    str(compile_record.get("second_patch_apply_stdout", "")),
                    str(compile_record.get("first_patch_apply_stderr", "")),
                    str(compile_record.get("first_patch_apply_stdout", "")),
                    str(compile_record.get("second_compile_stderr", "")),
                    str(compile_record.get("second_compile_stdout", "")),
                    str(compile_record.get("first_compile_stderr", "")),
                    str(compile_record.get("first_compile_stdout", "")),
                )
                if part.strip()
            )
        )
        if patch_apply_failed and not reached_compile:
            if error_blob:
                return (
                    f"Attempt {last_attempt['attempt_index']} produced a patch that did not apply cleanly. "
                    f"Preserve the same child intent but fix the unified diff format and patch application issues below. "
                    f"Return only a valid unified diff with correct ---/+++ file headers, @@ hunk headers, and context lines copied exactly from the target file. "
                    f"Do not include prose or code fences.\n"
                    f"{error_blob}"
                )
            return (
                f"Attempt {last_attempt['attempt_index']} produced a patch that did not apply cleanly. "
                f"Preserve the same child intent, but return only a valid unified diff with correct ---/+++ file headers, @@ hunk headers, and exact target-file context lines."
            )
        if error_blob:
            return (
                f"Attempt {last_attempt['attempt_index']} did not compile. Preserve the same child intent but fix the build issues below and avoid unrelated changes:\n"
                f"{error_blob}"
            )
        return f"Attempt {last_attempt['attempt_index']} did not compile. Preserve the same child intent and make a simpler compilable edit."

    feature_eval = last_attempt.get("feature_evaluation") or {}
    candidate_result = feature_eval.get("candidate_result") or {}
    evaluation = candidate_result.get("evaluation") or {}
    candidate_value = evaluation.get("candidate_value")
    observed_delta = evaluation.get("observed_delta")
    lower = child_task.get("target_delta_lower_bound")
    upper = child_task.get("target_delta_upper_bound")
    center = child_task.get("target_delta_center")
    desired_sign = 1.0 if child_task["direction"] == "increase" else -1.0
    signed_delta = None
    if observed_delta is not None:
        signed_delta = float(observed_delta) * desired_sign

    gap_text = "Move the candidate closer to the requested target band."
    if signed_delta is not None:
        if signed_delta < float(lower):
            gap_text = "The change was too weak. Push the requested feature further in the same direction."
        elif upper is not None and abs(float(observed_delta)) >= float(upper):
            gap_text = "The change overshot the requested band. Back off and make a smaller, more targeted adjustment."
        elif signed_delta < 0:
            gap_text = "The feature moved in the wrong direction. Reverse the tactic while keeping the edit local."

    observed_parts = []
    if candidate_result.get("microbenchmark"):
        observed_parts.append(f"new benchmark name {candidate_result['microbenchmark']}")
    if candidate_value is not None:
        observed_parts.append(f"candidate value {float(candidate_value):.6g}")
    if observed_delta is not None:
        observed_parts.append(f"observed delta {format_signed_delta(float(observed_delta))}")
    observed_summary = ", ".join(observed_parts) if observed_parts else "no measured value available"
    target_center_text, target_range_text = format_signed_target_window(
        str(child_task["direction"]),
        float(lower),
        None if upper is None else float(upper),
        float(center),
    )
    return (
        f"Attempt {last_attempt['attempt_index']} compiled but missed the target for {child_task['feature_display_name']}. "
        f"Observed {observed_summary}. Requested delta about {target_center_text} with acceptable range {target_range_text}. "
        f"{gap_text}"
    )


def save_group_success(
    group: dict[str, Any],
    winner_key: str,
    winner_record: dict[str, Any],
    winner_evaluation: dict[str, Any],
    retrieval_name: str,
    current_file: Path,
    evaluation_path: Path,
) -> None:
    edit_dir = SUCCESSFUL_EDITS_DIR / str(group["grouped_job_id"])
    if edit_dir.exists():
        shutil.rmtree(edit_dir)
    edit_dir.mkdir(parents=True, exist_ok=True)

    original_text = current_file.read_text(encoding="utf-8", errors="ignore")
    final_path = Path(str(winner_record["compile_record"]["final_file"]))
    final_text = final_path.read_text(encoding="utf-8", errors="ignore")
    child_task = group["child_tasks"][winner_key]
    candidate_eval = (winner_evaluation.get("candidate_result") or {}).get("evaluation") or {}
    metadata = {
        "grouped_job_id": group["grouped_job_id"],
        "task_id": child_task["task_id"],
        "task": child_task["task"],
        "target_file": group["target_file"],
        "binary_name": group["binary"],
        "source_microbenchmark": group["source_microbenchmark"],
        "feature_name": child_task["feature_name"],
        "direction": child_task["direction"],
        "magnitude": child_task["magnitude"],
        "baseline_value": child_task["baseline_value"],
        "candidate_value": candidate_eval.get("candidate_value"),
        "observed_delta": candidate_eval.get("observed_delta"),
        "new_microbenchmark_name": (winner_evaluation.get("candidate_result") or {}).get("microbenchmark"),
        "retrieval_json": retrieval_name,
        "performance_success": True,
        "evaluation_path": str(evaluation_path),
        "expected_suffix": child_task["expected_suffix"],
        "winning_attempt_index": winner_record.get("winning_attempt_index"),
    }
    (edit_dir / "original.cpp").write_text(original_text, encoding="utf-8")
    (edit_dir / "final.cpp").write_text(final_text, encoding="utf-8")
    (edit_dir / "metadata.json").write_text(
        json.dumps(json_ready(metadata), indent=2),
        encoding="utf-8",
    )
    write_json(edit_dir / "task.json", child_task)


def run_grouped_task_pipeline(
    group: dict[str, Any],
    baseline_microbenchmarks: list[str],
    top_k: int,
    perfspect_bin: str,
    perf_events: str,
    amd: bool,
    skip_existing: bool,
    max_child_attempts: int,
) -> dict[str, Any]:
    grouped_job_id = str(group["grouped_job_id"])
    run_dir = PROPERTY_RUNS_DIR / grouped_job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    task_json = run_dir / "task.json"
    write_json(task_json, group)

    evaluation_path = run_dir / "evaluation.json"
    if skip_existing and evaluation_path.exists():
        return json.loads(evaluation_path.read_text(encoding="utf-8"))

    current_file = Path(str(group.get("source_root", str(FOLLY_SRC_ROOT)))) / str(group["target_file"])
    result: dict[str, Any] = {
        "grouped_job_id": grouped_job_id,
        "task": group["task"],
        "target_file": group["target_file"],
        "binary": group["binary"],
        "feature_name": group["feature_name"],
        "max_child_attempts": max_child_attempts,
        "children": {},
        "success": False,
    }

    if group.get("missing_child_keys"):
        result["reason"] = f"Incomplete grouped job; missing child targets: {group['missing_child_keys']}"
        write_json(evaluation_path, result)
        return result

    for child_key, child_task in group["child_tasks"].items():
        result["children"][child_key] = {
            "task_id": child_task["task_id"],
            "direction": child_task["direction"],
            "magnitude": child_task["magnitude"],
            "expected_suffix": child_task["expected_suffix"],
            "attempts": [],
            "attempt_count": 0,
            "success": False,
        }

    retrieval_name = f"{grouped_job_id}_retrieval.json"
    status(f"[grouped-job] {grouped_job_id}: retrieve start")
    retrieve_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "03_retrieve_edit_context.py"),
        "--task",
        group["task"],
        "--target-file",
        str(group["target_file"]),
        "--binary-name",
        str(group["binary"]),
        "--top-k",
        str(top_k),
        "--output",
        retrieval_name,
    ]
    retrieve_proc = run_cmd_logged(retrieve_cmd, run_dir / "01_retrieve.json", cwd=SCRIPT_DIR)
    result["retrieve"] = {"returncode": retrieve_proc.returncode, "retrieval_json": retrieval_name}
    if retrieve_proc.returncode != 0:
        status(f"[grouped-job] {grouped_job_id}: retrieve failed")
        result["reason"] = "Retrieval step failed."
        write_json(evaluation_path, result)
        return result
    status(f"[grouped-job] {grouped_job_id}: retrieve complete")

    winner_key: str | None = None
    winner_score = float("-inf")
    build_dir = Path(str(group.get("build_dir", str(CMAKE_BUILD_DIR)))).resolve()
    binary_path = resolve_binary_path(build_dir, str(group["binary"]), str(group["source_root"]))
    pending_child_keys = list(group["child_tasks"].keys())
    retry_feedback: dict[str, str] = {}
    result["generation_attempts"] = []

    for attempt_index in range(1, max_child_attempts + 1):
        if not pending_child_keys:
            break
        status(
            f"[grouped-job] {grouped_job_id}: attempt {attempt_index}/{max_child_attempts} "
            f"start for children {', '.join(pending_child_keys)}"
        )

        attempt_group = json.loads(json.dumps(group))
        attempt_group["child_tasks"] = {
            child_key: attempt_group["child_tasks"][child_key]
            for child_key in pending_child_keys
        }
        attempt_group["attempt_index"] = attempt_index
        attempt_group["max_child_attempts"] = max_child_attempts
        attempt_group["retry_feedback"] = {
            child_key: retry_feedback[child_key]
            for child_key in pending_child_keys
            if retry_feedback.get(child_key)
        }
        attempt_task_json = run_dir / f"task_attempt_{attempt_index}.json"
        write_json(attempt_task_json, attempt_group)

        generated_manifest_name = f"{grouped_job_id}_attempt_{attempt_index}_grouped_candidates.json"
        generate_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "04_generate_full_file.py"),
            "--target-file",
            str(group["target_file"]),
            "--current-file",
            str(current_file),
            "--retrieval-json",
            retrieval_name,
            "--grouped-task-json",
            str(attempt_task_json),
            "--example-id",
            f"{grouped_job_id}__attempt_{attempt_index}",
            "--output-manifest",
            generated_manifest_name,
        ]
        generate_proc = run_cmd_logged(
            generate_cmd,
            run_dir / f"02_generate_attempt_{attempt_index}.json",
            cwd=SCRIPT_DIR,
        )
        if generate_proc.returncode != 0:
            status(
                f"[grouped-job] {grouped_job_id}: attempt {attempt_index} generation failed"
            )
        else:
            status(
                f"[grouped-job] {grouped_job_id}: attempt {attempt_index} generation complete"
            )
        result["generation_attempts"].append(
            {
                "attempt_index": attempt_index,
                "returncode": generate_proc.returncode,
                "manifest": generated_manifest_name,
                "child_keys": list(pending_child_keys),
            }
        )

        attempt_generation_reason: str | None = None
        generated_manifest: dict[str, Any] | None = None
        if generate_proc.returncode != 0:
            attempt_generation_reason = "Grouped generation step failed."
        else:
            generated_manifest_path = SCRIPT_DIR / "outputs" / "repaired" / generated_manifest_name
            if not generated_manifest_path.exists():
                generated_manifest_path = OUTPUTS_DIR / "repaired" / generated_manifest_name
            if not generated_manifest_path.exists():
                attempt_generation_reason = "Grouped generation did not produce a candidate manifest."
            else:
                generated_manifest = json.loads(generated_manifest_path.read_text(encoding="utf-8"))

        next_pending: list[str] = []
        for child_key in pending_child_keys:
            child_task = group["child_tasks"][child_key]
            child_result = result["children"][child_key]
            status(
                f"[grouped-job] {grouped_job_id}: attempt {attempt_index} child {child_key} compile start"
            )
            attempt_record: dict[str, Any] = {
                "attempt_index": attempt_index,
                "candidate_manifest": generated_manifest_name,
                "generation_returncode": generate_proc.returncode,
                "success": False,
            }
            if attempt_generation_reason is not None:
                attempt_record["reason"] = attempt_generation_reason
                child_result["attempts"].append(attempt_record)
                child_result["attempt_count"] = len(child_result["attempts"])
                child_result["reason"] = attempt_generation_reason
                next_pending.append(child_key)
                retry_feedback[child_key] = (
                    f"Attempt {attempt_index} failed before candidate generation completed. Produce a simpler, deterministic full-file rewrite for this child target."
                )
                continue

            candidate_info = (generated_manifest.get("candidates") or {}).get(child_key) if generated_manifest else None
            if not candidate_info:
                attempt_record["reason"] = "Grouped generation did not return a candidate for this child target."
                child_result["attempts"].append(attempt_record)
                child_result["attempt_count"] = len(child_result["attempts"])
                child_result["reason"] = attempt_record["reason"]
                next_pending.append(child_key)
                retry_feedback[child_key] = (
                    f"Attempt {attempt_index} omitted this child target. Return a complete file for {child_key} with the required deterministic suffix {child_task['expected_suffix']}."
                )
                continue

            child_example_id = f"{grouped_job_id}__{child_key}__attempt_{attempt_index}"
            candidate_file = str(candidate_info["candidate_file"])
            compile_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "05_rewrite_compile_repair.py"),
                "--task",
                str(child_task["task"]),
                "--target-file",
                str(group["target_file"]),
                "--binary-name",
                str(group["binary"]),
                "--current-file",
                str(current_file),
                "--build-dir",
                str(build_dir),
                "--retrieval-json",
                retrieval_name,
                "--rewritten-file",
                candidate_file,
                "--example-id",
                child_example_id,
                "--no-save-success-example",
            ]
            compile_proc = run_cmd_logged(
                compile_cmd,
                run_dir / f"03_compile_{child_key}_attempt_{attempt_index}.json",
                cwd=SCRIPT_DIR,
            )
            attempt_record["compile"] = {"returncode": compile_proc.returncode}

            compile_log_path = COMPILE_LOGS_DIR / f"{child_example_id}.json"
            compile_record = {}
            if compile_log_path.exists():
                compile_record = json.loads(compile_log_path.read_text(encoding="utf-8"))
            attempt_record["compile_record"] = compile_record
            if not bool(compile_record.get("success")):
                status(
                    f"[grouped-job] {grouped_job_id}: attempt {attempt_index} child {child_key} compile failed"
                )
                attempt_record["reason"] = "Compile/repair step did not produce a successful build."
                child_result["attempts"].append(attempt_record)
                child_result["attempt_count"] = len(child_result["attempts"])
                child_result["reason"] = attempt_record["reason"]
                next_pending.append(child_key)
                retry_feedback[child_key] = build_child_retry_feedback(child_task, child_result)
                continue

            status(
                f"[grouped-job] {grouped_job_id}: attempt {attempt_index} child {child_key} feature evaluation start"
            )
            evaluation = evaluate_group_child(
                child_task=child_task,
                baseline_microbenchmarks=baseline_microbenchmarks,
                binary_path=binary_path,
                perfspect_bin=perfspect_bin,
                perf_events=perf_events,
                amd=amd,
                output_dir=run_dir / "feature_eval" / child_key / f"attempt_{attempt_index}",
            )
            attempt_record["feature_evaluation"] = evaluation
            attempt_record["success"] = bool(evaluation.get("success"))
            if not attempt_record["success"]:
                status(
                    f"[grouped-job] {grouped_job_id}: attempt {attempt_index} child {child_key} feature evaluation failed"
                )
                attempt_record["reason"] = evaluation.get("reason") or (
                    "Compiled candidate did not satisfy the requested feature delta."
                )
                child_result["attempts"].append(attempt_record)
                child_result["attempt_count"] = len(child_result["attempts"])
                child_result["reason"] = attempt_record["reason"]
                next_pending.append(child_key)
                retry_feedback[child_key] = build_child_retry_feedback(child_task, child_result)
                continue

            score = float(
                ((evaluation.get("candidate_result") or {}).get("evaluation") or {}).get(
                    "score",
                    float("-inf"),
                )
            )
            attempt_record["score"] = score
            child_result["attempts"].append(attempt_record)
            child_result["attempt_count"] = len(child_result["attempts"])
            child_result["success"] = True
            child_result["reason"] = ""
            child_result["compile_record"] = compile_record
            child_result["feature_evaluation"] = evaluation
            child_result["winning_attempt_index"] = attempt_index
            if child_key in retry_feedback:
                retry_feedback.pop(child_key, None)
            if score > winner_score:
                winner_score = score
                winner_key = child_key
            status(
                f"[grouped-job] {grouped_job_id}: attempt {attempt_index} child {child_key} succeeded with score {score:.4f}"
            )

        pending_child_keys = [
            child_key for child_key in next_pending if not result["children"][child_key].get("success")
        ]
        if pending_child_keys:
            status(
                f"[grouped-job] {grouped_job_id}: attempt {attempt_index} complete, retrying children {', '.join(pending_child_keys)}"
            )
        else:
            status(f"[grouped-job] {grouped_job_id}: attempt {attempt_index} complete")

    if winner_key is not None:
        winner_record = result["children"][winner_key]
        winner_evaluation = winner_record["feature_evaluation"]
        save_group_success(
            group=group,
            winner_key=winner_key,
            winner_record=winner_record,
            winner_evaluation=winner_evaluation,
            retrieval_name=retrieval_name,
            current_file=current_file,
            evaluation_path=evaluation_path,
        )
        result["success"] = True
        result["winner_child_key"] = winner_key
        result["winner_task_id"] = group["child_tasks"][winner_key]["task_id"]
        status(f"[grouped-job] {grouped_job_id}: success via {winner_key}")
    else:
        if pending_child_keys:
            result["reason"] = (
                f"No child candidate satisfied the requested feature delta after {max_child_attempts} attempts."
            )
        else:
            result["reason"] = "No child candidate compiled and satisfied the requested feature delta."
        status(f"[grouped-job] {grouped_job_id}: failed - {result['reason']}")

    write_json(evaluation_path, result)
    return result


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    successful = sum(1 for r in results if r.get("success"))
    compiled = 0
    for result in results:
        if any(
            bool((child.get("compile_record") or {}).get("success"))
            or any(bool((attempt.get("compile_record") or {}).get("success")) for attempt in (child.get("attempts") or []))
            for child in (result.get("children") or {}).values()
        ):
            compiled += 1
    return {
        "total": total,
        "compiled": compiled,
        "successful": successful,
        "failed": total - successful,
    }


def cmd_generate(args: argparse.Namespace) -> int:
    binary_filter = set(args.binary) if args.binary else None
    microbenchmark_filter = set(args.microbenchmark) if args.microbenchmark else None
    feature_filter = set(args.feature_name) if args.feature_name else None
    requested_features = tuple(args.feature_name) if args.feature_name else DEFAULT_TARGET_FEATURES
    feature_df = load_feature_table(args.xlsx)
    source_map = parse_benchmark_binary_sources(args.cmake)
    all_binaries = sorted(set(feature_df["binary"].tolist()))
    covered_binaries = sorted(binary for binary in all_binaries if binary in source_map)
    skipped_binaries = sorted(binary for binary in all_binaries if binary not in source_map)

    tasks, sampling_summary = build_task_manifest(
        xlsx_path=args.xlsx,
        cmake_path=args.cmake,
        requested_features=requested_features,
        binary_filter=binary_filter,
        microbenchmark_filter=microbenchmark_filter,
        feature_filter=feature_filter,
        sample_large_binaries=args.sample_large_binaries,
        small_binary_threshold=args.small_binary_threshold,
        large_binary_sample_cap=args.large_binary_sample_cap,
        large_binary_sample_scale=args.large_binary_sample_scale,
        max_tasks=args.max_tasks,
    )
    append_jsonl(args.output, tasks)

    sample_path = args.output.with_suffix(".sample.json")
    sample_payload = {
        "count": len(tasks),
        "sampling": {
            "sample_large_binaries": args.sample_large_binaries,
            "small_binary_threshold": args.small_binary_threshold,
            "large_binary_sample_cap": args.large_binary_sample_cap,
            "large_binary_sample_scale": args.large_binary_sample_scale,
            "binaries": sampling_summary,
        },
        "covered_binaries": covered_binaries,
        "skipped_binaries": skipped_binaries,
        "sample_tasks": [
            {
                **task,
                "retrieval_query": build_retrieval_query(
                    task_text=task["task"],
                    target_file=task["target_file"],
                    binary_name=task["binary"],
                    microbenchmark_name=task["source_microbenchmark"],
                    anchors=task["style_anchors"],
                    feature_name=task["feature_name"],
                    direction=task["direction"],
                    magnitude=task["magnitude"],
                ),
            }
            for task in tasks[: min(5, len(tasks))]
        ],
    }
    write_json(sample_path, sample_payload)
    print(f"Wrote {len(tasks)} tasks to {args.output}")
    print(f"Wrote sample preview to {sample_path}")
    if skipped_binaries:
        skipped = ", ".join(skipped_binaries)
        print(
            "Skipped binaries that are not mapped by the current Folly CMake-based source resolver: "
            f"{skipped}"
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    rows = load_manifest(args.manifest)
    feature_df = load_feature_table(args.xlsx)
    baseline_by_binary = {
        binary_name: sorted(set(group["microbenchmark"].astype(str).tolist()))
        for binary_name, group in feature_df.groupby("binary", sort=False)
    }
    grouped_jobs = group_manifest_rows(rows)
    task_id_filter = set(args.task_id)
    binary_filter = set(args.binary)
    microbenchmark_filter = set(args.microbenchmark)
    start_index = max(0, int(args.start_index))
    build_dir_map: dict[str, str] = {}
    for mapping in args.build_dir_map:
        if "=" not in mapping:
            raise ValueError(f"Invalid --build-dir-map value: {mapping!r}")
        old_dir, new_dir = mapping.split("=", 1)
        build_dir_map[str(Path(old_dir).resolve())] = str(Path(new_dir).resolve())

    selected: list[dict[str, Any]] = []
    matched_count = 0
    for group in grouped_jobs:
        child_task_ids = {child["task_id"] for child in group["child_tasks"].values()}
        if task_id_filter and group["grouped_job_id"] not in task_id_filter and child_task_ids.isdisjoint(task_id_filter):
            continue
        if binary_filter and group["binary"] not in binary_filter:
            continue
        if microbenchmark_filter and group["source_microbenchmark"] not in microbenchmark_filter:
            continue
        if matched_count < start_index:
            matched_count += 1
            continue

        task = json.loads(json.dumps(group))
        original_build_dir = str(Path(str(task.get("build_dir", str(CMAKE_BUILD_DIR)))).resolve())
        if original_build_dir in build_dir_map:
            task["build_dir"] = build_dir_map[original_build_dir]
        elif args.build_dir_override:
            task["build_dir"] = str(args.build_dir_override)
        for child in task["child_tasks"].values():
            child["build_dir"] = task["build_dir"]
        selected.append(task)
        matched_count += 1
        if args.limit is not None and len(selected) >= args.limit:
            break

    if not selected:
        print("No tasks matched the requested filters.", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    for task in selected:
        status(f"Running grouped job {task['grouped_job_id']}")
        result = run_grouped_task_pipeline(
            group=task,
            baseline_microbenchmarks=baseline_by_binary.get(task["binary"], []),
            top_k=args.top_k,
            perfspect_bin=args.perfspect_bin,
            perf_events=args.perf_events,
            amd=args.amd,
            skip_existing=args.skip_existing,
            max_child_attempts=args.max_child_attempts,
        )
        results.append(result)
        result_status = "success" if result.get("success") else "failed"
        print(f"  -> {result_status}", flush=True)

    summary = summarize_results(results)
    summary_path = PROPERTY_RUNS_DIR / "last_run_summary.json"
    write_json(summary_path, {"summary": summary, "results": results})
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote run summary to {summary_path}", flush=True)
    return 0 if summary["successful"] > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate property-targeted Folly benchmark edit tasks and optionally run the existing retrieval/edit/compile pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate the prompt/task manifest from all_features.xlsx")
    generate.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    generate.add_argument("--cmake", type=Path, default=DEFAULT_CMAKE)
    generate.add_argument("--output", type=Path, default=TASKS_DIR / "property_edit_tasks.jsonl")
    generate.add_argument("--binary", action="append", default=[])
    generate.add_argument("--microbenchmark", action="append", default=[])
    generate.add_argument("--feature-name", action="append", default=[])
    generate.add_argument("--sample-large-binaries", action="store_true")
    generate.add_argument("--small-binary-threshold", type=int, default=DEFAULT_SMALL_BINARY_THRESHOLD)
    generate.add_argument("--large-binary-sample-cap", type=int, default=DEFAULT_LARGE_BINARY_SAMPLE_CAP)
    generate.add_argument("--large-binary-sample-scale", type=float, default=DEFAULT_LARGE_BINARY_SAMPLE_SCALE)
    generate.add_argument("--max-tasks", type=int, default=None)
    generate.set_defaults(func=cmd_generate)

    run = subparsers.add_parser("run", help="Run the retrieval/edit/compile/evaluate pipeline for tasks in a manifest")
    run.add_argument("--manifest", type=Path, default=TASKS_DIR / "property_edit_tasks.jsonl")
    run.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    run.add_argument("--task-id", action="append", default=[])
    run.add_argument("--binary", action="append", default=[])
    run.add_argument("--microbenchmark", action="append", default=[])
    run.add_argument("--start-index", type=int, default=0)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--top-k", type=int, default=8)
    run.add_argument("--build-dir-override", type=Path, default=None)
    run.add_argument("--build-dir-map", action="append", default=[])
    run.add_argument(
        "--perfspect-bin",
        default="perf",
        help="Deprecated compatibility flag; TMA evaluation now uses perf stat topdown metrics.",
    )
    run.add_argument(
        "--perf-events",
        default=(
            "cycles,instructions,L1-icache-load-misses,iTLB-load-misses,"
            "L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,"
            "LLC-load-misses,branch-load-misses,branch-misses,r2424"
        ),
    )
    run.add_argument("--max-child-attempts", type=int, default=DEFAULT_MAX_CHILD_ATTEMPTS)
    run.add_argument("--amd", action="store_true", help="Skip Intel PT extraction during evaluation")
    run.add_argument("--skip-existing", action="store_true")
    run.set_defaults(func=cmd_run)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
