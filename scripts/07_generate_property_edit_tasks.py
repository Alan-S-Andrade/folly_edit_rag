#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
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
from config.settings import (  # noqa: E402
    CMAKE_BUILD_DIR,
    COMPILE_LOGS_DIR,
    FOLLY_SRC_ROOT,
    OUTPUTS_DIR,
    SUCCESSFUL_EDITS_DIR,
)

DEFAULT_XLSX = BENCHMARK_REPLICATION_ROOT / "search_engine" / "all_features.xlsx"
DEFAULT_CMAKE = FOLLY_SRC_ROOT / "CMakeLists.txt"
TASKS_DIR = FOLLY_EDIT_RAG_ROOT / "data" / "property_edit_tasks"
PROPERTY_RUNS_DIR = OUTPUTS_DIR / "property_edit_runs"
TASKS_DIR.mkdir(parents=True, exist_ok=True)
PROPERTY_RUNS_DIR.mkdir(parents=True, exist_ok=True)

ID_COLUMNS = {"binary", "binary_path", "command", "microbenchmark"}
DEFAULT_INCLUDE_PREFIXES = ("perf_stat.", "perfspect.", "intel_pt.")
STATUS_PREFIX = "status."
MAGNITUDE_BANDS = {
    "minimally": (0.10, 0.30),
    "moderately": (0.30, 0.60),
    "substantially": (0.60, 1.00),
    "significantly": (1.00, None),
}
MAGNITUDE_TARGET_FACTORS = {
    "minimally": 0.20,
    "moderately": 0.45,
    "substantially": 0.80,
    "significantly": 1.20,
}


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
    for prefix in ("perf_stat.", "perfspect.", "intel_pt."):
        if feature_name.startswith(prefix):
            return feature_name[len(prefix) :]
    return feature_name


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


def select_numeric_feature_columns(
    df: pd.DataFrame,
    include_prefixes: tuple[str, ...],
) -> list[str]:
    selected: list[str] = []
    for column in df.columns:
        if column in ID_COLUMNS or column.startswith(STATUS_PREFIX):
            continue
        if include_prefixes and not any(column.startswith(prefix) for prefix in include_prefixes):
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            selected.append(column)
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


def load_feature_table(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    required = {"binary", "microbenchmark"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required XLSX columns: {sorted(missing)}")
    return df


def parse_sources_token_line(text: str) -> list[str]:
    out: list[str] = []
    for token in text.split():
        if token.endswith((".cpp", ".cc", ".cxx", ".c")):
            out.append(token)
    return out


def parse_benchmark_binary_sources(cmake_path: Path) -> dict[str, list[Path]]:
    lines = cmake_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    current_dir = Path(".")
    result: dict[str, list[Path]] = {}
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        dir_match = re.match(r"^DIRECTORY\s+(.+?)/?\s*$", stripped)
        if dir_match:
            current_dir = Path(dir_match.group(1).strip())
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
            result[binary_name] = [current_dir / src for src in sources]
        i = j
    return result


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
) -> str:
    display = feature_display_name(feature_name)
    subject = binary_subject(binary_name)
    anchor_clause = ""
    if anchors:
        anchor_clause = " and nearby variants such as " + ", ".join(anchors)

    lines = [
        f"Add a new {subject} variant to {Path(target_file).name} for the {binary_name} binary.",
        f"Keep the coding style and benchmark registration patterns consistent with the existing {microbenchmark_name} benchmark{anchor_clause}.",
        "Do not delete, rename, or weaken existing benchmarks; preserve the existing benchmark harness and main() unless a build fix absolutely requires a local adjustment.",
        f"The new variant should appear as an additional microbenchmark in --bm_list and should {direction_phrase(direction)} {display} {magnitude} relative to {microbenchmark_name}.",
        f"Current baseline for {microbenchmark_name}: {display}={baseline_value:.6g}.",
    ]
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


def task_bounds(scale: float, magnitude: str) -> tuple[float, float | None, float]:
    lower_factor, upper_factor = MAGNITUDE_BANDS[magnitude]
    return (
        lower_factor * scale,
        None if upper_factor is None else upper_factor * scale,
        MAGNITUDE_TARGET_FACTORS[magnitude] * scale,
    )


def build_task_manifest(
    xlsx_path: Path,
    cmake_path: Path,
    include_prefixes: tuple[str, ...],
    binary_filter: set[str] | None,
    microbenchmark_filter: set[str] | None,
    feature_filter: set[str] | None,
    max_tasks: int | None,
) -> list[dict[str, Any]]:
    df = load_feature_table(xlsx_path)
    feature_columns = select_numeric_feature_columns(df, include_prefixes)
    source_map = parse_benchmark_binary_sources(cmake_path)
    text_cache: dict[Path, str] = {}

    tasks: list[dict[str, Any]] = []
    for binary_name, group in df.groupby("binary", sort=True):
        if binary_filter and binary_name not in binary_filter:
            continue
        if binary_name not in source_map:
            continue

        scales = {
            feature_name: compute_feature_scale(group[feature_name])
            for feature_name in feature_columns
        }

        source_rel_paths = source_map[binary_name]
        known_microbenchmarks = sorted({str(v) for v in group["microbenchmark"].tolist()})

        for index, row in group.iterrows():
            microbenchmark_name = str(row["microbenchmark"])
            if microbenchmark_filter and microbenchmark_name not in microbenchmark_filter:
                continue

            target_rel_path = choose_target_file(
                FOLLY_SRC_ROOT,
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
                        lower_bound, upper_bound, target_center = task_bounds(scale, magnitude)
                        task_text = build_task_text(
                            binary_name=binary_name,
                            target_file=str(target_rel_path),
                            microbenchmark_name=microbenchmark_name,
                            anchors=anchors,
                            feature_name=feature_name,
                            direction=direction,
                            magnitude=magnitude,
                            baseline_value=baseline_value,
                        )
                        task_id = slugify(
                            f"{binary_name}_{microbenchmark_name}_{feature_name}_{direction}_{magnitude}"
                        )
                        task = {
                            "task_id": task_id,
                            "task": task_text,
                            "retrieval_query": build_retrieval_query(
                                task_text=task_text,
                                target_file=str(target_rel_path),
                                binary_name=binary_name,
                                microbenchmark_name=microbenchmark_name,
                                anchors=anchors,
                                feature_name=feature_name,
                                direction=direction,
                                magnitude=magnitude,
                            ),
                            "binary": binary_name,
                            "target_file": str(target_rel_path),
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
                            "known_microbenchmarks": known_microbenchmarks,
                        }
                        tasks.append(task)
                        if max_tasks is not None and len(tasks) >= max_tasks:
                            return tasks
    return tasks


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    binary_path: Path,
    perfspect_bin: str,
    perf_events: str,
    amd: bool,
    output_dir: Path,
) -> dict[str, Any]:
    baseline_set = set(task["known_microbenchmarks"])
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


def update_success_metadata(task: dict[str, Any], evaluation: dict[str, Any]) -> None:
    edit_dir = SUCCESSFUL_EDITS_DIR / task["task_id"]
    meta_path = edit_dir / "metadata.json"
    if not meta_path.exists():
        return
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    best = evaluation.get("best_candidate") or {}
    best_eval = best.get("evaluation") or {}
    metadata.update(
        {
            "task_id": task["task_id"],
            "source_microbenchmark": task["source_microbenchmark"],
            "feature_name": task["feature_name"],
            "direction": task["direction"],
            "magnitude": task["magnitude"],
            "baseline_value": task["baseline_value"],
            "performance_success": bool(evaluation.get("success")),
            "new_microbenchmark_name": best.get("microbenchmark"),
            "candidate_value": best_eval.get("candidate_value"),
            "observed_delta": best_eval.get("observed_delta"),
            "evaluation_path": str(PROPERTY_RUNS_DIR / task["task_id"] / "evaluation.json"),
        }
    )
    meta_path.write_text(json.dumps(json_ready(metadata), indent=2), encoding="utf-8")
    write_json(edit_dir / "task.json", task)


def run_task_pipeline(
    task: dict[str, Any],
    top_k: int,
    perfspect_bin: str,
    perf_events: str,
    amd: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    task_id = task["task_id"]
    run_dir = PROPERTY_RUNS_DIR / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    task_json = run_dir / "task.json"
    write_json(task_json, task)

    evaluation_path = run_dir / "evaluation.json"
    if skip_existing and evaluation_path.exists():
        return json.loads(evaluation_path.read_text(encoding="utf-8"))

    retrieval_name = f"{task_id}_retrieval.json"
    rewritten_name = f"{task_id}_candidate_full.cpp"
    current_file = FOLLY_SRC_ROOT / task["target_file"]

    step_results: dict[str, Any] = {
        "task_id": task_id,
        "task": task["task"],
        "target_file": task["target_file"],
        "binary": task["binary"],
    }

    retrieve_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "03_retrieve_edit_context.py"),
        "--task",
        task["task"],
        "--target-file",
        task["target_file"],
        "--binary-name",
        task["binary"],
        "--top-k",
        str(top_k),
        "--output",
        retrieval_name,
    ]
    retrieve_proc = run_cmd_logged(retrieve_cmd, run_dir / "01_retrieve.json", cwd=SCRIPT_DIR)
    step_results["retrieve"] = {"returncode": retrieve_proc.returncode}
    if retrieve_proc.returncode != 0:
        step_results["success"] = False
        step_results["reason"] = "Retrieval step failed."
        write_json(evaluation_path, step_results)
        return step_results

    generate_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "04_generate_full_file.py"),
        "--task",
        task["task"],
        "--target-file",
        task["target_file"],
        "--current-file",
        str(current_file),
        "--retrieval-json",
        retrieval_name,
        "--output",
        rewritten_name,
    ]
    generate_proc = run_cmd_logged(generate_cmd, run_dir / "02_generate.json", cwd=SCRIPT_DIR)
    step_results["generate"] = {"returncode": generate_proc.returncode}
    if generate_proc.returncode != 0:
        step_results["success"] = False
        step_results["reason"] = "Full-file generation step failed."
        write_json(evaluation_path, step_results)
        return step_results

    compile_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "05_rewrite_compile_repair.py"),
        "--task",
        task["task"],
        "--target-file",
        task["target_file"],
        "--binary-name",
        task["binary"],
        "--retrieval-json",
        retrieval_name,
        "--rewritten-file",
        rewritten_name,
        "--example-id",
        task_id,
    ]
    compile_proc = run_cmd_logged(compile_cmd, run_dir / "03_compile.json", cwd=SCRIPT_DIR)
    step_results["compile"] = {"returncode": compile_proc.returncode}

    compile_log_path = COMPILE_LOGS_DIR / f"{task_id}.json"
    compile_record = {}
    if compile_log_path.exists():
        compile_record = json.loads(compile_log_path.read_text(encoding="utf-8"))
    step_results["compile_record"] = compile_record
    compile_success = bool(compile_record.get("success"))
    if not compile_success:
        step_results["success"] = False
        step_results["reason"] = "Compile/repair step did not produce a successful build."
        write_json(evaluation_path, step_results)
        return step_results

    binary_path = (CMAKE_BUILD_DIR / task["binary"]).resolve()
    evaluation = evaluate_task_run(
        task=task,
        binary_path=binary_path,
        perfspect_bin=perfspect_bin,
        perf_events=perf_events,
        amd=amd,
        output_dir=run_dir / "feature_eval",
    )
    step_results["feature_evaluation"] = evaluation
    step_results["success"] = bool(evaluation.get("success"))
    if not step_results["success"]:
        step_results["reason"] = evaluation.get("reason") or "Compiled, but the new benchmark did not satisfy the requested feature delta."
    update_success_metadata(task, evaluation)
    write_json(evaluation_path, step_results)
    return step_results


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    successful = sum(1 for r in results if r.get("success"))
    compiled = sum(1 for r in results if (r.get("compile_record") or {}).get("success"))
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
    include_prefixes = tuple(args.include_prefix)
    feature_df = load_feature_table(args.xlsx)
    source_map = parse_benchmark_binary_sources(args.cmake)
    all_binaries = sorted(set(feature_df["binary"].tolist()))
    covered_binaries = sorted(binary for binary in all_binaries if binary in source_map)
    skipped_binaries = sorted(binary for binary in all_binaries if binary not in source_map)

    tasks = build_task_manifest(
        xlsx_path=args.xlsx,
        cmake_path=args.cmake,
        include_prefixes=include_prefixes,
        binary_filter=binary_filter,
        microbenchmark_filter=microbenchmark_filter,
        feature_filter=feature_filter,
        max_tasks=args.max_tasks,
    )
    append_jsonl(args.output, tasks)

    sample_path = args.output.with_suffix(".sample.json")
    sample_payload = {
        "count": len(tasks),
        "covered_binaries": covered_binaries,
        "skipped_binaries": skipped_binaries,
        "sample_tasks": tasks[: min(5, len(tasks))],
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
    selected: list[dict[str, Any]] = []
    for row in rows:
        if args.task_id and row["task_id"] not in set(args.task_id):
            continue
        if args.binary and row["binary"] not in set(args.binary):
            continue
        if args.microbenchmark and row["source_microbenchmark"] not in set(args.microbenchmark):
            continue
        selected.append(row)
        if args.limit is not None and len(selected) >= args.limit:
            break

    if not selected:
        print("No tasks matched the requested filters.", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    for task in selected:
        print(f"Running task {task['task_id']}")
        result = run_task_pipeline(
            task=task,
            top_k=args.top_k,
            perfspect_bin=args.perfspect_bin,
            perf_events=args.perf_events,
            amd=args.amd,
            skip_existing=args.skip_existing,
        )
        results.append(result)
        status = "success" if result.get("success") else "failed"
        print(f"  -> {status}")

    summary = summarize_results(results)
    summary_path = PROPERTY_RUNS_DIR / "last_run_summary.json"
    write_json(summary_path, {"summary": summary, "results": results})
    print(json.dumps(summary, indent=2))
    print(f"Wrote run summary to {summary_path}")
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
    generate.add_argument(
        "--include-prefix",
        action="append",
        default=list(DEFAULT_INCLUDE_PREFIXES),
        help="Feature prefix to include when generating tasks. Repeat to include multiple prefixes.",
    )
    generate.add_argument("--binary", action="append", default=[])
    generate.add_argument("--microbenchmark", action="append", default=[])
    generate.add_argument("--feature-name", action="append", default=[])
    generate.add_argument("--max-tasks", type=int, default=None)
    generate.set_defaults(func=cmd_generate)

    run = subparsers.add_parser("run", help="Run the retrieval/edit/compile/evaluate pipeline for tasks in a manifest")
    run.add_argument("--manifest", type=Path, default=TASKS_DIR / "property_edit_tasks.jsonl")
    run.add_argument("--task-id", action="append", default=[])
    run.add_argument("--binary", action="append", default=[])
    run.add_argument("--microbenchmark", action="append", default=[])
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--top-k", type=int, default=8)
    run.add_argument("--perfspect-bin", default="perfspect")
    run.add_argument(
        "--perf-events",
        default=(
            "cycles,instructions,L1-icache-load-misses,iTLB-load-misses,"
            "L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,"
            "LLC-load-misses,branch-load-misses,branch-misses,r2424"
        ),
    )
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
