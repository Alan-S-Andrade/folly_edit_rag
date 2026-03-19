#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import vertexai
from vertexai.generative_models import GenerativeModel

from config.settings import (
    LOCATION,
    MODEL_NAME,
    PROJECT_ID,
    SUCCESSFUL_EDITS_DIR,
    REPAIRED_DIR,
    RETRIEVALS_DIR,
)

FULLFILE_PROMPT = """You are editing an existing Folly benchmark source file in place.

Rules:
- Return only the complete updated source file contents for the target file.
- Preserve benchmark harness structure, includes, namespace usage, and main() unless the task explicitly requires a change.
- Use only APIs present in the retrieved context or already present in the target file.
- Keep unrelated code unchanged where possible.
- Do not invent new build targets or unrelated helper utilities.
- Keep the result compilable in the existing Folly/DCPerf build.

Task:
{task}

Target file:
{target_file}

Current target file contents:
{current_file}

Retrieved context:
{retrieved}

Return only the complete updated source file contents for {target_file}.
"""

GROUPED_PROMPT = """You are editing an existing Folly benchmark source file in place.

Rules:
- You must produce {child_count} separate complete source-file rewrites in a single response.
- Each rewrite must add exactly one new benchmark variant for its own child target and leave the rest of the file unchanged except for minimal support edits.
- Preserve includes, namespace usage, benchmark harness structure, and main() unless absolutely required by the child target.
- Use only APIs present in the retrieved context or already present in the target file.
- Do not delete, rename, or weaken existing benchmarks.
- Do not invent new build targets or unrelated helper utilities.
- Each new benchmark name must include the exact suffix required for that child target.
- Return the response using exactly the marker format shown below, with no extra commentary.

Required response format:
<<<BEGIN {first_child_key}>>>
<complete file for {first_child_key}>
<<<END {first_child_key}>>>
... repeat for every child key ...

Target file:
{target_file}

Source microbenchmark:
{source_microbenchmark}

Target feature:
{feature_name}

Attempt:
{attempt_index} of {max_child_attempts}

Feature-specific edit guidance:
{feature_guidance}

Refinement feedback:
{refinement_feedback}

Prior successful examples for the same target file:
{successful_examples}

Current target file contents:
{current_file}

Child targets:
{child_specs}

Retrieved context:
{retrieved}
"""


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_retrieved_context(retrieval_json: str) -> str:
    retrieved_contexts = json.loads(load_text(RETRIEVALS_DIR / retrieval_json))
    return "\n\n".join(f"[CONTEXT {i + 1}]\n{c['text']}" for i, c in enumerate(retrieved_contexts))


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _parse_grouped_response(text: str, child_keys: list[str]) -> dict[str, str]:
    stripped = _strip_code_fences(text)
    candidates: dict[str, str] = {}
    for child_key in child_keys:
        pattern = re.compile(
            rf"<<<BEGIN {re.escape(child_key)}>>>\s*(.*?)\s*<<<END {re.escape(child_key)}>>>",
            flags=re.DOTALL,
        )
        match = pattern.search(stripped)
        if not match:
            raise ValueError(f"Missing response block for child target {child_key}")
        candidates[child_key] = match.group(1).rstrip() + "\n"
    return candidates


def _is_frontend_or_pt_feature(feature_name: str) -> bool:
    lowered = feature_name.lower()
    return (
        lowered.startswith("intel_pt.")
        or "icache" in lowered
        or "itlb" in lowered
        or "frontend" in lowered
    )


def _build_feature_guidance(feature_name: str, direction: str = "") -> str:
    guidance = [
        "- Prefer small, local helper or benchmark-body changes over broad file rewrites.",
        "- Keep semantics close to the source microbenchmark and avoid unrelated scaffolding.",
        "- Preserve existing registration patterns and benchmark naming conventions except for the required deterministic suffix.",
    ]
    if _is_frontend_or_pt_feature(feature_name):
        guidance.extend(
            [
                "- For frontend and Intel PT features, prefer noinline helper fan-out, helper call chains, branch-shaping loops, and folly::doNotOptimizeAway to keep added work visible in the trace.",
                '- Only if simpler C++ and Folly edits are insufficient, you may use a tiny targeted inline-asm fence in a helper, such as asm volatile("" ::: "memory") or asm volatile("" : "+r"(x) :: "memory"), to prevent over-optimization.',
                "- Keep any asm local, minimal, and tied only to frontend or Intel PT shaping. Do not use asm as a general editing primitive.",
            ]
        )
        lowered = feature_name.lower()
        if "l1-icache-load-misses" in lowered and direction == "increase":
            guidance.extend(
                [
                    "- For increasing L1 icache MPKI, grow the hot executed instruction footprint with many distinct noinline helper functions, each with a substantial straight-line body.",
                    "- Increase helper variant count and reduce helper packing with alignment or separation when feasible, so the active code is less likely to fit comfortably in L1I.",
                ]
            )
    else:
        guidance.append(
            "- Do not use inline asm for this feature; prefer normal C++ and Folly edits, helper calls, loop structure, branch structure, and benchmark parameters."
        )
    return "\n".join(guidance)


def _build_refinement_feedback(grouped_task: dict[str, object]) -> str:
    feedback = grouped_task.get("retry_feedback")
    if not isinstance(feedback, dict) or not feedback:
        return "- No previous attempt feedback. Produce the best first-pass candidate for each child target."

    lines: list[str] = []
    for child_key, child_feedback in feedback.items():
        if not child_feedback:
            continue
        lines.append(f"- {child_key}: {str(child_feedback).strip()}")
    if not lines:
        return "- No previous attempt feedback. Produce the best first-pass candidate for each child target."
    return "\n".join(lines)


def _trim_example_text(text: str, max_chars: int = 2200) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "\n..."


def _load_successful_examples(target_file: str, limit: int = 2) -> str:
    examples: list[tuple[str, str]] = []
    for metadata_path in sorted(SUCCESSFUL_EDITS_DIR.glob("*/metadata.json")):
        try:
            metadata = json.loads(load_text(metadata_path))
        except Exception:
            continue
        if str(metadata.get("target_file")) != target_file:
            continue
        edit_dir = metadata_path.parent
        task_path = edit_dir / "task.json"
        final_path = edit_dir / "final.cpp"
        if not task_path.exists() or not final_path.exists():
            continue
        try:
            task = json.loads(load_text(task_path))
        except Exception:
            continue
        examples.append((str(task.get("task", "")).strip(), _trim_example_text(load_text(final_path))))

    if not examples:
        return "- No prior successful examples were found for this target file."

    rendered: list[str] = []
    for idx, (task_text, final_text) in enumerate(examples[:limit], start=1):
        rendered.append(
            "\n".join(
                [
                    f"[SUCCESS EXAMPLE {idx}]",
                    f"Task: {task_text}",
                    "Successful file excerpt:",
                    final_text,
                ]
            )
        )
    return "\n\n".join(rendered)


def generate_single(model: GenerativeModel, args: argparse.Namespace) -> None:
    retrieved = load_retrieved_context(args.retrieval_json)
    current_file = load_text(Path(args.current_file))

    prompt = FULLFILE_PROMPT.format(
        task=args.task.strip(),
        target_file=args.target_file,
        current_file=current_file,
        retrieved=retrieved,
    )
    print(f"[generate] starting single-file rewrite for {args.target_file}", flush=True)
    resp = model.generate_content(prompt)
    out_path = REPAIRED_DIR / args.output
    out_path.write_text(resp.text, encoding="utf-8")
    print(f"[generate] wrote rewritten source to {out_path}", flush=True)


def generate_grouped(model: GenerativeModel, args: argparse.Namespace) -> None:
    grouped_task = json.loads(load_text(Path(args.grouped_task_json)))
    child_tasks = grouped_task["child_tasks"]
    child_keys = list(child_tasks.keys())
    retrieved = load_retrieved_context(args.retrieval_json)
    current_file = load_text(Path(args.current_file))
    successful_examples = _load_successful_examples(str(grouped_task["target_file"]))

    child_specs = []
    for child_key in child_keys:
        child = child_tasks[child_key]
        child_specs.append(
            "\n".join(
                [
                    f"- Child key: {child_key}",
                    f"  Intent: {child['task']}",
                    f"  Direction: {child['direction']}",
                    f"  Magnitude: {child['magnitude']}",
                    f"  Required benchmark-name suffix: {child['expected_suffix']}",
                ]
            )
        )

    prompt = GROUPED_PROMPT.format(
        child_count=len(child_keys),
        first_child_key=child_keys[0],
        target_file=grouped_task["target_file"],
        source_microbenchmark=grouped_task["source_microbenchmark"],
        feature_name=grouped_task["feature_name"],
        attempt_index=grouped_task.get("attempt_index", 1),
        max_child_attempts=grouped_task.get("max_child_attempts", 1),
        feature_guidance=_build_feature_guidance(
            str(grouped_task["feature_name"]),
            str(grouped_task.get("direction", "")),
        ),
        refinement_feedback=_build_refinement_feedback(grouped_task),
        successful_examples=successful_examples,
        current_file=current_file,
        child_specs="\n\n".join(child_specs),
        retrieved=retrieved,
    )
    print(
        f"[generate] grouped job {grouped_task['grouped_job_id']} attempt "
        f"{grouped_task.get('attempt_index', 1)}/{grouped_task.get('max_child_attempts', 1)} "
        f"for {len(child_keys)} child targets",
        flush=True,
    )
    resp = model.generate_content(prompt)

    example_dir = REPAIRED_DIR / args.example_id
    example_dir.mkdir(parents=True, exist_ok=True)
    raw_response_path = example_dir / "grouped_raw_response.txt"
    raw_response_path.write_text(resp.text, encoding="utf-8")

    candidates = _parse_grouped_response(resp.text, child_keys)
    manifest: dict[str, object] = {
        "grouped_job_id": grouped_task["grouped_job_id"],
        "target_file": grouped_task["target_file"],
        "feature_name": grouped_task["feature_name"],
        "raw_response": str(raw_response_path),
        "candidates": {},
    }
    for child_key, file_text in candidates.items():
        candidate_path = example_dir / f"{child_key}_candidate_full.cpp"
        candidate_path.write_text(file_text, encoding="utf-8")
        manifest["candidates"][child_key] = {
            "candidate_file": str(candidate_path),
            "expected_suffix": child_tasks[child_key]["expected_suffix"],
            "direction": child_tasks[child_key]["direction"],
            "magnitude": child_tasks[child_key]["magnitude"],
            "task_id": child_tasks[child_key]["task_id"],
        }

    out_path = REPAIRED_DIR / args.output_manifest
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[generate] wrote grouped candidate manifest to {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="")
    parser.add_argument("--target-file", required=True)
    parser.add_argument("--current-file", required=True)
    parser.add_argument("--retrieval-json", default="sample_retrieval.json")
    parser.add_argument("--output", default="candidate_full.cpp")
    parser.add_argument("--grouped-task-json", default="")
    parser.add_argument("--output-manifest", default="grouped_candidates.json")
    parser.add_argument("--example-id", default="latest_group")
    args = parser.parse_args()

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)

    if args.grouped_task_json:
        generate_grouped(model, args)
    else:
        generate_single(model, args)


if __name__ == "__main__":
    main()
