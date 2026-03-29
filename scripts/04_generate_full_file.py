#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from config.settings import (
    GENERATION_MODEL_NAME,
    SUCCESSFUL_EDITS_DIR,
    REPAIRED_DIR,
    RETRIEVALS_DIR,
)
from llm_provider import create_text_model
from local_patch_utils import (
    build_edit_location_guidance,
    build_registration_idiom_guidance,
    normalize_generated_file,
    prepare_prompt_file_view,
    restore_prompt_file_omissions,
    validate_generated_source_output,
)

FULLFILE_PROMPT = """You are editing an existing Folly benchmark source file in place.

Edit target:
{target_file}

Output rules:
- Return only the complete updated contents of {target_file}.
- The first non-whitespace characters of your response must already be source code from {target_file}, not an English sentence.
- Do not return a diff.
- Do not return explanations, headings, benchmark output, or code fences.
- Keep the edit local and minimal.
- Keep the result compilable in the existing Folly/DCPerf build.

Hard constraints:
- Preserve benchmark harness structure, includes, namespace usage, and main() unless the task explicitly requires a change.
- Use only APIs already present in the current file or clearly implied by its existing headers.
- Keep unrelated code unchanged where possible.
- Add exactly one new benchmark registration and do not introduce any other new benchmark names.
- In derived_from_nearest mode, keep the source benchmark intact and place the new variant adjacent to the source benchmark's existing registration or the explicit insertion site described in the task.
- In from_scratch mode, keep the reference benchmark intact, add one new from-scratch benchmark registration at the placement site described in the task, and do not clone the reference benchmark body.
- Do not invent new build targets or unrelated helper utilities.
- If the current file contents contain an omitted disabled-appendix placeholder, keep that placeholder unchanged in the returned file.

Generation mode:
{iteration_mode_guidance}

Internal reasoning rule:
- Diagnose the mismatch, infer the subsystem cause, and choose one minimal fix.
- Treat counter gaps as symptoms; do not optimize numbers directly.
- Keep helpful mechanisms already present unless the task explicitly says otherwise.
- Do not include this reasoning in the output.

Task:
{task}

Control state:
- Attempt: {attempt_index} of {max_attempts}
- Initial generation mode: {initial_generation_mode}
{diagnosis_block}{feature_guidance_block}{refinement_feedback_block}{successful_examples_block}{retrieved_block}

Current target file contents:
{current_file}

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
- If the current file contents contain an omitted disabled-appendix placeholder, keep that placeholder unchanged in every returned file block.
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

Required internal reasoning protocol before writing code:
- Diagnose the highest-priority mismatch for each child target before choosing an edit.
- Treat counter gaps as symptoms; target the underlying subsystem cause with one minimal local change.
- Preserve metrics already near target and do not rewrite from scratch.
- Do not include this reasoning in the output; return only the required file blocks.

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


def _trim_prompt_section(text: str, *, max_lines: int = 12, max_chars: int = 1400) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    lines = cleaned.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    clipped = "\n".join(lines).strip()
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars].rstrip() + "..."
    return clipped


def _format_optional_block(label: str, text: str, *, max_lines: int, max_chars: int) -> str:
    clipped = _trim_prompt_section(text, max_lines=max_lines, max_chars=max_chars)
    if not clipped:
        return ""
    indented = "\n".join(f"  {line}" for line in clipped.splitlines())
    return f"- {label}:\n{indented}\n"


def _is_empty_retrieval_notice(text: str) -> bool:
    cleaned = text.strip()
    return cleaned.startswith("No retrieved context is available for this attempt.")


def load_retrieved_context(retrieval_json: str) -> str:
    retrieved_contexts = json.loads(load_text(RETRIEVALS_DIR / retrieval_json))
    if not retrieved_contexts:
        return (
            "No retrieved context is available for this attempt. "
            "Use the current target file as the authoritative style, API, and registration reference."
        )
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


INVALID_GENERATION_PROMPT = """Your previous response was invalid because: {reason}

Return only the complete updated source file contents for {target_file}.
Do not include prose, headings, benchmark output, context labels, or code fences.
Start immediately with source code from {target_file}; do not preface the file with any explanation.

Task:
{task}

Local registration idiom to preserve:
{registration_guidance}

Current target file contents:
{current_file}
"""


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


def _is_frontend_hardware_feature(feature_name: str) -> bool:
    lowered = feature_name.lower()
    return (
        "icache" in lowered
        or "itlb" in lowered
        or "frontend" in lowered
    )


def _build_feature_guidance(feature_name: str, direction: str = "") -> str:
    guidance = [
        "- Prefer small, local helper or benchmark-body changes over broad file rewrites.",
        "- Keep semantics close to the source microbenchmark and avoid unrelated scaffolding.",
        "- Preserve existing registration patterns and benchmark naming conventions except for the required deterministic suffix.",
    ]
    if _is_frontend_hardware_feature(feature_name):
        guidance.extend(
            [
                "- For frontend hardware counters, prefer compact noinline helper fan-out, helper call chains, branch-shaping loops, and folly::doNotOptimizeAway to keep the added work on the hot executed path.",
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


def _successful_example_rank(metadata: dict, target_file: str, binary_name: str) -> tuple[int, int, int, str]:
    metadata_target = str(metadata.get("target_file", "")).strip()
    metadata_binary = str(metadata.get("binary_name", "")).strip()
    metadata_parent = str(Path(metadata_target).parent) if metadata_target else ""
    target_parent = str(Path(target_file).parent)
    same_file = metadata_target == target_file
    same_binary = bool(binary_name) and metadata_binary == binary_name
    same_parent = metadata_parent == target_parent
    return (
        0 if same_file else 1,
        0 if same_binary else 1,
        0 if same_parent else 1,
        metadata_target,
    )


def _successful_example_match_label(metadata: dict, target_file: str, binary_name: str) -> str:
    metadata_target = str(metadata.get("target_file", "")).strip()
    metadata_binary = str(metadata.get("binary_name", "")).strip()
    if metadata_target == target_file:
        return "same_target_file"
    if binary_name and metadata_binary == binary_name:
        return "same_binary"
    if metadata_target and Path(metadata_target).parent == Path(target_file).parent:
        return "same_directory"
    return "related"


def _load_successful_examples(target_file: str, binary_name: str = "", limit: int = 2) -> str:
    examples: list[tuple[tuple[int, int, int, str], str, str, str]] = []
    for metadata_path in sorted(SUCCESSFUL_EDITS_DIR.glob("*/metadata.json")):
        try:
            metadata = json.loads(load_text(metadata_path))
        except Exception:
            continue
        if not bool(metadata.get("compile_success", True)):
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
        rank = _successful_example_rank(metadata, target_file, binary_name)
        match_label = _successful_example_match_label(metadata, target_file, binary_name)
        examples.append(
            (
                rank,
                match_label,
                str(task.get("task", "")).strip(),
                _trim_example_text(load_text(final_path)),
            )
        )

    if not examples:
        return "- No prior successful examples were found for this target file."

    examples.sort(key=lambda row: row[0])
    rendered: list[str] = []
    for idx, (_, match_label, task_text, final_text) in enumerate(examples[:limit], start=1):
        rendered.append(
            "\n".join(
                [
                    f"[SUCCESS EXAMPLE {idx} | {match_label}]",
                    f"Task: {task_text}",
                    "Successful file excerpt:",
                    final_text,
                ]
            )
        )
    return "\n\n".join(rendered)


def generate_single(model, args: argparse.Namespace) -> None:
    retrieved = load_retrieved_context(args.retrieval_json)
    current_file = load_text(Path(args.current_file))
    prompt_file_view, omitted_blocks = prepare_prompt_file_view(current_file)
    retry_feedback = ""
    if args.retry_feedback_file:
        retry_path = Path(args.retry_feedback_file)
        if retry_path.exists():
            retry_feedback = load_text(retry_path).strip()
    if not retry_feedback:
        retry_feedback = (
            "- No previous attempt feedback. This is the initial directional patch: create one minimal variant close to the existing benchmark family, "
            "use bounded deltas, and do not try to solve every counter in one shot."
        )
    diagnosis = ""
    if args.diagnosis_file:
        diagnosis_path = Path(args.diagnosis_file)
        if diagnosis_path.exists():
            diagnosis = load_text(diagnosis_path).strip()
    successful_examples = _load_successful_examples(str(args.target_file), str(args.binary_name))
    corrective_iteration = bool(args.retry_feedback_file and Path(args.retry_feedback_file).exists())
    edit_location_guidance = build_edit_location_guidance(
        current_file,
        args.source_microbenchmark or args.task,
    )
    registration_guidance = build_registration_idiom_guidance(
        current_file,
        args.source_microbenchmark or args.task,
    )
    compact_feature_guidance = (
        "\n".join(
            line
            for line in [
                args.feature_guidance.strip(),
                f"Edit placement guidance: {edit_location_guidance}" if edit_location_guidance else "",
                f"Registration idiom guidance: {registration_guidance}" if registration_guidance else "",
            ]
            if line
        )
        or "No additional feature guidance was provided."
    )
    feature_guidance_block = _format_optional_block(
        "Feature-specific edit guidance",
        compact_feature_guidance,
        max_lines=12,
        max_chars=1400,
    )
    diagnosis_block = _format_optional_block(
        "Diagnostician playbook",
        diagnosis,
        max_lines=12,
        max_chars=1400,
    )
    refinement_feedback_block = _format_optional_block(
        "Refinement feedback",
        retry_feedback,
        max_lines=10,
        max_chars=1200,
    )
    successful_examples_block = _format_optional_block(
        "Prior successful examples for the same target file",
        successful_examples,
        max_lines=20,
        max_chars=2200,
    )
    retrieved_block = ""
    if retrieved and not _is_empty_retrieval_notice(retrieved):
        retrieved_block = _format_optional_block(
            "Retrieved context",
            retrieved,
            max_lines=24,
            max_chars=2400,
        )

    prompt = FULLFILE_PROMPT.format(
        task=args.task.strip(),
        target_file=args.target_file,
        attempt_index=args.attempt_index,
        max_attempts=args.max_attempts,
        initial_generation_mode=args.initial_generation_mode.strip() or "derived_from_nearest",
        diagnosis_block=diagnosis_block,
        iteration_mode_guidance=(
            "This is a corrective iteration over an existing generated variant. "
            "Do not rewrite from scratch. Make one minimal corrective patch only. "
            "Preserve existing helpful mechanisms. Prioritize: 1. buildability 2. IPC 3. primary target counter 4. secondary counters. "
            "Change one primary lever at a time."
            if corrective_iteration
            else (
                "This is the initial directional step. Start with one compact from-scratch benchmark addition that uses the local file's style and registration patterns without cloning the reference benchmark body. "
                "Use bounded deltas and preserve benchmark character instead of trying to solve every counter in one shot."
                if args.initial_generation_mode.strip() == "from_scratch"
                else "This is the initial directional step. Start with one minimal local variant close to the source benchmark family. "
                "Use bounded deltas and preserve IPC and benchmark character instead of trying to solve every counter in one shot."
            )
        ),
        feature_guidance_block=feature_guidance_block,
        refinement_feedback_block=refinement_feedback_block,
        successful_examples_block=successful_examples_block,
        retrieved_block=retrieved_block,
        current_file=prompt_file_view,
    )
    out_path = REPAIRED_DIR / args.output
    current_prompt = prompt
    generated_file = prompt_file_view
    for _ in range(2):
        print("[generate] prompt-begin", flush=True)
        print(current_prompt, flush=True)
        print("[generate] prompt-end", flush=True)
        print(f"[generate] starting single-file rewrite for {args.target_file}", flush=True)
        print(
            f"[generate] waiting on Gemini inference for attempt {args.attempt_index}/{args.max_attempts}",
            flush=True,
        )
        started_at = time.monotonic()
        resp = model.generate_content(current_prompt)
        elapsed_s = time.monotonic() - started_at
        print(f"[generate] Gemini inference returned after {elapsed_s:.1f}s", flush=True)
        generated_file = normalize_generated_file(resp.text)
        generated_file = restore_prompt_file_omissions(generated_file, omitted_blocks)
        reason = validate_generated_source_output(
            generated_file,
            original_text=prompt_file_view,
        )
        if reason is None:
            break
        print(f"[generate] rejected model response: {reason}", flush=True)
        current_prompt = INVALID_GENERATION_PROMPT.format(
            reason=reason,
            target_file=args.target_file,
            task=args.task.strip(),
            registration_guidance=registration_guidance
            or "Preserve the existing local registration idiom exactly.",
            current_file=prompt_file_view,
        )
    out_path.write_text(generated_file, encoding="utf-8")
    print(f"[generate] wrote rewritten source to {out_path}", flush=True)


def generate_grouped(model, args: argparse.Namespace) -> None:
    grouped_task = json.loads(load_text(Path(args.grouped_task_json)))
    child_tasks = grouped_task["child_tasks"]
    child_keys = list(child_tasks.keys())
    retrieved = load_retrieved_context(args.retrieval_json)
    current_file = load_text(Path(args.current_file))
    prompt_file_view, omitted_blocks = prepare_prompt_file_view(current_file)
    successful_examples = _load_successful_examples(
        str(grouped_task["target_file"]),
        str(grouped_task.get("binary", "")),
    )

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
        current_file=prompt_file_view,
        child_specs="\n\n".join(child_specs),
        retrieved=retrieved,
    )
    print("[generate] grouped-prompt-begin", flush=True)
    print(prompt, flush=True)
    print("[generate] grouped-prompt-end", flush=True)
    print(
        f"[generate] grouped job {grouped_task['grouped_job_id']} attempt "
        f"{grouped_task.get('attempt_index', 1)}/{grouped_task.get('max_child_attempts', 1)} "
        f"for {len(child_keys)} child targets",
        flush=True,
    )
    print("[generate] waiting on Gemini grouped inference", flush=True)
    started_at = time.monotonic()
    resp = model.generate_content(prompt)
    elapsed_s = time.monotonic() - started_at
    print(f"[generate] Gemini grouped inference returned after {elapsed_s:.1f}s", flush=True)

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
        file_text = restore_prompt_file_omissions(file_text, omitted_blocks)
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
    parser.add_argument("--binary-name", default="")
    parser.add_argument("--current-file", required=True)
    parser.add_argument("--retrieval-json", default="sample_retrieval.json")
    parser.add_argument("--output", default="candidate_full.cpp")
    parser.add_argument("--attempt-index", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--feature-guidance", default="")
    parser.add_argument("--diagnosis-file", default="")
    parser.add_argument("--retry-feedback-file", default="")
    parser.add_argument("--initial-generation-mode", default="derived_from_nearest")
    parser.add_argument("--grouped-task-json", default="")
    parser.add_argument("--output-manifest", default="grouped_candidates.json")
    parser.add_argument("--example-id", default="latest_group")
    args = parser.parse_args()

    model = create_text_model(GENERATION_MODEL_NAME)

    if args.grouped_task_json:
        generate_grouped(model, args)
    else:
        generate_single(model, args)


if __name__ == "__main__":
    main()
