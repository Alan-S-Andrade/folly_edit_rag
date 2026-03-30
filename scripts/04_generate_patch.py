#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config.settings import (
    GENERATION_MODEL_NAME,
    PATCHES_DIR,
    PATCH_FILENAME,
    RETRIEVALS_DIR,
)
from llm_provider import create_text_model
from local_patch_utils import (
    build_unified_diff,
    build_edit_location_guidance,
    build_registration_idiom_guidance,
    normalize_generated_file,
    prepare_prompt_file_view,
    render_retrieved_context,
    repair_truncated_tail,
    restore_prompt_file_omissions,
    validate_generated_source_output,
)

PATCH_PROMPT = """You are editing an existing Folly benchmark source file in place.

Task:
{task}

Edit target:
{target_file}

Hard constraints:
- Return only the complete updated contents of {target_file}.
- The first non-whitespace characters of your response must already be source code from {target_file}, not an English sentence.
- Keep the edit local, minimal, and compilable in the existing Folly/DCPerf build.
{benchmark_mode_constraints}
{benchmark_name_constraints}
- Do not modify unrelated benchmark families, harness code, includes, namespace usage, or main() unless a small local build fix is required.
- Use only APIs already present in the current file or clearly implied by existing headers.
- Do not invent new build targets or unrelated helper utilities.
- If the current file contents contain an omitted disabled-appendix placeholder, keep that placeholder unchanged in the returned file.

Mode:
{iteration_mode_guidance}

Control state:
- Reference benchmark: {reference_microbenchmark}
- Current working benchmark anchor: {source_microbenchmark}
- Attempt: {attempt_index} of {max_attempts}
- Initial generation mode: {initial_generation_mode}
{working_state_block}{diagnosis_block}{feature_guidance_block}{refinement_feedback_block}{retrieved_block}

Current full target file contents:
{current_file}
"""


INVALID_GENERATION_PROMPT = """Your previous response was invalid because: {reason}

Return only the complete updated source file contents for {target_file}.
Do not include prose, headings, benchmark output, context labels, or code fences.
Start immediately with source code from {target_file}; do not preface the file with any explanation.
Keep the edit local, minimal, and compilable.

Task:
{task}

Local registration idiom to preserve:
{registration_guidance}

Current target file contents:
{current_file}
"""


def load_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore')


def _trim_prompt_section(text: str, *, max_lines: int = 10, max_chars: int = 1200) -> str:
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


def _normalize_task_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("Task:"):
        cleaned = cleaned[len("Task:"):].lstrip()
    return cleaned


def _is_carried_forward_iteration(
    *,
    corrective_iteration: bool,
    working_state: str,
    reference_microbenchmark: str,
    source_microbenchmark: str,
) -> bool:
    if not corrective_iteration:
        return False
    working_state_lc = working_state.lower()
    if "carried-forward" in working_state_lc:
        return True
    return (
        bool(reference_microbenchmark.strip())
        and bool(source_microbenchmark.strip())
        and reference_microbenchmark.strip() != source_microbenchmark.strip()
    )


def _build_prompt_task(
    *,
    corrective_iteration: bool,
    carried_forward_iteration: bool,
    original_task: str,
    target_file: str,
    reference_microbenchmark: str,
    source_microbenchmark: str,
) -> str:
    reference_name = reference_microbenchmark.strip() or "(not provided)"
    source_name = source_microbenchmark.strip() or reference_name
    if corrective_iteration:
        if carried_forward_iteration:
            return (
                f"Refine the existing carried-forward benchmark `{source_name}` in place in {target_file}.\n"
                f"Treat `{source_name}` as the only active working benchmark anchor for this attempt.\n"
                f"Do not add a new benchmark registration, do not rewrite from scratch, and do not return to the "
                f"original reference benchmark `{reference_name}`.\n"
                "Make one minimal corrective patch to the current benchmark body or its nearby local state, and keep "
                "all unrelated benchmark families and harness code unchanged."
            )
        return (
            f"Correct the current attempt for benchmark family `{source_name}` in {target_file}.\n"
            "This is an iterative correction pass, not an initial generation pass.\n"
            f"If the current file still only contains the original reference benchmark `{reference_name}`, you may add "
            "exactly one derived benchmark. Otherwise refine the existing generated benchmark in place.\n"
            "Do not rewrite from scratch, do not add multiple generated benchmarks, and make one minimal corrective "
            "change guided by the latest measured feedback."
        )
    return _normalize_task_text(original_task)


def _build_benchmark_mode_constraints(*, carried_forward_iteration: bool) -> str:
    if carried_forward_iteration:
        return (
            "- Refine the existing carried-forward generated benchmark in place.\n"
            "- Do not add a second generated benchmark registration.\n"
            "- Do not revert to editing the original reference benchmark body."
        )
    return (
        "- Keep the original reference benchmark unchanged unless the task explicitly says to refine a "
        "carried-forward generated benchmark in place.\n"
        "- Add exactly one new benchmark registration when working from the original file."
    )


def _build_benchmark_name_constraints(
    *,
    carried_forward_iteration: bool,
    reference_microbenchmark: str,
    source_microbenchmark: str,
) -> str:
    names: list[str] = []
    for candidate in (reference_microbenchmark, source_microbenchmark):
        cleaned = " ".join(str(candidate).split()).strip()
        if cleaned and cleaned not in names:
            names.append(cleaned)
    if carried_forward_iteration:
        return (
            "- Keep the existing carried-forward benchmark name if you refine it in place.\n"
            "- Do not add a new registration that reuses any existing `--bm_list` benchmark name."
        )
    if names:
        rendered = ", ".join(f"`{name}`" for name in names)
        return (
            "- The added benchmark name must be unique in `--bm_list`.\n"
            f"- Do not reuse an existing benchmark name such as {rendered}."
        )
    return "- The added benchmark name must be unique in `--bm_list` and must not reuse an existing benchmark name."


def _load_forbidden_benchmark_names(path_str: str) -> set[str]:
    if not path_str.strip():
        return set()
    path = Path(path_str)
    if not path.exists():
        return set()
    try:
        payload = json.loads(load_text(path))
    except Exception:
        return set()
    names = payload.get("benchmark_names", []) if isinstance(payload, dict) else payload
    if not isinstance(names, list):
        return set()
    return {" ".join(str(name).split()).strip() for name in names if str(name).strip()}


def _generate_validated_file(
    model,
    *,
    prompt: str,
    task: str,
    target_file: str,
    prompt_file_view: str,
    original_file_text: str,
    omitted_blocks: dict[str, str],
    registration_guidance: str,
    attempt_index: int,
    max_attempts: int,
    source_anchor: str,
    forbidden_benchmark_names: set[str],
    require_new_benchmark_name: bool,
) -> str:
    current_prompt = prompt
    for inference_attempt in range(1, 3):
        print("[generate] prompt-begin", flush=True)
        print(current_prompt, flush=True)
        print("[generate] prompt-end", flush=True)
        print(f'[generate] starting local patch generation for {target_file}', flush=True)
        print(
            f"[generate] waiting on Gemini inference for attempt {attempt_index}/{max_attempts} "
            f"using source anchor {source_anchor}",
            flush=True,
        )
        started_at = time.monotonic()
        resp = model.generate_content(current_prompt)
        elapsed_s = time.monotonic() - started_at
        print(f"[generate] Gemini inference returned after {elapsed_s:.1f}s", flush=True)
        generated_file = normalize_generated_file(resp.text)
        generated_file = restore_prompt_file_omissions(generated_file, omitted_blocks)
        generated_file = repair_truncated_tail(generated_file, original_file_text)
        reason = validate_generated_source_output(
            generated_file,
            original_text=prompt_file_view,
            forbidden_benchmark_names=forbidden_benchmark_names,
            require_new_benchmark_name=require_new_benchmark_name,
        )
        if reason is None:
            return generated_file
        print(f"[generate] rejected model response: {reason}", flush=True)
        current_prompt = INVALID_GENERATION_PROMPT.format(
            reason=reason,
            target_file=target_file,
            task=task,
            registration_guidance=registration_guidance
            or "Preserve the existing local registration idiom exactly.",
            current_file=prompt_file_view,
        )
    print("[generate] all inference attempts rejected by validation; falling back to original file", flush=True)
    return original_file_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--target-file', required=True)
    parser.add_argument('--binary-name', default='')
    parser.add_argument('--reference-microbenchmark', default='')
    parser.add_argument('--source-microbenchmark', default='')
    parser.add_argument('--current-file', required=True)
    parser.add_argument('--retrieval-json', default='sample_retrieval.json')
    parser.add_argument('--attempt-index', type=int, default=1)
    parser.add_argument('--max-attempts', type=int, default=1)
    parser.add_argument('--feature-guidance', default='')
    parser.add_argument('--diagnosis-file', default='')
    parser.add_argument('--working-state-file', default='')
    parser.add_argument('--retry-feedback-file', default='')
    parser.add_argument('--initial-generation-mode', default='derived_from_nearest')
    parser.add_argument('--forbidden-benchmark-names-file', default='')
    parser.add_argument('--output', default=PATCH_FILENAME)
    args = parser.parse_args()

    model = create_text_model(GENERATION_MODEL_NAME)

    retrieved_contexts = json.loads(load_text(RETRIEVALS_DIR / args.retrieval_json))
    retrieved = render_retrieved_context(
        retrieved_contexts,
        target_file=args.target_file,
        source_microbenchmark=args.source_microbenchmark,
    )
    current_file = load_text(Path(args.current_file))
    prompt_file_view, omitted_blocks = prepare_prompt_file_view(current_file)
    edit_location_guidance = build_edit_location_guidance(
        current_file,
        args.source_microbenchmark or args.reference_microbenchmark or args.task,
    )
    registration_guidance = build_registration_idiom_guidance(
        current_file,
        args.source_microbenchmark or args.reference_microbenchmark or args.task,
    )
    retry_feedback = ''
    if args.retry_feedback_file:
        retry_path = Path(args.retry_feedback_file)
        if retry_path.exists():
            retry_feedback = load_text(retry_path).strip()
    corrective_iteration = bool(retry_feedback)
    if not retry_feedback:
        retry_feedback = (
            '- No previous attempt feedback. This is attempt 1: create one nearby variant, optimize IPC plus one or two dominant frontend/control counters first, '
            'and keep remaining counters directional only.'
        )
    working_state = ''
    if args.working_state_file:
        working_state_path = Path(args.working_state_file)
        if working_state_path.exists():
            working_state = load_text(working_state_path).strip()
    if not working_state:
        working_state = (
            '- No carried-forward attempt state was provided. '
            'If the current file only contains the original reference benchmark, add exactly one derived benchmark.'
        )
    diagnosis = ''
    if args.diagnosis_file:
        diagnosis_path = Path(args.diagnosis_file)
        if diagnosis_path.exists():
            diagnosis = load_text(diagnosis_path).strip()
    carried_forward_iteration = _is_carried_forward_iteration(
        corrective_iteration=corrective_iteration,
        working_state=working_state,
        reference_microbenchmark=args.reference_microbenchmark,
        source_microbenchmark=args.source_microbenchmark,
    )
    forbidden_benchmark_names = _load_forbidden_benchmark_names(args.forbidden_benchmark_names_file)
    prompt_task = _build_prompt_task(
        corrective_iteration=corrective_iteration,
        carried_forward_iteration=carried_forward_iteration,
        original_task=args.task,
        target_file=args.target_file,
        reference_microbenchmark=args.reference_microbenchmark,
        source_microbenchmark=args.source_microbenchmark,
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
    working_state_block = _format_optional_block(
        "Working attempt state",
        working_state,
        max_lines=6,
        max_chars=700,
    )
    diagnosis_block = _format_optional_block(
        "Diagnostician playbook",
        diagnosis,
        max_lines=12,
        max_chars=1400,
    )
    feature_guidance_block = _format_optional_block(
        "Feature-specific edit guidance",
        compact_feature_guidance,
        max_lines=8,
        max_chars=900,
    )
    refinement_feedback_block = _format_optional_block(
        "Refinement feedback",
        retry_feedback,
        max_lines=8,
        max_chars=900,
    )
    retrieved_block = ""
    if retrieved and not _is_empty_retrieval_notice(retrieved) and not corrective_iteration:
        retrieved_block = _format_optional_block(
            "Retrieved context",
            retrieved,
            max_lines=12,
            max_chars=1400,
        )

    prompt = PATCH_PROMPT.format(
        task=prompt_task,
        benchmark_mode_constraints=_build_benchmark_mode_constraints(
            carried_forward_iteration=carried_forward_iteration
        ),
        benchmark_name_constraints=_build_benchmark_name_constraints(
            carried_forward_iteration=carried_forward_iteration,
            reference_microbenchmark=args.reference_microbenchmark,
            source_microbenchmark=args.source_microbenchmark,
        ),
        reference_microbenchmark=args.reference_microbenchmark.strip() or args.source_microbenchmark.strip() or '(not provided)',
        source_microbenchmark=args.source_microbenchmark.strip() or '(not provided)',
        attempt_index=args.attempt_index,
        max_attempts=args.max_attempts,
        initial_generation_mode=args.initial_generation_mode.strip() or 'derived_from_nearest',
        iteration_mode_guidance=(
            "This is iterative correction on the current generated benchmark. "
            "Do not rewrite from scratch. Make one minimal corrective patch only. "
            "Prioritize buildability, IPC, and the single biggest remaining hardware-counter miss. "
            "Protect counters already close to target and change one primary lever at a time."
            if corrective_iteration
            else (
                "This is attempt 1. Add one compact from-scratch benchmark that stays close to the source family without cloning the reference benchmark body. "
                "Optimize buildability, IPC, and at most one or two dominant frontend/control counters first. "
                "Keep memory-side counters directional only and preserve benchmark character."
                if args.initial_generation_mode.strip() == 'from_scratch'
                else "This is attempt 1. Add one nearby local variant close to the source benchmark family. "
                "Optimize buildability, IPC, and at most one or two dominant frontend/control counters first. "
                "Keep remaining counters directional only and preserve benchmark character."
            )
        ),
        working_state=working_state,
        diagnosis_block=diagnosis_block,
        feature_guidance_block=feature_guidance_block,
        working_state_block=working_state_block,
        refinement_feedback_block=refinement_feedback_block,
        retrieved_block=retrieved_block,
        target_file=args.target_file,
        current_file=prompt_file_view,
    )
    generated_file = _generate_validated_file(
        model,
        prompt=prompt,
        task=prompt_task,
        target_file=args.target_file,
        prompt_file_view=prompt_file_view,
        original_file_text=current_file,
        omitted_blocks=omitted_blocks,
        registration_guidance=registration_guidance,
        attempt_index=args.attempt_index,
        max_attempts=args.max_attempts,
        source_anchor=args.source_microbenchmark or args.reference_microbenchmark or "(unknown)",
        forbidden_benchmark_names=forbidden_benchmark_names,
        require_new_benchmark_name=not carried_forward_iteration,
    )
    patch_text = build_unified_diff(current_file, generated_file, args.target_file)
    if not patch_text.strip():
        raise RuntimeError('Model returned no effective file changes; refusing to emit an empty patch.')
    out_path = PATCHES_DIR / args.output
    out_path.write_text(patch_text, encoding='utf-8')
    print(f'[generate] wrote patch to {out_path}', flush=True)

    # Persist cumulative token usage for cross-process aggregation
    from llm_provider import save_token_usage
    token_usage_path = PATCHES_DIR / f"{args.output}.token_usage.json"
    save_token_usage(token_usage_path)


if __name__ == '__main__':
    main()
