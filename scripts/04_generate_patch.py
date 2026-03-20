#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import vertexai
from vertexai.generative_models import GenerativeModel

from config.settings import (
    GENERATION_MODEL_NAME,
    LOCATION,
    PATCHES_DIR,
    PATCH_FILENAME,
    PROJECT_ID,
    RETRIEVALS_DIR,
)
from local_patch_utils import (
    build_local_source_excerpt,
    normalize_generated_patch,
    render_retrieved_context,
)

PATCH_PROMPT = """You are editing an existing Folly benchmark file with a small local patch.

Rules:
- Return only a unified diff patch.
- The patch must apply cleanly to the target file.
- Keep the patch local to the provided source benchmark excerpt.
- Keep the original reference benchmark unchanged.
- If the current working benchmark anchor is still the original reference benchmark, add exactly one new derived benchmark registration immediately after it.
- If the current working benchmark anchor is already a carried-forward generated benchmark, refine that carried-forward benchmark in place and do not add a second generated benchmark.
- Preserve benchmark harness structure, includes, namespace usage, and main() unless the task explicitly requires a change.
- Use only APIs present in the local excerpt, retrieved context, or already present in the current file.
- Prefer duplicating and locally editing the existing benchmark code or registration instead of rewriting unrelated code.
- Do not invent new build targets or unrelated helper utilities.
- Keep the result compilable in the existing Folly/DCPerf build.

Task:
{task}

Original reference benchmark:
{reference_microbenchmark}

Current working benchmark anchor:
{source_microbenchmark}

Attempt:
{attempt_index} of {max_attempts}

Working attempt state:
{working_state}

Feature-specific edit guidance:
{feature_guidance}

Refinement feedback:
{refinement_feedback}

Target file:
{target_file}

Local source excerpt from the current target file:
{local_excerpt}

Retrieved context:
{retrieved}

Return only a unified diff patch against the target file.
"""


def load_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore')


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
    parser.add_argument('--working-state-file', default='')
    parser.add_argument('--retry-feedback-file', default='')
    parser.add_argument('--output', default=PATCH_FILENAME)
    args = parser.parse_args()

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(GENERATION_MODEL_NAME)

    retrieved_contexts = json.loads(load_text(RETRIEVALS_DIR / args.retrieval_json))
    retrieved = render_retrieved_context(
        retrieved_contexts,
        target_file=args.target_file,
        source_microbenchmark=args.source_microbenchmark,
    )
    current_file = load_text(Path(args.current_file))
    local_excerpt = build_local_source_excerpt(
        current_file,
        args.source_microbenchmark or args.task,
    )
    retry_feedback = ''
    if args.retry_feedback_file:
        retry_path = Path(args.retry_feedback_file)
        if retry_path.exists():
            retry_feedback = load_text(retry_path).strip()
    if not retry_feedback:
        retry_feedback = '- No previous attempt feedback. Produce the best first-pass local patch.'
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

    prompt = PATCH_PROMPT.format(
        task=args.task.strip(),
        reference_microbenchmark=args.reference_microbenchmark.strip() or args.source_microbenchmark.strip() or '(not provided)',
        source_microbenchmark=args.source_microbenchmark.strip() or '(not provided)',
        attempt_index=args.attempt_index,
        max_attempts=args.max_attempts,
        working_state=working_state,
        feature_guidance=args.feature_guidance.strip() or '- No additional feature guidance was provided.',
        refinement_feedback=retry_feedback,
        target_file=args.target_file,
        local_excerpt=local_excerpt,
        retrieved=retrieved,
    )
    print(f'[generate] starting local patch generation for {args.target_file}', flush=True)
    resp = model.generate_content(prompt)
    out_path = PATCHES_DIR / args.output
    out_path.write_text(normalize_generated_patch(resp.text), encoding='utf-8')
    print(f'[generate] wrote patch to {out_path}', flush=True)


if __name__ == '__main__':
    main()
