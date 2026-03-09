#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import vertexai
from vertexai.generative_models import GenerativeModel

from config.settings import (
    CMAKE_BUILD_DIR,
    CMAKE_PARALLEL,
    COMPILE_LOGS_DIR,
    FOLLY_SRC_ROOT,
    LOCATION,
    MODEL_NAME,
    PROJECT_ID,
    REPAIRED_DIR,
    RETRIEVALS_DIR,
    RUN_TIMEOUT_SEC,
    SUCCESSFUL_EDITS_DIR,
)

REPAIR_PROMPT = """You are repairing a rewritten Folly benchmark source file.

Rules:
- Return only the complete corrected source file contents for the target file.
- Preserve the requested benchmark intent.
- Minimize unrelated edits.
- Use only APIs visible in the current file or retrieved context.
- Keep the result compilable in the existing Folly/DCPerf build.
- Preserve includes, benchmark harness structure, and main() unless the build errors require a specific change.

Task:
{task}

Target file:
{target_file}

Original source before rewrite:
{original_file}

Current broken rewritten source:
{broken_file}

Retrieved context:
{retrieved}

Build errors:
{errors}

Return only the complete corrected source file contents for {target_file}.
"""


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None):
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def compile_binary(binary_name: str) -> tuple[int, str, str]:
    return run(
        ['cmake', '--build', str(CMAKE_BUILD_DIR), '--parallel', str(CMAKE_PARALLEL), '--target', binary_name],
        timeout=RUN_TIMEOUT_SEC,
    )


def repair_file(
    model: GenerativeModel,
    task: str,
    target_file: str,
    original_file: str,
    broken_file: str,
    retrieved_contexts: list[dict],
    errors: str,
) -> str:
    retrieved = '\n\n'.join(
        f"[CONTEXT {i + 1}]\n{c['text']}" for i, c in enumerate(retrieved_contexts)
    )
    prompt = REPAIR_PROMPT.format(
        task=task.strip(),
        target_file=target_file,
        original_file=original_file,
        broken_file=broken_file,
        retrieved=retrieved,
        errors=errors,
    )
    return model.generate_content(prompt).text


def save_success_example(
    example_id: str,
    task: str,
    target_file: str,
    binary_name: str,
    original_text: str,
    final_file_text: str,
    retrieval_json: str,
) -> None:
    out_dir = SUCCESSFUL_EDITS_DIR / example_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'original.cpp').write_text(original_text, encoding='utf-8')
    (out_dir / 'final.cpp').write_text(final_file_text, encoding='utf-8')
    (out_dir / 'metadata.json').write_text(
        json.dumps(
            {
                'task': task,
                'target_file': target_file,
                'binary_name': binary_name,
                'compile_success': True,
                'retrieval_json': retrieval_json,
            },
            indent=2,
        ),
        encoding='utf-8',
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--target-file', required=True, help='Relative path under wdl_sources/folly')
    parser.add_argument('--binary-name', required=True)
    parser.add_argument('--retrieval-json', default='sample_retrieval.json')
    parser.add_argument('--rewritten-file', default='candidate_full.cpp')
    parser.add_argument('--example-id', default='latest_success')
    args = parser.parse_args()

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)

    src_file = FOLLY_SRC_ROOT / args.target_file
    original_text = src_file.read_text(encoding='utf-8', errors='ignore')
    rewritten_path = REPAIRED_DIR / args.rewritten_file
    retrieval_contexts = json.loads((RETRIEVALS_DIR / args.retrieval_json).read_text())
    rewritten_text = rewritten_path.read_text(encoding='utf-8', errors='ignore')

    record = {
        'task': args.task,
        'target_file': args.target_file,
        'binary_name': args.binary_name,
        'rewritten_file': str(rewritten_path),
    }

    backup = src_file.with_suffix(src_file.suffix + '.bak_folly_rag')
    shutil.copy2(src_file, backup)
    try:
        src_file.write_text(rewritten_text, encoding='utf-8')
        rc, bso, bse = compile_binary(args.binary_name)
        record['first_compile_rc'] = rc
        record['first_compile_stdout'] = bso
        record['first_compile_stderr'] = bse

        final_text = rewritten_text
        repaired = False
        if rc != 0:
            repaired = True
            fixed_text = repair_file(
                model,
                args.task,
                args.target_file,
                original_text,
                rewritten_text,
                retrieval_contexts,
                bso + '\n' + bse,
            )
            repaired_path = REPAIRED_DIR / 'candidate_repair_full.cpp'
            repaired_path.write_text(fixed_text, encoding='utf-8')
            src_file.write_text(fixed_text, encoding='utf-8')
            rc2, so2, se2 = compile_binary(args.binary_name)
            record['second_compile_rc'] = rc2
            record['second_compile_stdout'] = so2
            record['second_compile_stderr'] = se2
            final_text = fixed_text
            record['repaired'] = True
            record['success'] = rc2 == 0
        else:
            record['repaired'] = False
            record['success'] = True

        if record['success']:
            save_success_example(
                args.example_id,
                args.task,
                args.target_file,
                args.binary_name,
                original_text,
                final_text,
                args.retrieval_json,
            )
            (REPAIRED_DIR / 'final_applied.cpp').write_text(final_text, encoding='utf-8')
    finally:
        shutil.move(str(backup), str(src_file))

    out_path = COMPILE_LOGS_DIR / f'{args.example_id}.json'
    out_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))
    print(f'Wrote compile log to {out_path}')


if __name__ == '__main__':
    main()
