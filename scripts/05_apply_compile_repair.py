#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
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
    PATCHES_DIR,
    PATCHED_FILENAME,
    PATCH_FILENAME,
    PROJECT_ID,
    REPAIRED_DIR,
    REPAIRED_PATCH_FILENAME,
    RETRIEVALS_DIR,
    RUN_TIMEOUT_SEC,
    SUCCESSFUL_EDITS_DIR,
)

REPAIR_PROMPT = """You are repairing a unified diff patch for an existing Folly benchmark file.

Rules:
- Return only a unified diff patch.
- Preserve the requested benchmark intent.
- Minimize unrelated edits.
- Use only APIs visible in the current file or retrieved context.
- Keep the patch compilable in the existing Folly/DCPerf build.

Task:
{task}

Target file:
{target_file}

Current file contents:
{current_file}

Retrieved context:
{retrieved}

Broken patch:
{bad_patch}

Patch/apply/build errors:
{errors}

Return only a corrected unified diff patch.
"""


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None):
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def try_apply_patch(src_file: Path, patch_file: Path, workdir: Path) -> tuple[bool, str, str, Path]:
    target_copy = workdir / src_file.name
    shutil.copy2(src_file, target_copy)
    rc, so, se = run(['patch', '-u', str(target_copy), str(patch_file)], cwd=workdir)
    return rc == 0, so, se, target_copy


def compile_binary(binary_name: str) -> tuple[int, str, str]:
    return run(['cmake', '--build', str(CMAKE_BUILD_DIR), '--parallel', str(CMAKE_PARALLEL), '--target', binary_name], timeout=RUN_TIMEOUT_SEC)


def repair_patch(model: GenerativeModel, task: str, target_file: str, current_file: str, retrieved_contexts: list[dict], bad_patch: str, errors: str) -> str:
    retrieved = '\n\n'.join(
        f"[CONTEXT {i + 1}]\n{c['text']}" for i, c in enumerate(retrieved_contexts)
    )
    prompt = REPAIR_PROMPT.format(
        task=task.strip(),
        target_file=target_file,
        current_file=current_file,
        retrieved=retrieved,
        bad_patch=bad_patch,
        errors=errors,
    )
    return model.generate_content(prompt).text


def save_success_example(example_id: str, task: str, target_file: str, binary_name: str, patch_text: str, final_file_text: str, retrieval_json: str) -> None:
    out_dir = SUCCESSFUL_EDITS_DIR / example_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'patch.diff').write_text(patch_text, encoding='utf-8')
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
    parser.add_argument('--patch-file', default=PATCH_FILENAME)
    parser.add_argument('--example-id', default='latest_success')
    args = parser.parse_args()

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)

    src_file = FOLLY_SRC_ROOT / args.target_file
    current_file = src_file.read_text(encoding='utf-8', errors='ignore')
    patch_path = PATCHES_DIR / args.patch_file
    retrieval_contexts = json.loads((RETRIEVALS_DIR / args.retrieval_json).read_text())
    patch_text = patch_path.read_text(encoding='utf-8', errors='ignore')

    record = {
        'task': args.task,
        'target_file': args.target_file,
        'binary_name': args.binary_name,
        'patch_file': str(patch_path),
    }

    with tempfile.TemporaryDirectory(prefix='folly_patch_') as td:
        workdir = Path(td)
        ok, so, se, patched_copy = try_apply_patch(src_file, patch_path, workdir)
        record['first_patch_apply_stdout'] = so
        record['first_patch_apply_stderr'] = se
        record['first_patch_apply_success'] = ok

        if not ok:
            fixed_patch = repair_patch(model, args.task, args.target_file, current_file, retrieval_contexts, patch_text, so + '\n' + se)
            fixed_patch_path = REPAIRED_DIR / REPAIRED_PATCH_FILENAME
            fixed_patch_path.write_text(fixed_patch, encoding='utf-8')
            ok, so, se, patched_copy = try_apply_patch(src_file, fixed_patch_path, workdir)
            record['second_patch_apply_stdout'] = so
            record['second_patch_apply_stderr'] = se
            record['second_patch_apply_success'] = ok
            patch_text = fixed_patch
            patch_path = fixed_patch_path

        if ok:
            final_text = patched_copy.read_text(encoding='utf-8', errors='ignore')
            (REPAIRED_DIR / PATCHED_FILENAME).write_text(final_text, encoding='utf-8')
            # Apply patch to actual source tree before build.
            backup = src_file.with_suffix(src_file.suffix + '.bak_folly_rag')
            shutil.copy2(src_file, backup)
            try:
                shutil.copy2(patched_copy, src_file)
                rc, bso, bse = compile_binary(args.binary_name)
                record['compile_rc'] = rc
                record['compile_stdout'] = bso
                record['compile_stderr'] = bse
                record['success'] = rc == 0
                if rc == 0:
                    save_success_example(args.example_id, args.task, args.target_file, args.binary_name, patch_text, final_text, args.retrieval_json)
            finally:
                shutil.move(str(backup), str(src_file))
        else:
            record['compile_rc'] = None
            record['compile_stdout'] = ''
            record['compile_stderr'] = 'Patch could not be applied.'
            record['success'] = False

    out_path = COMPILE_LOGS_DIR / f'{args.example_id}.json'
    out_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))
    print(f'Wrote compile log to {out_path}')


if __name__ == '__main__':
    main()
