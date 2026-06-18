#!/usr/bin/env python3
"""Materialize canonical successful-edit exemplars from raw pipeline outputs.

The live/cloud pipeline records every run under ``outputs/compile_logs`` and the
rewritten files under ``outputs/repaired``. This is verbose and not the shape the
RAG/SFT stages consume. ``06_build_tuning_dataset.py`` expects each successful
edit normalized under ``data/successful_edits/<id>/`` as the triple:

    metadata.json   - task, target_file, binary_name, retrieval_json, ...
    original.cpp    - the file before the edit
    final.cpp       - the compiled, successful rewrite

This script walks the success=True compile logs and produces exactly that layout,
copying the referenced ``final_file`` and ``current_file`` and deriving the
matching retrieval JSON name. Identical final files (same task, same content) are
de-duplicated so the tuning set is not skewed by repeated attempts.

It is read-only with respect to the raw outputs; it only writes under
``data/successful_edits/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from config.settings import (
    COMPILE_LOGS_DIR,
    RETRIEVALS_DIR,
    SUCCESSFUL_EDITS_DIR,
)

# Fields worth carrying into the per-edit metadata.json. Anything verbose
# (stdout/stderr/full attempt logs) is intentionally dropped.
META_FIELDS = [
    'task',
    'reference_microbenchmark',
    'source_microbenchmark',
    'target_file',
    'binary_name',
    'feature_name',
    'direction',
    'magnitude',
    'performance_success',
    'new_microbenchmark_name',
    'baseline_value',
    'candidate_value',
    'observed_delta',
    'grouped_job_id',
    'task_id',
    'compile_attempt_count',
    'repaired',
]


def _is_success(log: dict) -> bool:
    return str(log.get('success')).strip().lower() == 'true'


def _retrieval_name_for(log_path: Path) -> str:
    """Derive the retrieval JSON filename for a given compile-log filename.

    Compile logs look like ``<base>__attempt_N.json`` and the matching retrieval
    file is ``<base>_retrieval.json``.
    """
    stem = log_path.stem  # drops .json
    if '__attempt_' in stem:
        stem = stem.split('__attempt_')[0]
    return f'{stem}_retrieval.json'


def _content_key(log: dict, final_path: Path) -> str:
    """De-dup key: task identity + final file content hash."""
    h = hashlib.sha1()
    h.update((log.get('task', '') or '').encode('utf-8', 'ignore'))
    h.update(b'\x00')
    h.update((log.get('target_file', '') or '').encode('utf-8', 'ignore'))
    h.update(b'\x00')
    try:
        h.update(final_path.read_bytes())
    except OSError:
        h.update(final_path.name.encode())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dest',
        default=str(SUCCESSFUL_EDITS_DIR),
        help='Output directory for normalized successful edits.',
    )
    parser.add_argument(
        '--dedupe',
        action='store_true',
        default=True,
        help='Skip identical (task, final-content) exemplars (default on).',
    )
    parser.add_argument(
        '--no-dedupe',
        dest='dedupe',
        action='store_false',
        help='Keep every successful attempt, even duplicates.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report what would be written without copying files.',
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    logs = sorted(COMPILE_LOGS_DIR.glob('*.json'))

    seen: set[str] = set()
    written = 0
    skipped_dupe = 0
    skipped_missing = 0
    skipped_fail = 0

    for log_path in logs:
        try:
            log = json.loads(log_path.read_text(errors='ignore'))
        except (json.JSONDecodeError, OSError):
            continue
        if not _is_success(log):
            skipped_fail += 1
            continue

        final_src = log.get('final_file') or log.get('final_patch_file')
        original_src = log.get('current_file') or log.get('live_source_file')
        if not final_src or not original_src:
            skipped_missing += 1
            continue
        final_path = Path(final_src)
        original_path = Path(original_src)
        if not final_path.exists() or not original_path.exists():
            skipped_missing += 1
            continue

        if args.dedupe:
            key = _content_key(log, final_path)
            if key in seen:
                skipped_dupe += 1
                continue
            seen.add(key)

        example_id = log_path.stem
        edit_dir = dest / example_id

        retrieval_name = _retrieval_name_for(log_path)
        retrieval_exists = (RETRIEVALS_DIR / retrieval_name).exists()

        meta = {k: log[k] for k in META_FIELDS if k in log}
        meta['example_id'] = example_id
        meta['retrieval_json'] = retrieval_name if retrieval_exists else ''
        meta['source_compile_log'] = log_path.name

        if args.dry_run:
            written += 1
            continue

        edit_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_path, edit_dir / 'original.cpp')
        shutil.copyfile(final_path, edit_dir / 'final.cpp')
        (edit_dir / 'metadata.json').write_text(
            json.dumps(meta, indent=2), encoding='utf-8'
        )
        written += 1

    action = 'would write' if args.dry_run else 'wrote'
    print(f'{action} {written} successful-edit exemplars to {dest}')
    print(
        f'  skipped: {skipped_fail} non-success, '
        f'{skipped_missing} missing files, {skipped_dupe} duplicates'
    )


if __name__ == '__main__':
    main()
