#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.settings import RETRIEVALS_DIR, SUCCESSFUL_EDITS_DIR, TUNING_DIR


def build_row(meta: dict, original_text: str, final_text: str, retrieved_texts: str) -> dict:
    user_text = f"""Task: Edit an existing Folly benchmark source file in place.

Intent:
{meta.get('task', '')}

Target file:
{meta.get('target_file', '')}

Target binary:
{meta.get('binary_name', '')}

Constraints:
- emit the complete updated file contents for the target file
- preserve the existing benchmark harness and main() unless unnecessary
- use only APIs visible in the current file or retrieved context
- keep unrelated code unchanged where possible
- keep the result compilable in the existing Folly/DCPerf build

Original file contents:
{original_text}

Retrieved context:
{retrieved_texts}
"""
    return {
        'contents': [
            {'role': 'user', 'parts': [{'text': user_text}]},
            {'role': 'model', 'parts': [{'text': final_text}]},
        ],
        'metadata': {
            'target_file': meta.get('target_file', ''),
            'binary_name': meta.get('binary_name', ''),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='train.jsonl')
    args = parser.parse_args()

    out_path = TUNING_DIR / args.output
    count = 0
    with out_path.open('w', encoding='utf-8') as out:
        for edit_dir in sorted([p for p in SUCCESSFUL_EDITS_DIR.iterdir() if p.is_dir()]):
            meta_path = edit_dir / 'metadata.json'
            original_path = edit_dir / 'original.cpp'
            final_path = edit_dir / 'final.cpp'
            if not (meta_path.exists() and original_path.exists() and final_path.exists()):
                continue
            meta = json.loads(meta_path.read_text())
            if meta.get('performance_success') is False:
                continue
            original_text = original_path.read_text(errors='ignore')
            final_text = final_path.read_text(errors='ignore')
            retrieval_json_name = meta.get('retrieval_json', '')
            retrieved_texts = ''
            if retrieval_json_name:
                retrieval_path = RETRIEVALS_DIR / retrieval_json_name
                if retrieval_path.exists():
                    hits = json.loads(retrieval_path.read_text())
                    retrieved_texts = '\n\n'.join(h['text'] for h in hits)
            row = build_row(meta, original_text, final_text, retrieved_texts)
            row['metadata'].update({
                key: meta.get(key)
                for key in [
                    'grouped_job_id',
                    'task_id',
                    'source_microbenchmark',
                    'feature_name',
                    'direction',
                    'magnitude',
                    'performance_success',
                    'new_microbenchmark_name',
                    'baseline_value',
                    'candidate_value',
                    'observed_delta',
                ]
                if key in meta
            })
            out.write(json.dumps(row) + '\n')
            count += 1
    print(f'Wrote {count} rows to {out_path}')


if __name__ == '__main__':
    main()
