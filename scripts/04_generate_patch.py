#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import vertexai
from vertexai.generative_models import GenerativeModel

from config.settings import (
    LOCATION,
    MODEL_NAME,
    PATCHES_DIR,
    PATCH_FILENAME,
    PROJECT_ID,
    RETRIEVALS_DIR,
)

PATCH_PROMPT = """You are editing an existing Folly benchmark file.

Rules:
- Return only a unified diff patch.
- The patch must apply cleanly to the target file.
- Preserve benchmark harness structure, includes, namespace usage, and main() unless the task explicitly requires a change.
- Use only APIs present in the retrieved context or already present in the target file.
- Prefer minimal edits.
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

Return only a unified diff patch against the target file.
"""


def load_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--target-file', required=True)
    parser.add_argument('--current-file', required=True)
    parser.add_argument('--retrieval-json', default='sample_retrieval.json')
    parser.add_argument('--output', default=PATCH_FILENAME)
    args = parser.parse_args()

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)

    retrieved_contexts = json.loads(load_text(RETRIEVALS_DIR / args.retrieval_json))
    retrieved = '\n\n'.join(
        f"[CONTEXT {i + 1}]\n{c['text']}" for i, c in enumerate(retrieved_contexts)
    )
    current_file = load_text(Path(args.current_file))

    prompt = PATCH_PROMPT.format(
        task=args.task.strip(),
        target_file=args.target_file,
        current_file=current_file,
        retrieved=retrieved,
    )
    resp = model.generate_content(prompt)
    out_path = PATCHES_DIR / args.output
    out_path.write_text(resp.text, encoding='utf-8')
    print(f'Wrote patch to {out_path}')


if __name__ == '__main__':
    main()
