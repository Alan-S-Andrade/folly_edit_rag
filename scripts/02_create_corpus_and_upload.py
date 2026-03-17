#!/usr/bin/env python3
"""Create a Vertex AI RAG corpus and upload local Folly RAG docs."""
from __future__ import annotations

import json
from pathlib import Path

import vertexai
from vertexai.preview import rag

from config.settings import (
    CORPUS_INFO_FILENAME,
    EMBEDDING_MODEL,
    LOCATION,
    MANIFESTS_DIR,
    MANIFEST_FILENAME,
    PROJECT_ID,
    RAG_DOCS_DIR,
)

CORPUS_DISPLAY_NAME = 'folly-benchmark-edit-rag'


def main() -> None:
    if not PROJECT_ID or PROJECT_ID == 'your-project-id':
        raise SystemExit(
            'Set VERTEX_PROJECT_ID to a Google Cloud project you can access before running step 2.'
        )

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    embedding_model_config = rag.EmbeddingModelConfig(
        publisher_model=EMBEDDING_MODEL
    )

    corpus = rag.create_corpus(
        display_name=CORPUS_DISPLAY_NAME,
        embedding_model_config=embedding_model_config,
    )
    print('Created corpus:', corpus.name)

    manifest = json.loads((MANIFESTS_DIR / MANIFEST_FILENAME).read_text())
    uploaded = []
    failed = []

    for entry in manifest:
        rel_doc = entry['doc_file']
        doc_path = RAG_DOCS_DIR / rel_doc
        try:
            print(f"Uploading {rel_doc} ({doc_path.stat().st_size} bytes)")
            rag_file = rag.upload_file(
                corpus_name=corpus.name,
                path=str(doc_path),
                display_name=Path(rel_doc).name,
                description=f"{entry.get('kind', 'rag_doc')}::{entry.get('logical_name', '')}",
            )
            uploaded.append({
                'doc_file': rel_doc,
                'display_name': Path(rel_doc).name,
                'rag_file_name': getattr(rag_file, 'name', ''),
                'kind': entry.get('kind', ''),
                'logical_name': entry.get('logical_name', ''),
            })
        except Exception as exc:  # pragma: no cover - depends on live Vertex env
            failed.append({
                'doc_file': rel_doc,
                'kind': entry.get('kind', ''),
                'logical_name': entry.get('logical_name', ''),
                'error': repr(exc),
            })
            print(f'FAILED: {rel_doc} {exc!r}')

    corpus_info = {
        'corpus_name': corpus.name,
        'display_name': CORPUS_DISPLAY_NAME,
        'uploaded_count': len(uploaded),
        'failed_count': len(failed),
        'uploaded': uploaded,
        'failed': failed,
    }
    out_path = MANIFESTS_DIR / CORPUS_INFO_FILENAME
    out_path.write_text(json.dumps(corpus_info, indent=2), encoding='utf-8')
    print(f'Wrote corpus info to {out_path}')


if __name__ == '__main__':
    main()
