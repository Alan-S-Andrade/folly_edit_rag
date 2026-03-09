#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import vertexai
from vertexai import rag

from config.settings import (
    CORPUS_INFO_FILENAME,
    LOCATION,
    MANIFESTS_DIR,
    PROJECT_ID,
    RETRIEVALS_DIR,
    RETRIEVAL_RERANK_MODEL,
    TOP_K,
)


def load_corpus_name() -> str:
    corpus_info = json.loads((MANIFESTS_DIR / CORPUS_INFO_FILENAME).read_text())
    return corpus_info['corpus_name']


def retrieve_context(query_text: str, top_k: int = TOP_K, rerank: bool = True) -> list[dict]:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    corpus_name = load_corpus_name()
    retrieval_config = rag.RagRetrievalConfig(top_k=top_k)
    if rerank:
        retrieval_config = rag.RagRetrievalConfig(
            top_k=top_k,
            ranking=rag.Ranking(
                llm_ranker=rag.LlmRanker(model_name=RETRIEVAL_RERANK_MODEL)
            ),
        )

    response = rag.retrieval_query(
        rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
        text=query_text,
        rag_retrieval_config=retrieval_config,
    )

    out = []
    for i, ctx in enumerate(response.contexts.contexts):
        out.append({
            'rank': i + 1,
            'source_uri': getattr(ctx, 'source_uri', ''),
            'text': ctx.text,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, help='Editing task or request')
    parser.add_argument('--target-file', required=True, help='Relative path under wdl_sources/folly')
    parser.add_argument('--binary-name', default='', help='Benchmark binary name if known')
    parser.add_argument('--top-k', type=int, default=TOP_K)
    parser.add_argument('--no-rerank', action='store_true')
    parser.add_argument('--output', default='sample_retrieval.json')
    args = parser.parse_args()

    query = f"""You are retrieving context for a Folly benchmark editing task.

Task:
{args.task}

Target file:
{args.target_file}

Target binary:
{args.binary_name}

Retrieve the most useful context for making a minimal, compilable patch.
Prioritize:
- the exact target file
- BENCHMARK blocks from that file
- the matching CMake BENCHMARK mapping
- similar benchmark blocks from sibling benchmark files
- prior successful edits if available
- nearby API usage and helper patterns
"""

    hits = retrieve_context(query, top_k=args.top_k, rerank=not args.no_rerank)
    out_path = RETRIEVALS_DIR / args.output
    out_path.write_text(json.dumps(hits, indent=2), encoding='utf-8')
    print(f'Retrieved {len(hits)} contexts -> {out_path}')


if __name__ == '__main__':
    main()
