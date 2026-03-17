#!/usr/bin/env python3
"""Prepare a Folly/DCPerf RAG corpus for patch-based benchmark editing.

Produces local .txt docs in data/rag_docs/ grouped into:
  - benchmark_src : benchmark files and per-BENCHMARK chunks
  - build_map     : CMake BENCHMARK declarations
  - successful_edit: prior successful edit exemplars (optional)

The docs are intended for Vertex AI RAG local-file upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from config.settings import (
    FOLLY_CMAKE,
    FOLLY_SRC_ROOT,
    FOLLY_TEST_ROOT,
    MANIFESTS_DIR,
    MANIFEST_FILENAME,
    MAX_LOCAL_UPLOAD_BYTES,
    RAG_DOCS_DIR,
    SUCCESSFUL_EDITS_DIR,
    WDL_BENCH_ROOT,
)

INCLUDE_RE = re.compile(r'#include\s*[<"]([^>"]+)[>"]')
BENCHMARK_START_RE = re.compile(r'^\s*BENCHMARK(?:_RELATIVE)?\s*\(([^,\)]+)')
MAIN_RE = re.compile(r'\bint\s+main\s*\(')
CMAKE_BENCH_RE = re.compile(r'^\s*BENCHMARK\s+(\S+)\s+SOURCES\s+(.+?)\s*$')
FBTHRIFT_SRC_ROOT = WDL_BENCH_ROOT / 'wdl_sources' / 'fbthrift'
EXTRA_BENCHMARK_SOURCES = [
    FBTHRIFT_SRC_ROOT / 'thrift' / 'lib' / 'cpp' / 'util' / 'test' / 'VarintUtilsBench.cpp',
    FBTHRIFT_SRC_ROOT / 'thrift' / 'lib' / 'cpp2' / 'test' / 'ProtocolBench.cpp',
]
FBTHRIFT_BUILD_MAPS = [
    (
        'VarintUtilsBench',
        Path('thrift/lib/cpp/util/test/CMakeLists.txt'),
        'VarintUtilsBench.cpp',
        'add_executable(VarintUtilsBench VarintUtilsBench.cpp)',
    ),
    (
        'ProtocolBench',
        Path('thrift/lib/cpp2/test/CMakeLists.txt'),
        'ProtocolBench.cpp + generated thrift sources',
        'add_executable(ProtocolBench ${SOURCE_FILES} ${ADDITIONAL_SOURCE_FILES} ${GENERATED_THRIFT_SOURCES})',
    ),
]


def clean_text(s: str) -> str:
    s = s.replace("\x00", "")
    return ''.join(ch for ch in s if ch == '\n' or ch == '\t' or ord(ch) >= 32)


def digest_text(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8', errors='ignore')).hexdigest()


def infer_tags(code: str, path: Path) -> list[str]:
    tags = set()
    lower = code.lower()
    if 'BenchmarkSuspender' in code:
        tags.add('benchmark-suspender')
    if 'doNotOptimizeAway' in code:
        tags.add('dnoa')
    if 'folly::Random' in code or 'Random::' in code or 'Random.h' in code:
        tags.add('random')
    if 'ThreadLocal' in code:
        tags.add('thread-local')
    if 'std::random_device' in code or 'mt19937' in code or 'xoshiro' in lower:
        tags.add('rng')
    if 'for (' in code or 'for(' in code:
        tags.add('loop')
    if 'if (' in code or 'if(' in code:
        tags.add('branch')
    if MAIN_RE.search(code):
        tags.add('has-main')
    if 'gflags::ParseCommandLineFlags' in code:
        tags.add('gflags')
    if 'runBenchmarks' in code:
        tags.add('run-benchmarks')
    if path.name.endswith('Benchmark.cpp'):
        tags.add('benchmark-file')
    return sorted(tags)


def extract_benchmark_blocks(code: str) -> list[tuple[str, str]]:
    lines = code.splitlines()
    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = BENCHMARK_START_RE.match(line)
        if m:
            starts.append((i, m.group(1).strip()))
    blocks: list[tuple[str, str]] = []
    for idx, (start, name) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        blocks.append((name, '\n'.join(lines[start:end]).strip() + '\n'))
    return blocks


def write_doc(text: str, subdir: str, logical_name: str) -> dict:
    text = clean_text(text)
    digest = digest_text(text)
    out_dir = RAG_DOCS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{digest[:16]}.txt'
    out_path.write_text(text, encoding='utf-8')
    return {
        'doc_file': str(out_path.relative_to(RAG_DOCS_DIR)),
        'logical_name': logical_name,
        'sha256': digest,
        'size_bytes': out_path.stat().st_size,
        'oversize_local_upload': out_path.stat().st_size > MAX_LOCAL_UPLOAD_BYTES,
    }


def prepare_benchmark_source_docs(limit: int | None = None) -> list[dict]:
    manifest: list[dict] = []
    files = sorted(FOLLY_TEST_ROOT.glob('*Benchmark*.cpp'))
    files.extend([p for p in EXTRA_BENCHMARK_SOURCES if p.exists()])
    files = sorted(set(files))
    if limit is not None:
        files = files[:limit]

    for path in files:
        code = clean_text(path.read_text(errors='ignore'))
        if path.is_relative_to(FOLLY_SRC_ROOT):
            rel = path.relative_to(FOLLY_SRC_ROOT)
        else:
            rel = path.relative_to(FBTHRIFT_SRC_ROOT)
        headers = sorted(set(INCLUDE_RE.findall(code)))
        tags = infer_tags(code, path)
        blocks = extract_benchmark_blocks(code)

        overview = '\n'.join([
            f'PATH: {rel}',
            'KIND: benchmark_src_file',
            f'FILENAME: {path.name}',
            f'HEADERS: {", ".join(headers)}',
            f'TAGS: {", ".join(tags)}',
            f'BENCHMARK_NAMES: {", ".join(name for name, _ in blocks)}',
            'SUMMARY: Folly benchmark source file used for patch-grounded editing.',
            'CODE:',
            code,
        ])
        entry = write_doc(overview, 'benchmark_src', str(rel))
        entry.update({
            'kind': 'benchmark_src_file',
            'source_path': str(path),
            'relative_path': str(rel),
            'headers': headers,
            'tags': tags,
            'benchmark_names': [name for name, _ in blocks],
        })
        manifest.append(entry)

        for name, block in blocks:
            block_doc = '\n'.join([
                f'PATH: {rel}',
                'KIND: benchmark_block',
                f'FILENAME: {path.name}',
                f'BENCHMARK_NAME: {name}',
                f'HEADERS: {", ".join(headers)}',
                f'TAGS: {", ".join(tags)}',
                'SUMMARY: Individual Folly BENCHMARK block for patch retrieval.',
                'CODE:',
                block,
            ])
            block_entry = write_doc(block_doc, 'benchmark_src', f'{rel}::{name}')
            block_entry.update({
                'kind': 'benchmark_block',
                'source_path': str(path),
                'relative_path': str(rel),
                'benchmark_name': name,
                'headers': headers,
                'tags': tags,
            })
            manifest.append(block_entry)

        if MAIN_RE.search(code):
            main_idx = code.find('int main')
            main_block = code[main_idx:] if main_idx >= 0 else ''
            if main_block:
                main_doc = '\n'.join([
                    f'PATH: {rel}',
                    'KIND: benchmark_main',
                    f'FILENAME: {path.name}',
                    f'HEADERS: {", ".join(headers)}',
                    f'TAGS: {", ".join(tags)}',
                    'SUMMARY: Benchmark file main() / harness section.',
                    'CODE:',
                    main_block,
                ])
                main_entry = write_doc(main_doc, 'benchmark_src', f'{rel}::main')
                main_entry.update({
                    'kind': 'benchmark_main',
                    'source_path': str(path),
                    'relative_path': str(rel),
                    'headers': headers,
                    'tags': tags,
                })
                manifest.append(main_entry)
    return manifest


def prepare_build_map_docs() -> list[dict]:
    manifest: list[dict] = []
    cmake = clean_text(FOLLY_CMAKE.read_text(errors='ignore'))
    rel = FOLLY_CMAKE.relative_to(FOLLY_SRC_ROOT)
    for line in cmake.splitlines():
        m = CMAKE_BENCH_RE.match(line)
        if not m:
            continue
        binary, sources = m.group(1), m.group(2).strip()
        doc = '\n'.join([
            f'PATH: {rel}',
            'KIND: build_map',
            f'BINARY_NAME: {binary}',
            f'SOURCES: {sources}',
            'SUMMARY: CMake BENCHMARK declaration mapping a binary name to its source file(s).',
            'CODE:',
            line,
        ])
        entry = write_doc(doc, 'build_map', f'{binary}:{sources}')
        entry.update({
            'kind': 'build_map',
            'relative_path': str(rel),
            'binary_name': binary,
            'sources': sources,
        })
        manifest.append(entry)

    for binary, cmake_rel, sources, code_line in FBTHRIFT_BUILD_MAPS:
        doc = '\n'.join([
            f'PATH: {cmake_rel}',
            'KIND: build_map',
            f'BINARY_NAME: {binary}',
            f'SOURCES: {sources}',
            'SUMMARY: fbthrift CMake executable mapping for a benchmark binary.',
            'CODE:',
            code_line,
        ])
        entry = write_doc(doc, 'build_map', f'{binary}:{sources}')
        entry.update({
            'kind': 'build_map',
            'relative_path': str(cmake_rel),
            'binary_name': binary,
            'sources': sources,
        })
        manifest.append(entry)
    return manifest


def iter_successful_edit_dirs() -> Iterable[Path]:
    if not SUCCESSFUL_EDITS_DIR.exists():
        return []
    return sorted([p for p in SUCCESSFUL_EDITS_DIR.iterdir() if p.is_dir()])


def prepare_successful_edit_docs() -> list[dict]:
    manifest: list[dict] = []
    for edit_dir in iter_successful_edit_dirs():
        meta_path = edit_dir / 'metadata.json'
        patch_path = edit_dir / 'patch.diff'
        final_path = edit_dir / 'final.cpp'
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get('performance_success') is False:
            continue
        patch_text = patch_path.read_text(errors='ignore') if patch_path.exists() else ''
        final_text = final_path.read_text(errors='ignore') if final_path.exists() else ''
        doc = '\n'.join([
            f"EDIT_ID: {edit_dir.name}",
            'KIND: successful_edit',
            f"TARGET_FILE: {meta.get('target_file', '')}",
            f"TASK: {meta.get('task', '')}",
            f"BINARY_NAME: {meta.get('binary_name', '')}",
            f"COMPILE_SUCCESS: {meta.get('compile_success', True)}",
            'SUMMARY: Previously successful Folly edit trajectory exemplar.',
            'PATCH:',
            patch_text,
            'FINAL_FILE:',
            final_text,
        ])
        entry = write_doc(doc, 'successful_edit', edit_dir.name)
        entry.update({
            'kind': 'successful_edit',
            'edit_id': edit_dir.name,
            'metadata': meta,
        })
        manifest.append(entry)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit-benchmark-files', type=int, default=None)
    args = parser.parse_args()

    manifest: list[dict] = []
    manifest.extend(prepare_benchmark_source_docs(limit=args.limit_benchmark_files))
    manifest.extend(prepare_build_map_docs())
    manifest.extend(prepare_successful_edit_docs())

    out_path = MANIFESTS_DIR / MANIFEST_FILENAME
    out_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Wrote {len(manifest)} docs to {RAG_DOCS_DIR}')
    print(f'Manifest: {out_path}')


if __name__ == '__main__':
    main()
