# folly_edit_rag

A Folly/DCPerf benchmark-editing RAG pipeline built for full-file in-place source rewrites.

## What it does

This project builds a Vertex AI RAG corpus from:
- `wdl_sources/folly/folly/test/*Benchmark*.cpp`
- `wdl_sources/folly/CMakeLists.txt`
- prior successful edit exemplars under `data/successful_edits/`

It then uses Gemini 2.5 Pro to:
1. retrieve edit context,
2. generate the complete rewritten contents of an existing benchmark file,
3. overwrite the target file in situ for the build,
4. compile the corresponding benchmark binary,
5. repair the rewritten file on failure,
6. save successful rewrites for later supervised fine-tuning.

This is a code-grounding system only. PMU vectors are intentionally not part of the RAG corpus.

## Why this architecture

Vertex AI RAG Engine supports local-file upload with `upload_file()` and documents that the local upload path is a synchronous single-file upload with a 25 MB per-file limit. The retrieval API takes a `rag_retrieval_config=rag.RagRetrievalConfig(top_k=...)`, and reranking can be added through `Ranking` / `LlmRanker`. Gemini 2.5 Pro is available on Vertex AI and supervised fine-tuning is supported for Gemini 2.5 Pro.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Authenticate locally:

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Vertex AI quickstart docs explicitly call out ADC / `gcloud auth application-default login` for local development.

## Configure

Set environment variables before running:

```bash
export VERTEX_PROJECT_ID=your-project-id
export VERTEX_LOCATION=us-central1
export DC_PERF_ROOT=/proj/os-class-PG0/DCPerf
export CMAKE_BUILD_DIR=/proj/os-class-PG0/DCPerf/benchmarks/wdl_bench/wdl_build/build/folly
```

The scripts read these in `config/settings.py`.

## Pipeline

### 1) Prepare the local RAG docs

```bash
python scripts/01_prepare_folly_rag.py
```

This writes local docs under `data/rag_docs/` and a manifest under `data/manifests/folly_rag_manifest.json`.

### 2) Create the Vertex RAG corpus and upload local docs

```bash
python scripts/02_create_corpus_and_upload.py
```

### 3) Retrieve editing context

Example for `RandomBenchmark.cpp` / `random_benchmark`:

```bash
python scripts/03_retrieve_edit_context.py \
  --task "Add a new random benchmark variant in the same style as mt19937 and xoshiro256." \
  --target-file "folly/test/RandomBenchmark.cpp" \
  --binary-name "random_benchmark" \
  --output random_retrieval.json
```

### 4) Generate the complete rewritten file

```bash
python scripts/04_generate_full_file.py \
  --task "Add a new random benchmark variant in the same style as mt19937 and xoshiro256." \
  --target-file "folly/test/RandomBenchmark.cpp" \
  --current-file "/myd/wdl/DCPerf/benchmarks/wdl_bench/wdl_sources/folly/folly/test/RandomBenchmark.cpp" \
  --retrieval-json random_retrieval.json \
  --output random_candidate_full.cpp
```

### 5) Rewrite in place, compile, and repair

```bash
python scripts/05_rewrite_compile_repair.py \
  --task "Add a new random benchmark variant in the same style as mt19937 and xoshiro256." \
  --target-file "folly/test/RandomBenchmark.cpp" \
  --binary-name "random_benchmark" \
  --retrieval-json random_retrieval.json \
  --rewritten-file random_candidate_full.cpp \
  --example-id random_addition_001
```

This writes a compile log under `outputs/compile_logs/` and, on success, stores the successful rewrite under `data/successful_edits/random_addition_001/`.

### 6) Build a supervised tuning dataset from successful edits

```bash
python scripts/06_build_tuning_dataset.py --output train.jsonl
```

This produces `data/tuning/train.jsonl` in a retrieval-conditioned full-file rewrite format suitable for later supervised fine-tuning. Google documents supervised fine-tuning for Gemini models, including Gemini 2.5 Pro, and provides a separate guide for preparing tuning data.

## Expected corpus contents

The corpus is designed to retrieve:
- the exact benchmark file,
- individual `BENCHMARK(...)` blocks,
- the matching CMake mapping line,
- similar benchmark blocks from sibling files,
- prior successful full-file rewrite exemplars.

This keeps RAG tightly focused on compilable Folly edits instead of broad repository summarization.

## Notes

- `scripts/05_rewrite_compile_repair.py` temporarily overwrites the target source file in place only for the duration of the build, then restores the original file.
- It expects the build tree to already exist.
- No external patch utility is required.
- If your SDK surface differs slightly, keep the same architecture but adjust the import or generation calls to your installed `google-cloud-aiplatform` version.
# folly_edit_rag
