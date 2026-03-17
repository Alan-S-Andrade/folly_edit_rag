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

The prepared corpus now includes:
- Folly benchmark sources and Folly `CMakeLists.txt` mappings
- fbthrift benchmark sources for `ProtocolBench` and `VarintUtilsBench`
- successful edit exemplars, when available

### 2) Create the Vertex RAG corpus and upload local docs

```bash
python scripts/02_create_corpus_and_upload.py
```

If this fails with a project permission error, set a real Google Cloud project first:

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default set-quota-project YOUR_PROJECT_ID

export VERTEX_PROJECT_ID=YOUR_PROJECT_ID
export VERTEX_LOCATION=us-central1
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

## Property-edit task pipeline

After step 2, generate the task manifest from `search_engine/all_features.xlsx`:

```bash
python scripts/07_generate_property_edit_tasks.py generate \
  --output data/property_edit_tasks/property_edit_tasks.jsonl
```

This creates one task row per:
- source microbenchmark
- target feature
- direction: `increase` or `decrease`
- magnitude: `minimally`, `moderately`, `substantially`, `significantly`

The default target feature set is exactly:

- `block_size_P50`
- `family::arith_P50`
- `family::branch_P50`
- `family::load_P50`
- `family::logic_P50`
- `family::mem_P50`
- `family::move_P50`
- `family::other_P50`
- `family::store_P50`
- `jumps_P50`
- `not_taken_runs_P50`
- `raw_distances_P50`
- `taken_runs_P50`
- `IPC`
- `L1-dcache-load-misses_MPKI`
- `L1-dcache-loads_MPKI`
- `L1-dcache-stores_MPKI`
- `L1-icache-load-misses_MPKI`
- `L2-icache-load-misses_MPKI`
- `LLC-load-misses_MPKI`
- `branch-load-misses_MPKI`
- `branch-misses_MPKI`
- `iTLB-load-misses_MPKI`
- `TMA_Backend_Bound(%)`
- `TMA_Bad_Speculation(%)`
- `TMA_Frontend_Bound(%)`
- `TMA_Retiring(%)`

The `_P50` features come from the Intel PT path:
- `perf record`
- PT disassembly
- `frequencies.py`

After `generate`, the normal next step is to run a small filtered batch through retrieval, rewrite, compile/repair, and feature evaluation:

```bash
python scripts/07_generate_property_edit_tasks.py run \
  --manifest data/property_edit_tasks/property_edit_tasks.jsonl \
  --binary random_benchmark \
  --limit 10
```

`--limit 10` means "run only the first 10 matching task rows from the manifest." It is only a batch-size cap so you can test the pipeline without launching the full dataset.

Useful first runs:

```bash
python scripts/07_generate_property_edit_tasks.py run \
  --manifest data/property_edit_tasks/property_edit_tasks.jsonl \
  --binary random_benchmark \
  --limit 10
```

```bash
python scripts/07_generate_property_edit_tasks.py run \
  --manifest data/property_edit_tasks/property_edit_tasks.jsonl \
  --binary ProtocolBench \
  --limit 10
```

```bash
python scripts/07_generate_property_edit_tasks.py run \
  --manifest data/property_edit_tasks/property_edit_tasks.jsonl \
  --binary VarintUtilsBench \
  --limit 10
```

If you want a smaller manifest before running, generate one per binary or feature:

```bash
python scripts/07_generate_property_edit_tasks.py generate \
  --binary random_benchmark \
  --feature-name perf_stat.L1-icache-load-misses_MPKI \
  --output data/property_edit_tasks/random_icache_tasks.jsonl
```

Then run that smaller manifest:

```bash
python scripts/07_generate_property_edit_tasks.py run \
  --manifest data/property_edit_tasks/random_icache_tasks.jsonl \
  --limit 10
```

The `run` command marks an edit as successful only if:
- the edited source compiles
- a new microbenchmark appears in `--bm_list`
- post-build feature extraction is run on the new candidate
- the extracted feature value for the target metric moves in the requested direction
- the observed delta lands in the requested magnitude band

This means success labels are not inferred from the prompt alone. Every candidate is tagged by actually compiling it, extracting features from the built binary, and checking the requested metric delta against the baseline microbenchmark.

After enough successful runs accumulate under `data/successful_edits/`, export the supervised tuning dataset:

```bash
python scripts/06_build_tuning_dataset.py --output train.jsonl
```

## Live cloud flow

From the repository root, you can run the live cloud path with one command:

```bash
cd /myd/wdl/BenchmarkReplication
./run_cloud_tuning_e2e.sh \
  --output-dir /myd/wdl/BenchmarkReplication/outputs/cloud_demo \
  --benchmark-name bench_ilp_cloud \
  --cloud-total-seconds 10 \
  --cloud-workload-warmup-seconds 2 \
  --cloud-segment-seconds 3 \
  -- ./bench_ilp
```

This wrapper will:
- rebuild the canonical ScaNN index
- start local `adjustor.py` and `input_to_search_delta.py` services
- run `run_feature_extraction.py` in `--cloud-mode`
- run `cloud_segment_frontend.py`
- write the final replacement plan JSON

The main outputs are:
- `<output-dir>/<benchmark-name>_feature_extraction.json`
- `<output-dir>/<benchmark-name>_segment_replacements.json`

`<benchmark-name>_segment_replacements.json` now includes an `accuracy_report` section with:
- mean per-segment accuracy
- mean per-segment normalized absolute error
- mean-profile accuracy comparing the average chosen benchmark profile against the average extracted segment profile
- axis breakdowns for `overall`, `IPC`, and `branch_cache_tlb_mpki`

Before running it:
- export `VERTEX_PROJECT_ID`
- run `sudo -v` so the perf-based collectors and evaluation path do not block on a password prompt

## Notes

- `scripts/05_rewrite_compile_repair.py` temporarily overwrites the target source file in place only for the duration of the build, then restores the original file.
- It expects the build tree to already exist.
- No external patch utility is required.
- If your SDK surface differs slightly, keep the same architecture but adjust the import or generation calls to your installed `google-cloud-aiplatform` version.
# folly_edit_rag
