from pathlib import Path
import os

PROJECT_ID = os.environ.get('VERTEX_PROJECT_ID', 'your-project-id')
LOCATION = os.environ.get('VERTEX_LOCATION', 'us-central1')
MODEL_NAME = os.environ.get('VERTEX_MODEL_NAME', 'gemini-2.5-pro')
EMBEDDING_MODEL = os.environ.get(
    'VERTEX_EMBEDDING_MODEL',
    'publishers/google/models/text-embedding-005',
)

# Point this at your DCPerf checkout root.
DC_PERF_ROOT = Path(os.environ.get('DC_PERF_ROOT', '/myd/wdl/DCPerf'))
WDL_BENCH_ROOT = DC_PERF_ROOT / 'benchmarks' / 'wdl_bench'
FOLLY_SRC_ROOT = WDL_BENCH_ROOT / 'wdl_sources' / 'folly'
FOLLY_TEST_ROOT = FOLLY_SRC_ROOT / 'folly' / 'test'
FOLLY_CMAKE = FOLLY_SRC_ROOT / 'CMakeLists.txt'
DEFAULT_BUILD_DIR = WDL_BENCH_ROOT / 'wdl_build' / 'build' / 'folly'

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / 'data'
RAG_DOCS_DIR = DATA_DIR / 'rag_docs'
MANIFESTS_DIR = DATA_DIR / 'manifests'
TUNING_DIR = DATA_DIR / 'tuning'

OUTPUTS_DIR = REPO_ROOT / 'outputs'
RETRIEVALS_DIR = OUTPUTS_DIR / 'retrievals'
PATCHES_DIR = OUTPUTS_DIR / 'patches'
REPAIRED_DIR = OUTPUTS_DIR / 'repaired'
COMPILE_LOGS_DIR = OUTPUTS_DIR / 'compile_logs'
RUNS_DIR = OUTPUTS_DIR / 'runs'

SUCCESSFUL_EDITS_DIR = DATA_DIR / 'successful_edits'

for path in [
    DATA_DIR,
    RAG_DOCS_DIR,
    MANIFESTS_DIR,
    TUNING_DIR,
    OUTPUTS_DIR,
    RETRIEVALS_DIR,
    PATCHES_DIR,
    REPAIRED_DIR,
    COMPILE_LOGS_DIR,
    RUNS_DIR,
    SUCCESSFUL_EDITS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

PATCH_FILENAME = 'candidate.patch'
PATCHED_FILENAME = 'patched_target.cpp'
REPAIRED_PATCH_FILENAME = 'candidate_repair.patch'
MANIFEST_FILENAME = 'folly_rag_manifest.json'
CORPUS_INFO_FILENAME = 'corpus_info.json'

TOP_K = int(os.environ.get('RAG_TOP_K', '8'))
RETRIEVAL_RERANK_MODEL = os.environ.get('RAG_RERANK_MODEL', MODEL_NAME)

# Build defaults. Override for your setup if needed.
CMAKE_BUILD_DIR = Path(os.environ.get('CMAKE_BUILD_DIR', str(DEFAULT_BUILD_DIR)))
CMAKE_PARALLEL = os.environ.get('CMAKE_PARALLEL', '16')
RUN_TIMEOUT_SEC = int(os.environ.get('RUN_TIMEOUT_SEC', '120'))

# Safety guard: local-file upload limit documented by Vertex AI RAG Engine.
MAX_LOCAL_UPLOAD_BYTES = 25 * 1024 * 1024
