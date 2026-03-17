#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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
    PROJECT_ID,
    REPAIRED_DIR,
    RETRIEVALS_DIR,
    RUN_TIMEOUT_SEC,
    SUCCESSFUL_EDITS_DIR,
)

REPAIR_PROMPT = """You are repairing a rewritten Folly benchmark source file.

Rules:
- Return only the complete corrected source file contents for the target file.
- Preserve the requested benchmark intent.
- Minimize unrelated edits.
- Use only APIs visible in the current file or retrieved context.
- Keep the result compilable in the existing Folly/DCPerf build.
- Preserve includes, benchmark harness structure, and main() unless the build errors require a specific change.

Task:
{task}

Target file:
{target_file}

Original source before rewrite:
{original_file}

Current broken rewritten source:
{broken_file}

Retrieved context:
{retrieved}

Build errors:
{errors}

Return only the complete corrected source file contents for {target_file}.
"""


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None):
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def compile_binary(binary_name: str, build_dir: Path) -> tuple[int, str, str]:
    return run(
        ["cmake", "--build", str(build_dir), "--parallel", str(CMAKE_PARALLEL), "--target", binary_name],
        timeout=RUN_TIMEOUT_SEC,
    )


def repair_file(
    model: GenerativeModel,
    task: str,
    target_file: str,
    original_file: str,
    broken_file: str,
    retrieved_contexts: list[dict],
    errors: str,
) -> str:
    retrieved = "\n\n".join(
        f"[CONTEXT {i + 1}]\n{c['text']}" for i, c in enumerate(retrieved_contexts)
    )
    prompt = REPAIR_PROMPT.format(
        task=task.strip(),
        target_file=target_file,
        original_file=original_file,
        broken_file=broken_file,
        retrieved=retrieved,
        errors=errors,
    )
    return model.generate_content(prompt).text


def save_success_example(
    example_id: str,
    task: str,
    target_file: str,
    binary_name: str,
    original_text: str,
    final_file_text: str,
    retrieval_json: str,
) -> None:
    out_dir = SUCCESSFUL_EDITS_DIR / example_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "original.cpp").write_text(original_text, encoding="utf-8")
    (out_dir / "final.cpp").write_text(final_file_text, encoding="utf-8")
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task": task,
                "target_file": target_file,
                "binary_name": binary_name,
                "compile_success": True,
                "retrieval_json": retrieval_json,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def compile_candidate(
    *,
    model: GenerativeModel,
    task: str,
    target_file: str,
    binary_name: str,
    src_file: Path,
    build_dir: Path,
    rewritten_text: str,
    original_text: str,
    retrieval_contexts: list[dict],
    artifact_dir: Path,
    child_key: str,
) -> dict:
    candidate_path = artifact_dir / f"{child_key}_candidate_full.cpp"
    candidate_path.write_text(rewritten_text, encoding="utf-8")
    record = {
        "task": task,
        "target_file": target_file,
        "binary_name": binary_name,
        "candidate_file": str(candidate_path),
    }

    backup = src_file.with_suffix(src_file.suffix + f".{child_key}.bak_folly_rag")
    shutil.copy2(src_file, backup)
    try:
        print(
            f"[compile] {child_key}: compiling {binary_name} from {candidate_path.name}",
            flush=True,
        )
        src_file.write_text(rewritten_text, encoding="utf-8")
        rc, bso, bse = compile_binary(binary_name, build_dir)
        record["first_compile_rc"] = rc
        record["first_compile_stdout"] = bso
        record["first_compile_stderr"] = bse

        final_text = rewritten_text
        final_path = artifact_dir / f"{child_key}_final.cpp"
        if rc != 0:
            print(
                f"[compile] {child_key}: initial compile failed with rc={rc}; requesting repair",
                flush=True,
            )
            fixed_text = repair_file(
                model,
                task,
                target_file,
                original_text,
                rewritten_text,
                retrieval_contexts,
                bso + "\n" + bse,
            )
            repaired_path = artifact_dir / f"{child_key}_candidate_repair_full.cpp"
            repaired_path.write_text(fixed_text, encoding="utf-8")
            src_file.write_text(fixed_text, encoding="utf-8")
            print(
                f"[compile] {child_key}: compiling repaired candidate {repaired_path.name}",
                flush=True,
            )
            rc2, so2, se2 = compile_binary(binary_name, build_dir)
            record["second_compile_rc"] = rc2
            record["second_compile_stdout"] = so2
            record["second_compile_stderr"] = se2
            record["repaired"] = True
            record["success"] = rc2 == 0
            record["repaired_file"] = str(repaired_path)
            final_text = fixed_text
        else:
            record["repaired"] = False
            record["success"] = True

        if record["success"]:
            final_path.write_text(final_text, encoding="utf-8")
            record["final_file"] = str(final_path)
            print(f"[compile] {child_key}: compile succeeded", flush=True)
        else:
            print(f"[compile] {child_key}: compile failed after repair", flush=True)
    finally:
        shutil.move(str(backup), str(src_file))

    return record


def run_single(args: argparse.Namespace, model: GenerativeModel) -> None:
    src_file = Path(args.current_file) if args.current_file else (FOLLY_SRC_ROOT / args.target_file)
    build_dir = Path(args.build_dir)
    original_text = src_file.read_text(encoding="utf-8", errors="ignore")
    rewritten_path = Path(args.rewritten_file)
    if not rewritten_path.is_absolute():
        rewritten_path = REPAIRED_DIR / rewritten_path
    retrieval_contexts = json.loads((RETRIEVALS_DIR / args.retrieval_json).read_text())
    rewritten_text = rewritten_path.read_text(encoding="utf-8", errors="ignore")
    artifact_dir = REPAIRED_DIR / args.example_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    record = compile_candidate(
        model=model,
        task=args.task,
        target_file=args.target_file,
        binary_name=args.binary_name,
        src_file=src_file,
        build_dir=build_dir,
        rewritten_text=rewritten_text,
        original_text=original_text,
        retrieval_contexts=retrieval_contexts,
        artifact_dir=artifact_dir,
        child_key="single",
    )
    if record.get("success") and not args.no_save_success_example:
        final_text = Path(record["final_file"]).read_text(encoding="utf-8", errors="ignore")
        save_success_example(
            args.example_id,
            args.task,
            args.target_file,
            args.binary_name,
            original_text,
            final_text,
            args.retrieval_json,
        )
        (REPAIRED_DIR / "final_applied.cpp").write_text(final_text, encoding="utf-8")

    out_path = COMPILE_LOGS_DIR / f"{args.example_id}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    print(f"Wrote compile log to {out_path}", flush=True)


def run_grouped(args: argparse.Namespace, model: GenerativeModel) -> None:
    grouped_task = json.loads(Path(args.grouped_task_json).read_text(encoding="utf-8"))
    candidate_manifest = json.loads((REPAIRED_DIR / args.grouped_candidates_manifest).read_text(encoding="utf-8"))
    retrieval_contexts = json.loads((RETRIEVALS_DIR / args.retrieval_json).read_text())
    src_file = Path(args.current_file) if args.current_file else (FOLLY_SRC_ROOT / args.target_file)
    build_dir = Path(args.build_dir)
    original_text = src_file.read_text(encoding="utf-8", errors="ignore")

    artifact_dir = REPAIRED_DIR / args.example_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    grouped_record: dict[str, object] = {
        "grouped_job_id": grouped_task["grouped_job_id"],
        "target_file": grouped_task["target_file"],
        "binary_name": grouped_task["binary"],
        "children": {},
    }

    for child_key, child_task in grouped_task["child_tasks"].items():
        print(f"[compile] grouped child {child_key}: start", flush=True)
        candidate_info = candidate_manifest["candidates"].get(child_key)
        if not candidate_info:
            grouped_record["children"][child_key] = {
                "success": False,
                "reason": "Missing generated candidate for child target.",
            }
            print(f"[compile] grouped child {child_key}: missing candidate", flush=True)
            continue

        rewritten_path = Path(candidate_info["candidate_file"])
        rewritten_text = rewritten_path.read_text(encoding="utf-8", errors="ignore")
        child_record = compile_candidate(
            model=model,
            task=child_task["task"],
            target_file=grouped_task["target_file"],
            binary_name=grouped_task["binary"],
            src_file=src_file,
            build_dir=build_dir,
            rewritten_text=rewritten_text,
            original_text=original_text,
            retrieval_contexts=retrieval_contexts,
            artifact_dir=artifact_dir,
            child_key=child_key,
        )
        child_record["expected_suffix"] = child_task["expected_suffix"]
        child_record["task_id"] = child_task["task_id"]
        grouped_record["children"][child_key] = child_record
        status = "success" if child_record.get("success") else "failed"
        print(f"[compile] grouped child {child_key}: {status}", flush=True)

    out_path = COMPILE_LOGS_DIR / f"{args.example_id}.json"
    out_path.write_text(json.dumps(grouped_record, indent=2), encoding="utf-8")
    print(json.dumps(grouped_record, indent=2), flush=True)
    print(f"Wrote grouped compile log to {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="")
    parser.add_argument("--target-file", required=True, help="Relative path under wdl_sources/folly")
    parser.add_argument("--binary-name", required=True)
    parser.add_argument("--current-file", default="", help="Absolute path to the target source file. Falls back to FOLLY_SRC_ROOT/target-file.")
    parser.add_argument("--build-dir", default=str(CMAKE_BUILD_DIR), help="CMake build directory that owns the requested target.")
    parser.add_argument("--retrieval-json", default="sample_retrieval.json")
    parser.add_argument("--rewritten-file", default="candidate_full.cpp")
    parser.add_argument("--example-id", default="latest_success")
    parser.add_argument("--no-save-success-example", action="store_true")
    parser.add_argument("--grouped-task-json", default="")
    parser.add_argument("--grouped-candidates-manifest", default="")
    args = parser.parse_args()

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)

    if args.grouped_task_json:
        run_grouped(args, model)
    else:
        run_single(args, model)


if __name__ == "__main__":
    main()
