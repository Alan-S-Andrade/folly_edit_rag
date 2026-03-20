#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import vertexai
from vertexai.generative_models import GenerativeModel

from config.settings import (
    CMAKE_BUILD_DIR,
    CMAKE_PARALLEL,
    COMPILE_LOGS_DIR,
    FOLLY_SRC_ROOT,
    LOCATION,
    PATCHES_DIR,
    PATCH_FILENAME,
    PROJECT_ID,
    REPAIRED_DIR,
    REPAIR_MODEL_NAME,
    RETRIEVALS_DIR,
    RUN_TIMEOUT_SEC,
    SUCCESSFUL_EDITS_DIR,
)
from local_patch_utils import (
    build_local_source_excerpt,
    normalize_generated_patch,
    render_retrieved_context,
    trim_build_errors,
)

REPAIR_PROMPT = """You are repairing a local unified diff patch for an existing Folly benchmark file.

Rules:
- Return only a unified diff patch.
- Keep the patch local to the provided source benchmark excerpt.
- Preserve the requested benchmark intent.
- Keep the original reference benchmark unchanged.
- If the current working benchmark anchor is still the original reference benchmark, keep exactly one added generated benchmark relative to it.
- If the current working benchmark anchor is already a carried-forward generated benchmark, repair the patch so it refines that carried-forward benchmark in place and does not add a second generated benchmark.
- Fix only what is required for the patch to apply cleanly and compile.
- Use only APIs visible in the local excerpt, retrieved context, or already present in the current file.
- Keep the patch compilable in the existing Folly/DCPerf build.

Task:
{task}

Original reference benchmark:
{reference_microbenchmark}

Current working benchmark anchor:
{source_microbenchmark}

Target file:
{target_file}

Local source excerpt from the current file:
{local_excerpt}

Retrieved context:
{retrieved}

Current patch:
{bad_patch}

Patch/apply/build errors:
{errors}

Return only a corrected unified diff patch.
"""


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _cmake_cache_home_directory(build_dir: Path) -> Path | None:
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.exists():
        return None
    try:
        for line in cache_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                raw = line.split("=", 1)[1].strip()
                if raw:
                    return Path(raw).resolve()
    except Exception:
        return None
    return None


def _build_dir_matches_source(build_dir: Path, source_root: Path) -> bool:
    home_dir = _cmake_cache_home_directory(build_dir)
    if home_dir is None:
        return False
    return home_dir == source_root.resolve()


def _preferred_build_dir(build_dir: Path, source_root: Path) -> Path:
    if build_dir.name.endswith("_rag"):
        candidates = [build_dir, build_dir.with_name(build_dir.name[:-4])]
    else:
        candidates = [build_dir.with_name(f"{build_dir.name}_rag"), build_dir]

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)

    for candidate in ordered:
        if candidate.exists() and _build_dir_matches_source(candidate, source_root):
            return candidate

    for candidate in ordered:
        if candidate.exists():
            return candidate

    return ordered[0]


def _infer_source_root(src_file: Path, target_file: str) -> Path:
    rel_path = Path(target_file)
    if rel_path.is_absolute():
        return rel_path
    try:
        return src_file.resolve().parents[len(rel_path.parts) - 1]
    except Exception:
        return FOLLY_SRC_ROOT


def compile_binary(binary_name: str, build_dir: Path, source_root: Path) -> tuple[int, str, str, Path]:
    effective_build_dir = _preferred_build_dir(build_dir, source_root)
    stdout_prefix: list[str] = []

    if effective_build_dir != build_dir.resolve():
        stdout_prefix.append(
            f"[build-hygiene] requested build dir {build_dir.resolve()} -> using {effective_build_dir}"
        )

    if not _build_dir_matches_source(effective_build_dir, source_root):
        reconfigure_cmd = ["cmake", "-S", str(source_root), "-B", str(effective_build_dir)]
        stdout_prefix.append(f"[build-hygiene] reconfiguring build dir: {' '.join(reconfigure_cmd)}")
        cfg_rc, cfg_so, cfg_se = run(reconfigure_cmd, timeout=RUN_TIMEOUT_SEC)
        if cfg_so:
            stdout_prefix.append(cfg_so)
        if cfg_rc != 0:
            return cfg_rc, "\n".join(stdout_prefix), cfg_se, effective_build_dir

    rc, so, se = run(
        ["cmake", "--build", str(effective_build_dir), "--parallel", str(CMAKE_PARALLEL), "--target", binary_name],
        timeout=RUN_TIMEOUT_SEC,
    )
    if stdout_prefix:
        so = "\n".join(stdout_prefix + ([so] if so else []))
    return rc, so, se, effective_build_dir


def try_apply_patch(src_file: Path, patch_path: Path, workdir: Path) -> tuple[bool, str, str, Path]:
    target_copy = workdir / src_file.name
    shutil.copy2(src_file, target_copy)
    rc, so, se = run(["patch", "-u", str(target_copy), str(patch_path)], cwd=workdir)
    return rc == 0, so, se, target_copy


def repair_patch(
    model: GenerativeModel,
    *,
    task: str,
    reference_microbenchmark: str,
    source_microbenchmark: str,
    target_file: str,
    local_excerpt: str,
    retrieved_contexts: list[dict],
    bad_patch: str,
    errors: str,
) -> str:
    retrieved = render_retrieved_context(
        retrieved_contexts,
        target_file=target_file,
        source_microbenchmark=source_microbenchmark,
    )
    prompt = REPAIR_PROMPT.format(
        task=task.strip(),
        reference_microbenchmark=reference_microbenchmark.strip() or source_microbenchmark.strip() or "(not provided)",
        source_microbenchmark=source_microbenchmark.strip() or "(not provided)",
        target_file=target_file,
        local_excerpt=local_excerpt,
        retrieved=retrieved,
        bad_patch=bad_patch,
        errors=errors,
    )
    return normalize_generated_patch(model.generate_content(prompt).text)


def save_success_example(
    example_id: str,
    task: str,
    target_file: str,
    binary_name: str,
    patch_text: str,
    final_file_text: str,
    retrieval_json: str,
) -> None:
    out_dir = SUCCESSFUL_EDITS_DIR / example_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "patch.diff").write_text(patch_text, encoding="utf-8")
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
    reference_microbenchmark: str,
    source_microbenchmark: str,
    target_file: str,
    binary_name: str,
    base_file: Path,
    live_src_file: Path,
    build_dir: Path,
    source_root: Path,
    patch_text: str,
    retrieval_contexts: list[dict],
    artifact_dir: Path,
    child_key: str,
    max_compile_repair_attempts: int,
) -> dict:
    normalized_patch = normalize_generated_patch(patch_text)
    initial_patch_path = artifact_dir / f"{child_key}_candidate_patch_attempt_1.diff"
    initial_patch_path.write_text(normalized_patch, encoding="utf-8")
    current_patch_text = normalized_patch
    current_patch_path = initial_patch_path

    original_text = load_text(base_file)
    local_excerpt = build_local_source_excerpt(original_text, source_microbenchmark or task)
    record: dict[str, object] = {
        "task": task,
        "reference_microbenchmark": reference_microbenchmark,
        "source_microbenchmark": source_microbenchmark,
        "target_file": target_file,
        "binary_name": binary_name,
        "patch_file": str(initial_patch_path),
        "current_file": str(base_file.resolve()),
        "live_source_file": str(live_src_file.resolve()),
        "requested_build_dir": str(build_dir.resolve()),
        "max_compile_repair_attempts": int(max(1, max_compile_repair_attempts)),
        "source_root": str(source_root.resolve()),
    }

    total_attempts = int(max(1, max_compile_repair_attempts))
    compile_attempts: list[dict[str, object]] = []
    repaired = False
    repaired_patch_path: Path | None = None
    final_text = ""
    final_path = artifact_dir / f"{child_key}_final.cpp"
    success = False

    for attempt_index in range(1, total_attempts + 1):
        print(
            f"[compile] {child_key}: patch attempt {attempt_index}/{total_attempts} for {binary_name} "
            f"from {current_patch_path.name}",
            flush=True,
        )

        with tempfile.TemporaryDirectory(prefix="folly_patch_") as td:
            workdir = Path(td)
            apply_ok, apply_so, apply_se, patched_copy = try_apply_patch(base_file, current_patch_path, workdir)
            apply_summary = trim_build_errors(
                f"{apply_so}\n{apply_se}",
                target_file=target_file,
            )
            attempt_record: dict[str, object] = {
                "attempt_index": attempt_index,
                "patch_file": str(current_patch_path),
                "patch_apply_success": apply_ok,
                "patch_apply_stdout": apply_so,
                "patch_apply_stderr": apply_se,
                "patch_apply_summary": apply_summary,
            }
            compile_attempts.append(attempt_record)

            if attempt_index == 1:
                record["first_patch_apply_success"] = apply_ok
                record["first_patch_apply_stdout"] = apply_so
                record["first_patch_apply_stderr"] = apply_se
                record["first_patch_apply_summary"] = apply_summary
            elif attempt_index == 2:
                record["second_patch_apply_success"] = apply_ok
                record["second_patch_apply_stdout"] = apply_so
                record["second_patch_apply_stderr"] = apply_se
                record["second_patch_apply_summary"] = apply_summary

            if not apply_ok:
                if attempt_index >= total_attempts:
                    break
                repaired_patch_path = artifact_dir / f"{child_key}_candidate_patch_attempt_{attempt_index + 1}.diff"
                current_patch_text = repair_patch(
                    model,
                    task=task,
                    reference_microbenchmark=reference_microbenchmark,
                    source_microbenchmark=source_microbenchmark,
                    target_file=target_file,
                    local_excerpt=local_excerpt,
                    retrieved_contexts=retrieval_contexts,
                    bad_patch=current_patch_text,
                    errors=apply_summary or "Patch failed to apply.",
                )
                repaired_patch_path.write_text(current_patch_text, encoding="utf-8")
                current_patch_path = repaired_patch_path
                repaired = True
                continue

            backup = live_src_file.with_suffix(live_src_file.suffix + f".{child_key}.bak_folly_rag")
            shutil.copy2(live_src_file, backup)
            try:
                shutil.copy2(patched_copy, live_src_file)
                compile_rc, compile_so, compile_se, effective_build_dir = compile_binary(
                    binary_name,
                    build_dir,
                    source_root,
                )
            finally:
                shutil.move(str(backup), str(live_src_file))

            compile_summary = trim_build_errors(
                f"{compile_so}\n{compile_se}",
                target_file=target_file,
            )
            attempt_record["compile_rc"] = compile_rc
            attempt_record["compile_stdout"] = compile_so
            attempt_record["compile_stderr"] = compile_se
            attempt_record["compile_summary"] = compile_summary
            attempt_record["effective_build_dir"] = str(effective_build_dir)
            record["effective_build_dir"] = str(effective_build_dir)

            if attempt_index == 1:
                record["first_compile_rc"] = compile_rc
                record["first_compile_stdout"] = compile_so
                record["first_compile_stderr"] = compile_se
                record["first_compile_summary"] = compile_summary
            elif attempt_index == 2:
                record["second_compile_rc"] = compile_rc
                record["second_compile_stdout"] = compile_so
                record["second_compile_stderr"] = compile_se
                record["second_compile_summary"] = compile_summary

            if compile_rc == 0:
                success = True
                final_text = load_text(patched_copy)
                final_path.write_text(final_text, encoding="utf-8")
                break

            if attempt_index >= total_attempts:
                break

            repaired_patch_path = artifact_dir / f"{child_key}_candidate_patch_attempt_{attempt_index + 1}.diff"
            current_patch_text = repair_patch(
                model,
                task=task,
                reference_microbenchmark=reference_microbenchmark,
                source_microbenchmark=source_microbenchmark,
                target_file=target_file,
                local_excerpt=local_excerpt,
                retrieved_contexts=retrieval_contexts,
                bad_patch=current_patch_text,
                errors=compile_summary or "Build failed after patch applied.",
            )
            repaired_patch_path.write_text(current_patch_text, encoding="utf-8")
            current_patch_path = repaired_patch_path
            repaired = True

    record["compile_attempts"] = compile_attempts
    record["compile_attempt_count"] = len(compile_attempts)
    record["repaired"] = repaired
    record["success"] = success
    if repaired_patch_path is not None:
        record["repaired_patch_file"] = str(repaired_patch_path)
    if success:
        record["final_patch_file"] = str(current_patch_path)
        record["final_file"] = str(final_path)
        print(f"[compile] {child_key}: compile succeeded", flush=True)
    else:
        print(f"[compile] {child_key}: compile failed after {len(compile_attempts)} attempt(s)", flush=True)

    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--target-file", required=True, help="Relative path under wdl_sources/folly")
    parser.add_argument("--binary-name", required=True)
    parser.add_argument(
        "--current-file",
        default="",
        help="Absolute path to the patch base file for this attempt. Falls back to FOLLY_SRC_ROOT/target-file.",
    )
    parser.add_argument("--build-dir", default=str(CMAKE_BUILD_DIR))
    parser.add_argument(
        "--source-root",
        default="",
        help="Absolute path to the real source tree root used for compilation.",
    )
    parser.add_argument("--reference-microbenchmark", default="")
    parser.add_argument("--source-microbenchmark", default="")
    parser.add_argument("--retrieval-json", default="sample_retrieval.json")
    parser.add_argument("--patch-file", default=PATCH_FILENAME)
    parser.add_argument("--example-id", default="latest_success")
    parser.add_argument("--no-save-success-example", action="store_true")
    parser.add_argument(
        "--max-compile-repair-attempts",
        type=int,
        default=2,
        help="Maximum total patch/apply/build attempts for one generated variant, including repair retries.",
    )
    args = parser.parse_args()

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(REPAIR_MODEL_NAME)

    base_file = Path(args.current_file) if args.current_file else (FOLLY_SRC_ROOT / args.target_file)
    build_dir = Path(args.build_dir)
    source_root = (
        Path(args.source_root).resolve()
        if args.source_root
        else _infer_source_root(FOLLY_SRC_ROOT / args.target_file, args.target_file).resolve()
    )
    live_src_file = (
        Path(args.target_file).resolve()
        if Path(args.target_file).is_absolute()
        else (source_root / args.target_file).resolve()
    )
    patch_path = Path(args.patch_file)
    if not patch_path.is_absolute():
        patch_path = PATCHES_DIR / args.patch_file
    if not patch_path.exists():
        patch_path = Path(args.patch_file)
    if not base_file.exists():
        raise FileNotFoundError(f"Patch base file not found: {base_file}")
    if not live_src_file.exists():
        raise FileNotFoundError(f"Live source file not found: {live_src_file}")
    retrieval_contexts = json.loads((RETRIEVALS_DIR / args.retrieval_json).read_text())
    patch_text = load_text(patch_path)

    artifact_dir = REPAIRED_DIR / args.example_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    record = compile_candidate(
        model=model,
        task=args.task,
        reference_microbenchmark=args.reference_microbenchmark,
        source_microbenchmark=args.source_microbenchmark,
        target_file=args.target_file,
        binary_name=args.binary_name,
        base_file=base_file,
        live_src_file=live_src_file,
        build_dir=build_dir,
        source_root=source_root,
        patch_text=patch_text,
        retrieval_contexts=retrieval_contexts,
        artifact_dir=artifact_dir,
        child_key="single",
        max_compile_repair_attempts=args.max_compile_repair_attempts,
    )

    if record.get("success") and not args.no_save_success_example:
        final_text = load_text(Path(str(record["final_file"])))
        final_patch_text = load_text(Path(str(record["final_patch_file"])))
        save_success_example(
            args.example_id,
            args.task,
            args.target_file,
            args.binary_name,
            final_patch_text,
            final_text,
            args.retrieval_json,
        )

    out_path = COMPILE_LOGS_DIR / f"{args.example_id}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)
    print(f"Wrote compile log to {out_path}", flush=True)


if __name__ == "__main__":
    main()
