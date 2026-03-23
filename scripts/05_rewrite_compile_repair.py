#!/usr/bin/env python3
from __future__ import annotations

import argparse
import grp
import json
import os
import pwd
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.settings import (
    CMAKE_BUILD_DIR,
    CMAKE_PARALLEL,
    COMPILE_LOGS_DIR,
    FOLLY_SRC_ROOT,
    REPAIRED_DIR,
    REPAIR_MODEL_NAME,
    RETRIEVALS_DIR,
    RUN_TIMEOUT_SEC,
    SUCCESSFUL_EDITS_DIR,
)
from benchmark_sources import resolve_build_timeout_sec
from llm_provider import create_text_model
from local_patch_utils import (
    build_compile_repair_guidance,
    build_registration_idiom_guidance,
    prepare_prompt_file_view,
    restore_prompt_file_omissions,
    trim_build_errors,
    validate_generated_source_output,
)

REPAIR_PROMPT = """You are repairing a rewritten Folly benchmark source file.

Rules:
- Return only the complete corrected source file contents for the target file.
- Preserve the requested benchmark intent.
- Minimize unrelated edits.
- Use only APIs visible in the current file.
- Keep the result compilable in the existing Folly/DCPerf build.
- Preserve includes, benchmark harness structure, and main() unless the build errors require a specific change.
- Do not include headings, copied prompt text, benchmark output, contexts, or diff markers.
- If the current file contains an omitted disabled-appendix placeholder, keep that placeholder unchanged in the returned file.

Required internal reasoning protocol before writing code:
- Separate diagnosis from repair.
- First identify whether the dominant problem is buildability or a remaining counter mismatch.
- If buildability is broken, fix the smallest source issue that addresses the actual compiler/build error first.
- If counter mismatch remains, treat the counter gap as a symptom, infer the likely subsystem cause, and choose one minimal transformation that addresses that cause.
- Preserve mechanisms that are already helping and do not damage metrics already near target.
- Do not include this reasoning in the output; return only the updated source file contents.

Task:
{task_summary}

Target file:
{target_file}

Local file idiom to preserve:
{registration_guidance}

Compiler/build guidance:
{error_guidance}

{broken_file_section}

Build errors:
{errors}

{contamination_note}

Return only the complete corrected source file contents for {target_file}.
"""


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None):
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def _optional_prompt_section(title: str, body: str, *, max_chars: int | None = None) -> str:
    body = str(body).strip()
    if not body:
        return ""
    if max_chars is not None and len(body) > max_chars:
        body = body[:max_chars].rstrip() + "\n..."
    return f"{title}:\n{body}\n\n"


def _compact_repair_errors(text: str) -> str:
    return trim_build_errors(text, max_lines=40, max_chars=2500)


def _compact_task_for_repair(task: str, *, max_chars: int = 1200) -> str:
    compact = " ".join(str(task).split())
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip() + "..."
    return compact


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


def _cmake_prefix_path_for_source(source_root: Path) -> str | None:
    wdl_root = source_root.resolve().parents[1]
    installed_root = wdl_root / "wdl_build" / "installed"
    build_root = wdl_root / "wdl_build" / "build"

    def _matching_dirs(root: Path, prefixes: list[str]) -> list[Path]:
        matches: list[Path] = []
        if not root.exists():
            return matches
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            if any(entry.name == prefix or entry.name.startswith(f"{prefix}-") for prefix in prefixes):
                matches.append(entry)
        return matches

    candidates = (
        _matching_dirs(installed_root, ["fmt", "folly", "fizz", "wangle", "mvfst", "liboqs", "glog", "fbthrift"])
        + _matching_dirs(build_root, ["fmt", "folly_rag", "folly", "fizz", "wangle", "mvfst", "liboqs", "glog"])
    )

    prefixes = [str(path) for path in candidates if path.exists()]
    if not prefixes:
        return None
    return ";".join(prefixes)


def _find_cmake_package_dir(wdl_root: Path, package_subpath: str) -> Path | None:
    installed_root = wdl_root / "wdl_build" / "installed"
    matches = sorted(installed_root.glob(f"*/lib/cmake/{package_subpath}"))
    if matches:
        return matches[0]
    direct = installed_root / package_subpath
    return direct if direct.exists() else None


def _ensure_generated_output_dirs(build_dir: Path, binary_name: str) -> list[str]:
    created: list[str] = []
    build_make_paths = sorted(build_dir.rglob(f"CMakeFiles/{binary_name}.dir/build.make"))
    for build_make in build_make_paths:
        try:
            for line in build_make.read_text(errors="replace").splitlines():
                stripped = line.strip()
                if "--gen " not in stripped or " -o " not in stripped:
                    continue
                try:
                    tokens = shlex.split(stripped)
                except ValueError:
                    continue
                for idx, token in enumerate(tokens[:-1]):
                    if token != "-o":
                        continue
                    output_dir = Path(tokens[idx + 1])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    created.append(str(output_dir))
        except OSError:
            continue
    return sorted(set(created))


def _missing_generated_outputs_for_target(build_dir: Path, target_name: str) -> list[Path]:
    missing: list[Path] = []
    clean_files = sorted(build_dir.rglob(f"CMakeFiles/{target_name}.dir/cmake_clean.cmake"))
    for clean_file in clean_files:
        base_dir = clean_file.parent.parent.parent
        try:
            for line in clean_file.read_text(errors="replace").splitlines():
                stripped = line.strip()
                if not stripped.startswith('"') or "gen-cpp2" not in stripped:
                    continue
                rel_path = stripped.strip('"')
                output_path = (base_dir / rel_path).resolve()
                if not output_path.exists():
                    missing.append(output_path)
        except OSError:
            continue
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in missing:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _bootstrap_fbthrift_generated_targets(
    build_dir: Path,
    source_root: Path,
    binary_name: str,
    timeout_sec: int,
) -> tuple[list[str], tuple[int, str, str] | None]:
    source_root_str = str(source_root.resolve())
    if binary_name != "ProtocolBench" or "fbthrift" not in source_root_str:
        return [], None

    bootstrap_targets = [
        "field_mask-cpp2-target",
        "protocol-cpp2-target",
        "protocolconformance-cpp2-target",
    ]
    logs: list[str] = []
    for target_name in bootstrap_targets:
        missing_outputs = _missing_generated_outputs_for_target(build_dir, target_name)
        if not missing_outputs:
            continue
        preview = ", ".join(str(path) for path in missing_outputs[:3])
        if len(missing_outputs) > 3:
            preview += ", ..."
        logs.append(
            f"[build-hygiene] bootstrapping missing generated thrift target {target_name}: {preview}"
        )
        rc, so, se = run(
            ["cmake", "--build", str(build_dir), "--parallel", str(CMAKE_PARALLEL), "--target", target_name],
            timeout=timeout_sec,
        )
        if so:
            logs.append(so)
        if rc != 0:
            return logs, (rc, so, se)
    return logs, None


def compile_binary(binary_name: str, build_dir: Path, source_root: Path) -> tuple[int, str, str, Path]:
    effective_build_dir = _preferred_build_dir(build_dir, source_root)
    stdout_prefix: list[str] = []
    cmake_prefix_path = _cmake_prefix_path_for_source(source_root)
    wdl_root = source_root.resolve().parents[1]
    timeout_sec = resolve_build_timeout_sec(binary_name, RUN_TIMEOUT_SEC)

    if effective_build_dir != build_dir.resolve():
        stdout_prefix.append(
            f"[build-hygiene] requested build dir {build_dir.resolve()} -> using {effective_build_dir}"
        )

    if not _build_dir_matches_source(effective_build_dir, source_root):
        reconfigure_cmd = ["cmake", "-S", str(source_root), "-B", str(effective_build_dir)]
        package_dirs = {
            "fmt_DIR": _find_cmake_package_dir(wdl_root, "fmt"),
            "folly_DIR": _find_cmake_package_dir(wdl_root, "folly"),
            "Folly_DIR": _find_cmake_package_dir(wdl_root, "folly"),
            "fizz_DIR": _find_cmake_package_dir(wdl_root, "fizz"),
            "Fizz_DIR": _find_cmake_package_dir(wdl_root, "fizz"),
            "wangle_DIR": _find_cmake_package_dir(wdl_root, "wangle"),
            "Wangle_DIR": _find_cmake_package_dir(wdl_root, "wangle"),
            "mvfst_DIR": _find_cmake_package_dir(wdl_root, "mvfst"),
            "liboqs_DIR": _find_cmake_package_dir(wdl_root, "liboqs"),
            "glog_DIR": _find_cmake_package_dir(wdl_root, "glog"),
            "fbthrift_DIR": _find_cmake_package_dir(wdl_root, "fbthrift"),
            "FBThrift_DIR": _find_cmake_package_dir(wdl_root, "fbthrift"),
        }
        for key, value in package_dirs.items():
            if value is not None:
                reconfigure_cmd.extend([f"-D{key}={value}"])
        if cmake_prefix_path:
            reconfigure_cmd.extend([f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}"])
        stdout_prefix.append(f"[build-hygiene] reconfiguring build dir: {' '.join(reconfigure_cmd)}")
        cfg_rc, cfg_so, cfg_se = run(reconfigure_cmd, timeout=timeout_sec)
        if cfg_so:
            stdout_prefix.append(cfg_so)
        if cfg_rc != 0:
            return cfg_rc, "\n".join(stdout_prefix), cfg_se, effective_build_dir

    created_dirs = _ensure_generated_output_dirs(effective_build_dir, binary_name)
    if created_dirs:
        stdout_prefix.append(
            "[build-hygiene] ensured generator output dirs: " + ", ".join(created_dirs)
        )

    bootstrap_logs, bootstrap_failure = _bootstrap_fbthrift_generated_targets(
        effective_build_dir,
        source_root,
        binary_name,
        timeout_sec,
    )
    stdout_prefix.extend(bootstrap_logs)
    if bootstrap_failure is not None:
        rc, so, se = bootstrap_failure
        if stdout_prefix:
            so = "\n".join(stdout_prefix + ([so] if so else []))
        return rc, so, se, effective_build_dir

    rc, so, se = run(
        ["cmake", "--build", str(effective_build_dir), "--parallel", str(CMAKE_PARALLEL), "--target", binary_name],
        timeout=timeout_sec,
    )
    if stdout_prefix:
        so = "\n".join(stdout_prefix + ([so] if so else []))
    return rc, so, se, effective_build_dir


def repair_file(
    model,
    task: str,
    target_file: str,
    original_file: str,
    broken_file: str,
    retrieved_contexts: list[dict],
    errors: str,
) -> str:
    compact_errors = _compact_repair_errors(errors)
    current_file_issue = validate_generated_source_output(
        broken_file,
        original_text=original_file,
    )
    prompt_file_view, omitted_blocks = prepare_prompt_file_view(broken_file)
    registration_guidance = (
        build_registration_idiom_guidance(original_file, target_file)
        or "Preserve the existing local benchmark registration idiom and callback storage shape."
    )
    error_guidance = build_compile_repair_guidance(
        errors,
        original_file,
        source_microbenchmark="",
    ) or "- No additional compile-specific guidance."
    prompt = REPAIR_PROMPT.format(
        task_summary=_compact_task_for_repair(task),
        target_file=target_file,
        registration_guidance=registration_guidance,
        error_guidance=error_guidance,
        broken_file_section=_optional_prompt_section(
            "Current broken rewritten source",
            prompt_file_view,
        ),
        errors=compact_errors,
        contamination_note=(
            f"Important: the current broken file already contains non-code prompt contamination ({current_file_issue}). "
            "Remove that contamination first, then fix the compile error.\n"
            if current_file_issue
            else ""
        ),
    )
    current_prompt = prompt
    last_candidate = broken_file
    for _ in range(2):
        print("[repair] prompt-begin", flush=True)
        print(current_prompt, flush=True)
        print("[repair] prompt-end", flush=True)
        print("[repair] waiting on Gemini repair inference", flush=True)
        started_at = time.monotonic()
        result = _strip_code_fences(model.generate_content(current_prompt).text).rstrip() + "\n"
        result = restore_prompt_file_omissions(result, omitted_blocks)
        elapsed_s = time.monotonic() - started_at
        print(f"[repair] Gemini repair inference returned after {elapsed_s:.1f}s", flush=True)
        reason = validate_generated_source_output(
            result,
            original_text=original_file,
        )
        if reason is None:
            return result
        print(f"[repair] rejected model response: {reason}", flush=True)
        last_candidate = result
        current_prompt = (
            f"Your previous response was invalid because: {reason}\n\n"
            f"Return only the complete corrected source file contents for {target_file}.\n"
            "Do not include prompt labels, contexts, benchmark output, diff markers, or duplicated main() definitions.\n\n"
            f"Current broken file:\n{prompt_file_view}\n\n"
            f"Build errors:\n{compact_errors}\n"
        )
    return last_candidate if last_candidate.strip() else broken_file


def _looks_like_local_build_issue(summary: str) -> bool:
    lowered = summary.lower()
    needles = [
        "could not find a package configuration file",
        "cmake error",
        "unable to open check cache file for write",
        "permission denied",
        "operation not permitted",
        "no such file or directory",
        "cmakefiles/cmakeoutput.log",
    ]
    return any(needle in lowered for needle in needles)


def _looks_like_missing_generated_header_issue(summary: str) -> bool:
    lowered = summary.lower()
    return (
        "no such file or directory" in lowered
        and "gen-cpp2" in lowered
        and ("fatal error:" in lowered or "#include" in lowered)
    )


def _first_nonempty_line(text: str) -> str:
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _path_identity(path: Path) -> str:
    try:
        st = path.stat()
    except OSError as exc:
        return f"{path} (stat failed: {type(exc).__name__}: {exc})"
    try:
        owner = pwd.getpwuid(st.st_uid).pw_name
    except KeyError:
        owner = str(st.st_uid)
    try:
        group = grp.getgrgid(st.st_gid).gr_name
    except KeyError:
        group = str(st.st_gid)
    return f"{path} owner={owner}:{group} mode={st.st_mode & 0o777:o}"


def _preflight_inplace_source_edit(src_file: Path) -> str | None:
    try:
        with open(src_file, "r+b"):
            pass
        probe_fd, probe_name = tempfile.mkstemp(
            prefix=f"{src_file.name}.writetest.",
            suffix=".tmp",
            dir=str(src_file.parent),
        )
        os.close(probe_fd)
        os.unlink(probe_name)
        return None
    except OSError as exc:
        return (
            "local source tree is not writable for in-place compilation: "
            f"{src_file} ({type(exc).__name__}: {exc}). "
            f"file={_path_identity(src_file)}; parent={_path_identity(src_file.parent)}. "
            "Fix ownership/permissions or switch this binary to an isolated per-attempt source tree/worktree."
        )


def _inplace_source_edit_error(src_file: Path, exc: OSError) -> str:
    return (
        "local source replacement failed during compile preparation: "
        f"{src_file} ({type(exc).__name__}: {exc}). "
        f"file={_path_identity(src_file)}; parent={_path_identity(src_file.parent)}. "
        "This is a local filesystem/ownership issue, not a model repair issue. "
        "Fix permissions or use an isolated per-attempt source tree/worktree."
    )


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
    (out_dir / "task.json").write_text(
        json.dumps(
            {
                "task": task,
                "target_file": target_file,
                "binary_name": binary_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
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
    model,
    task: str,
    target_file: str,
    binary_name: str,
    source_microbenchmark: str,
    src_file: Path,
    build_dir: Path,
    rewritten_text: str,
    original_text: str,
    retrieval_contexts: list[dict],
    artifact_dir: Path,
    child_key: str,
    max_compile_repair_attempts: int,
) -> dict:
    candidate_path = artifact_dir / f"{child_key}_candidate_full.cpp"
    candidate_path.write_text(rewritten_text, encoding="utf-8")
    record = {
        "task": task,
        "target_file": target_file,
        "binary_name": binary_name,
        "candidate_file": str(candidate_path),
        "requested_build_dir": str(build_dir.resolve()),
        "max_compile_repair_attempts": int(max(1, max_compile_repair_attempts)),
    }
    source_root = _infer_source_root(src_file, target_file)
    record["source_root"] = str(source_root)
    record["inplace_source_edit_strategy"] = "shared_source_tree"
    preflight_issue = _preflight_inplace_source_edit(src_file)
    if preflight_issue:
        print(f"[compile] {child_key}: {preflight_issue}", flush=True)
        record["preflight_live_source_writable"] = False
        record["preflight_failure"] = preflight_issue
        record["compile_attempts"] = []
        record["compile_attempt_count"] = 0
        record["repaired"] = False
        record["success"] = False
        record["reason"] = preflight_issue
        return record
    record["preflight_live_source_writable"] = True

    try:
        backup_fd, backup_name = tempfile.mkstemp(
            prefix=f"{src_file.name}.{child_key}.",
            suffix=".bak_folly_rag",
            dir=str(src_file.parent),
        )
        os.close(backup_fd)
        backup = Path(backup_name)
        shutil.copy2(src_file, backup)
    except OSError as exc:
        backup_issue = _inplace_source_edit_error(src_file, exc)
        print(f"[compile] {child_key}: {backup_issue}", flush=True)
        record["compile_attempts"] = []
        record["compile_attempt_count"] = 0
        record["repaired"] = False
        record["success"] = False
        record["reason"] = backup_issue
        return record
    try:
        total_attempts = int(max(1, max_compile_repair_attempts))
        current_text = rewritten_text
        current_candidate_path = candidate_path
        compile_attempts: list[dict[str, object]] = []
        final_text = rewritten_text
        final_path = artifact_dir / f"{child_key}_final.cpp"
        repaired = False
        repaired_path: Path | None = None
        success = False
        print(
            f"[compile] {child_key}: editing from nearest/source benchmark "
            f"{source_microbenchmark or '(unknown)'} "
            f"for target file {target_file} and build target {binary_name}",
            flush=True,
        )

        for attempt_index in range(1, total_attempts + 1):
            print(
                f"[compile] {child_key}: compile attempt {attempt_index}/{total_attempts} "
                f"for {binary_name} from {current_candidate_path.name}",
                flush=True,
            )
            try:
                src_file.write_text(current_text, encoding="utf-8")
                rc, so, se, effective_build_dir = compile_binary(binary_name, build_dir, source_root)
            except OSError as exc:
                replace_summary = _inplace_source_edit_error(src_file, exc)
                print(f"[compile] {child_key}: {replace_summary}", flush=True)
                compile_attempt = {
                    "attempt_index": attempt_index,
                    "candidate_file": str(current_candidate_path),
                    "compile_rc": None,
                    "compile_stdout": "",
                    "compile_stderr": replace_summary,
                    "effective_build_dir": str(build_dir.resolve()),
                }
                compile_attempts.append(compile_attempt)
                if attempt_index == 1:
                    record["first_compile_rc"] = None
                    record["first_compile_stdout"] = ""
                    record["first_compile_stderr"] = replace_summary
                elif attempt_index == 2:
                    record["second_compile_rc"] = None
                    record["second_compile_stdout"] = ""
                    record["second_compile_stderr"] = replace_summary
                record["effective_build_dir"] = str(build_dir.resolve())
                record["reason"] = replace_summary
                break
            record["effective_build_dir"] = str(effective_build_dir)
            if rc == 0:
                print(f"[compile] {child_key}: local build succeeded", flush=True)
            else:
                print(f"[compile] {child_key}: local build failed with rc={rc}", flush=True)
                failure_headline = _first_nonempty_line(f"{so}\n{se}")
                if failure_headline:
                    print(f"[compile] {child_key}: build issue: {failure_headline}", flush=True)
                if _looks_like_local_build_issue(f"{so}\n{se}"):
                    print(
                        "[compile] detected likely local build/configuration or permission issue; "
                        "repair inference may not help until the environment is fixed",
                        flush=True,
                    )
                if _looks_like_missing_generated_header_issue(f"{so}\n{se}"):
                    print(
                        "[compile] missing generated thrift headers in the active fbthrift build tree; "
                        "this is a build bootstrap issue, not a source repair issue",
                        flush=True,
                    )
            compile_attempt = {
                "attempt_index": attempt_index,
                "candidate_file": str(current_candidate_path),
                "compile_rc": rc,
                "compile_stdout": so,
                "compile_stderr": se,
                "effective_build_dir": str(effective_build_dir),
            }
            compile_attempts.append(compile_attempt)

            if attempt_index == 1:
                record["first_compile_rc"] = rc
                record["first_compile_stdout"] = so
                record["first_compile_stderr"] = se
            elif attempt_index == 2:
                record["second_compile_rc"] = rc
                record["second_compile_stdout"] = so
                record["second_compile_stderr"] = se

            if rc == 0:
                success = True
                final_text = current_text
                break

            if _looks_like_missing_generated_header_issue(f"{so}\n{se}"):
                record["reason"] = (
                    "missing generated thrift headers in the active fbthrift build tree; "
                    "stop repair and fix/build generator targets first"
                )
                break

            if attempt_index >= total_attempts:
                break

            print(
                f"[compile] {child_key}: compile attempt {attempt_index} failed with rc={rc}; requesting repair",
                flush=True,
            )
            fixed_text = repair_file(
                model,
                task,
                target_file,
                original_text,
                current_text,
                retrieval_contexts,
                so + "\n" + se,
            )
            repaired_path = artifact_dir / f"{child_key}_candidate_repair_attempt_{attempt_index + 1}_full.cpp"
            repaired_path.write_text(fixed_text, encoding="utf-8")
            current_text = fixed_text
            current_candidate_path = repaired_path
            repaired = True

        record["compile_attempts"] = compile_attempts
        record["compile_attempt_count"] = len(compile_attempts)
        record["repaired"] = repaired
        record["success"] = success
        if repaired_path is not None:
            record["repaired_file"] = str(repaired_path)

        if success:
            final_path.write_text(final_text, encoding="utf-8")
            record["final_file"] = str(final_path)
            print(f"[compile] {child_key}: compile succeeded", flush=True)
        else:
            print(
                f"[compile] {child_key}: compile failed after {len(compile_attempts)} attempt(s) "
                f"while adjusting {source_microbenchmark or '(unknown)'}",
                flush=True,
            )
    finally:
        if backup.exists():
            shutil.move(str(backup), str(src_file))

    return record


def run_single(args: argparse.Namespace, model) -> None:
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
        source_microbenchmark="",
        src_file=src_file,
        build_dir=build_dir,
        rewritten_text=rewritten_text,
        original_text=original_text,
        retrieval_contexts=retrieval_contexts,
        artifact_dir=artifact_dir,
        child_key="single",
        max_compile_repair_attempts=args.max_compile_repair_attempts,
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


def run_grouped(args: argparse.Namespace, model) -> None:
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
            source_microbenchmark=child_task.get("source_microbenchmark", ""),
            src_file=src_file,
            build_dir=build_dir,
            rewritten_text=rewritten_text,
            original_text=original_text,
            retrieval_contexts=retrieval_contexts,
            artifact_dir=artifact_dir,
            child_key=child_key,
            max_compile_repair_attempts=args.max_compile_repair_attempts,
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
    parser.add_argument(
        "--max-compile-repair-attempts",
        type=int,
        default=20,
        help="Maximum total compile attempts for one generated variant, including repair retries.",
    )
    parser.add_argument("--grouped-task-json", default="")
    parser.add_argument("--grouped-candidates-manifest", default="")
    args = parser.parse_args()

    model = create_text_model(REPAIR_MODEL_NAME)

    if args.grouped_task_json:
        run_grouped(args, model)
    else:
        run_single(args, model)


if __name__ == "__main__":
    main()
