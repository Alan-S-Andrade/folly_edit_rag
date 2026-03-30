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
    PATCHES_DIR,
    PATCH_FILENAME,
    REPAIRED_DIR,
    REPAIR_MODEL_NAME,
    RETRIEVALS_DIR,
    RUN_TIMEOUT_SEC,
    SUCCESSFUL_EDITS_DIR,
)
from benchmark_sources import resolve_build_timeout_sec
from llm_provider import create_text_model
from local_patch_utils import (
    build_unified_diff,
    build_local_source_excerpt,
    build_registration_idiom_guidance,
    build_compile_repair_guidance,
    normalize_generated_file,
    normalize_generated_patch,
    prepare_prompt_file_view,
    render_retrieved_context,
    repair_truncated_tail,
    restore_prompt_file_omissions,
    trim_build_errors,
    validate_generated_source_output,
)

PATCH_REPAIR_PROMPT = """You are repairing a generated source-file rewrite for an existing C++ benchmark file.

Rules:
- Return only the complete updated source file contents for the target file.
- The first non-whitespace characters of your response must already be source code from the target file, not an English sentence.
- Do not include markdown fences, prose, or extra text before/after the file contents.
- Preserve the requested benchmark intent.
- Keep the original reference benchmark unchanged.
- If the current working benchmark anchor is still the original reference benchmark, keep exactly one added generated benchmark relative to it.
- If the current working benchmark anchor is already a carried-forward generated benchmark, repair the patch so it refines that carried-forward benchmark in place and does not add a second generated benchmark.
- Do not collapse the generated benchmark into a rename-only clone, registration-only alias, or bare wrapper around identical work.
- Fix only what is required for the file rewrite to compile and correspond to a local patch.
- Use only APIs already present in the current file.
- Keep the patch compilable in the existing Folly/DCPerf build.
- Do not return a diff. Return the full updated file contents only.
- Do not include headings, copied prompt text, benchmark output, or labels such as "Task:", "Retrieved context:", "Local source excerpt", or "Patch/apply/build errors:".
- If the current file contains an omitted disabled-appendix placeholder, keep that placeholder unchanged in the returned file.

Intent to preserve:
{task_summary}

Original reference benchmark:
{reference_microbenchmark}

Current working benchmark anchor:
{source_microbenchmark}

Target file:
{target_file}

Local file idiom to preserve:
{registration_guidance}

Repair guidance:
{error_guidance}

Current target file contents:
{current_file}

Current patch attempt:
{bad_patch}

Patch apply errors:
{errors}

Return only the complete updated source file contents for {target_file}.
"""


COMPILE_REPAIR_PROMPT = """You are repairing a generated C++ benchmark source file after a compile failure.

Rules:
- Return only the complete updated source file contents for the target file.
- The first non-whitespace characters of your response must already be source code from the target file, not an English sentence.
- Do not include markdown fences, prose, or extra text before/after the file contents.
- Fix only the smallest source issue needed for this file to compile.
- Preserve the requested benchmark intent and benchmark registration shape.
- Keep the original reference benchmark unchanged.
- Keep exactly one added generated benchmark relative to the current working benchmark anchor.
- Do not collapse the generated benchmark into a rename-only clone, registration-only alias, or bare wrapper around identical work.
- Use only APIs already present in the current file.
- Do not include headings, copied prompt text, benchmark output, excerpts, contexts, or diff markers.
- Do not include strings such as "Task:", "Retrieved context:", "Local source excerpt", or "Patch/apply/build errors:" in the file.
- If the current file contains an omitted disabled-appendix placeholder, keep that placeholder unchanged in the returned file.

Intent to preserve:
{task_summary}

Original reference benchmark:
{reference_microbenchmark}

Current working benchmark anchor:
{source_microbenchmark}

Target file:
{target_file}

Local file idiom to preserve:
{registration_guidance}

Compiler/build guidance:
{error_guidance}

Current broken file:
{current_file}

Compiler/build errors:
{errors}

{contamination_note}

Return only the complete updated source file contents for {target_file}.
"""


INVALID_RESPONSE_REPAIR_PROMPT = """Your previous response was invalid because: {reason}

Return only the complete updated source file contents for {target_file}.
Do not include any headings, prompt labels, benchmark output, diff markers, or duplicated main() definitions.
Start immediately with source code from {target_file}; do not preface the file with any explanation.

Current file:
{current_file}

Errors:
{errors}
"""


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


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


def _generate_validated_repair_file(
    model,
    *,
    prompt: str,
    target_file: str,
    current_file_text: str,
    prompt_file_text: str,
    original_file_text: str,
    errors: str,
    omitted_blocks: dict[str, str],
) -> str:
    last_candidate = current_file_text
    current_prompt = prompt
    for inference_attempt in range(1, 3):
        print("[repair] prompt-begin", flush=True)
        print(current_prompt, flush=True)
        print("[repair] prompt-end", flush=True)
        print("[repair] waiting on Gemini repair inference", flush=True)
        started_at = time.monotonic()
        candidate = normalize_generated_file(model.generate_content(current_prompt).text)
        candidate = restore_prompt_file_omissions(candidate, omitted_blocks)
        candidate = repair_truncated_tail(candidate, original_file_text)
        elapsed_s = time.monotonic() - started_at
        print(f"[repair] Gemini repair inference returned after {elapsed_s:.1f}s", flush=True)
        reason = validate_generated_source_output(
            candidate,
            original_text=original_file_text,
        )
        if reason is None:
            return candidate
        print(f"[repair] rejected model response: {reason}", flush=True)
        last_candidate = candidate
        current_prompt = INVALID_RESPONSE_REPAIR_PROMPT.format(
            reason=reason,
            target_file=target_file,
            current_file=prompt_file_text,
            errors=_compact_repair_errors(errors),
        )
    print("[repair] all inference attempts rejected by validation; falling back to current file", flush=True)
    return current_file_text


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


def compile_binary(
    binary_name: str,
    build_dir: Path,
    source_root: Path,
    *,
    timeout_override_sec: float | None = None,
) -> tuple[int, str, str, Path]:
    effective_build_dir = _preferred_build_dir(build_dir, source_root)
    stdout_prefix: list[str] = []
    cmake_prefix_path = _cmake_prefix_path_for_source(source_root)
    wdl_root = source_root.resolve().parents[1]
    timeout_sec = resolve_build_timeout_sec(binary_name, RUN_TIMEOUT_SEC)
    if timeout_override_sec is not None:
        timeout_sec = max(1, min(timeout_sec, int(max(1.0, timeout_override_sec))))

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


def try_apply_patch(src_file: Path, patch_path: Path, workdir: Path) -> tuple[bool, str, str, Path]:
    target_copy = workdir / src_file.name
    shutil.copy2(src_file, target_copy)
    rc, so, se = run(["patch", "-u", str(target_copy), str(patch_path)], cwd=workdir)
    return rc == 0, so, se, target_copy


def repair_patch(
    model,
    *,
    task: str,
    reference_microbenchmark: str,
    source_microbenchmark: str,
    target_file: str,
    base_file_text: str,
    current_file_text: str,
    local_excerpt: str,
    retrieved_contexts: list[dict],
    bad_patch: str,
    errors: str,
    repair_mode: str,
) -> str:
    compact_errors = _compact_repair_errors(errors)
    task_summary = _compact_task_for_repair(task)
    current_file_issue = validate_generated_source_output(
        current_file_text,
        original_text=base_file_text,
    )
    prompt_file_view, omitted_blocks = prepare_prompt_file_view(current_file_text)
    registration_guidance = (
        build_registration_idiom_guidance(
            base_file_text,
            source_microbenchmark or reference_microbenchmark,
        )
        or "Preserve the existing local benchmark registration idiom and callback storage shape."
    )
    error_guidance = build_compile_repair_guidance(
        errors,
        base_file_text,
        source_microbenchmark=source_microbenchmark or reference_microbenchmark,
    ) or "- No additional compile-specific guidance."
    if repair_mode == "apply":
        prompt = PATCH_REPAIR_PROMPT.format(
            task_summary=task_summary,
            reference_microbenchmark=reference_microbenchmark.strip() or source_microbenchmark.strip() or "(not provided)",
            source_microbenchmark=source_microbenchmark.strip() or "(not provided)",
            target_file=target_file,
            registration_guidance=registration_guidance,
            error_guidance=error_guidance,
            current_file=prompt_file_view,
            bad_patch=bad_patch[:6000].rstrip() + ("\n..." if len(bad_patch) > 6000 else ""),
            errors=compact_errors,
        )
    else:
        prompt = COMPILE_REPAIR_PROMPT.format(
            task_summary=task_summary,
            reference_microbenchmark=reference_microbenchmark.strip() or source_microbenchmark.strip() or "(not provided)",
            source_microbenchmark=source_microbenchmark.strip() or "(not provided)",
            target_file=target_file,
            registration_guidance=registration_guidance,
            error_guidance=error_guidance,
            current_file=prompt_file_view,
            errors=compact_errors,
            contamination_note=(
                f"Important: the current broken file already contains non-code prompt contamination ({current_file_issue}). "
                "Remove that contamination first, then fix the compile error.\n"
                if current_file_issue
                else ""
            ),
        )
    repaired_file_text = _generate_validated_repair_file(
        model,
        prompt=prompt,
        target_file=target_file,
        current_file_text=current_file_text,
        prompt_file_text=prompt_file_view,
        original_file_text=base_file_text,
        errors=compact_errors,
        omitted_blocks=omitted_blocks,
    )
    rendered_patch = build_unified_diff(base_file_text, repaired_file_text, target_file)
    if not rendered_patch.strip():
        raise RuntimeError("Repair model returned no effective file changes; refusing to emit an empty patch.")
    # If the repair fell back to current_file_text (all inferences rejected),
    # the diff is identical to the patch that just failed compile.  Signal the
    # caller to stop retrying by raising instead of re-submitting the same
    # broken code for another compile cycle.
    if repaired_file_text == current_file_text:
        raise RuntimeError(
            "Repair model could not produce a valid alternative; "
            "stopping repair loop to avoid re-submitting the same broken code."
        )
    return rendered_patch


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


def _deadline_from_budget_seconds(budget_seconds: float | None) -> float | None:
    if budget_seconds is None:
        return None
    try:
        value = float(budget_seconds)
    except (TypeError, ValueError):
        return None
    if value <= 0.0:
        return None
    return time.monotonic() + value


def _remaining_budget_seconds(deadline_monotonic: float | None) -> float | None:
    if deadline_monotonic is None:
        return None
    return max(0.0, float(deadline_monotonic) - time.monotonic())


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


def _preflight_inplace_source_edit(live_src_file: Path) -> str | None:
    try:
        with open(live_src_file, "r+b"):
            pass
        probe_fd, probe_name = tempfile.mkstemp(
            prefix=f"{live_src_file.name}.writetest.",
            suffix=".tmp",
            dir=str(live_src_file.parent),
        )
        os.close(probe_fd)
        os.unlink(probe_name)
        return None
    except OSError as exc:
        return (
            "local source tree is not writable for in-place compilation: "
            f"{live_src_file} ({type(exc).__name__}: {exc}). "
            f"file={_path_identity(live_src_file)}; parent={_path_identity(live_src_file.parent)}. "
            "Fix ownership/permissions or switch this binary to an isolated per-attempt source tree/worktree."
        )


def _inplace_source_edit_error(live_src_file: Path, exc: OSError) -> str:
    return (
        "local source replacement failed during compile preparation: "
        f"{live_src_file} ({type(exc).__name__}: {exc}). "
        f"file={_path_identity(live_src_file)}; parent={_path_identity(live_src_file.parent)}. "
        "This is a local filesystem/ownership issue, not a model repair issue. "
        "Fix permissions or use an isolated per-attempt source tree/worktree."
    )


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
    wall_clock_budget_s: float | None = None,
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
    deadline_monotonic = _deadline_from_budget_seconds(wall_clock_budget_s)
    if wall_clock_budget_s is not None:
        record["wall_clock_budget_s"] = float(wall_clock_budget_s)
    print(
        f"[compile] {child_key}: editing from nearest/source benchmark "
        f"{source_microbenchmark or reference_microbenchmark or '(unknown)'} "
        f"for target file {target_file} and build target {binary_name}",
        flush=True,
    )
    record["inplace_source_edit_strategy"] = "shared_source_tree"
    preflight_issue = _preflight_inplace_source_edit(live_src_file)
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

    total_attempts = int(max(1, max_compile_repair_attempts))
    compile_attempts: list[dict[str, object]] = []
    repaired = False
    repaired_patch_path: Path | None = None
    final_text = ""
    final_path = artifact_dir / f"{child_key}_final.cpp"
    success = False

    for attempt_index in range(1, total_attempts + 1):
        remaining_before_attempt = _remaining_budget_seconds(deadline_monotonic)
        if remaining_before_attempt is not None and remaining_before_attempt <= 0.0:
            record["reason"] = (
                f"compile/repair wall-clock budget exhausted before patch attempt {attempt_index}"
            )
            print(f"[compile] {child_key}: {record['reason']}", flush=True)
            break
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
                print(
                    f"[compile] {child_key}: patch apply failed locally; requesting repair inference",
                    flush=True,
                )
                failure_headline = _first_nonempty_line(apply_summary)
                if failure_headline:
                    print(f"[compile] {child_key}: patch/apply issue: {failure_headline}", flush=True)
                if attempt_index >= total_attempts:
                    break
                remaining_before_repair = _remaining_budget_seconds(deadline_monotonic)
                if remaining_before_repair is not None and remaining_before_repair <= 0.0:
                    record["reason"] = (
                        f"compile/repair wall-clock budget exhausted before repair attempt {attempt_index + 1}"
                    )
                    print(f"[compile] {child_key}: {record['reason']}", flush=True)
                    break
                repaired_patch_path = artifact_dir / f"{child_key}_candidate_patch_attempt_{attempt_index + 1}.diff"
                try:
                    current_patch_text = repair_patch(
                        model,
                        task=task,
                        reference_microbenchmark=reference_microbenchmark,
                        source_microbenchmark=source_microbenchmark,
                        target_file=target_file,
                        base_file_text=original_text,
                        current_file_text=original_text,
                        local_excerpt=local_excerpt,
                        retrieved_contexts=[],
                        bad_patch=current_patch_text,
                        errors=apply_summary or "Patch failed to apply.",
                        repair_mode="apply",
                    )
                except RuntimeError as exc:
                    record["reason"] = f"repair model gave up: {exc}"
                    print(f"[compile] {child_key}: {record['reason']}", flush=True)
                    break
                repaired_patch_path.write_text(current_patch_text, encoding="utf-8")
                current_patch_path = repaired_patch_path
                repaired = True
                continue

            try:
                backup_fd, backup_name = tempfile.mkstemp(
                    prefix=f"{live_src_file.name}.{child_key}.",
                    suffix=".bak_folly_rag",
                    dir=str(live_src_file.parent),
                )
                os.close(backup_fd)
                backup = Path(backup_name)
                shutil.copy2(live_src_file, backup)
                try:
                    shutil.copy2(patched_copy, live_src_file)
                    compile_timeout_override = _remaining_budget_seconds(deadline_monotonic)
                    if compile_timeout_override is not None and compile_timeout_override <= 0.0:
                        raise TimeoutError(
                            "compile/repair wall-clock budget exhausted before launching the next build"
                        )
                    compile_rc, compile_so, compile_se, effective_build_dir = compile_binary(
                        binary_name,
                        build_dir,
                        source_root,
                        timeout_override_sec=compile_timeout_override,
                    )
                finally:
                    if backup.exists():
                        shutil.move(str(backup), str(live_src_file))
            except (OSError, TimeoutError, subprocess.TimeoutExpired) as exc:
                replace_summary = (
                    str(exc)
                    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired))
                    else _inplace_source_edit_error(live_src_file, exc)
                )
                print(f"[compile] {child_key}: {replace_summary}", flush=True)
                attempt_record["compile_rc"] = None
                attempt_record["compile_stdout"] = ""
                attempt_record["compile_stderr"] = replace_summary
                attempt_record["compile_summary"] = replace_summary
                attempt_record["effective_build_dir"] = str(build_dir.resolve())
                record["effective_build_dir"] = str(build_dir.resolve())
                if attempt_index == 1:
                    record["first_compile_rc"] = None
                    record["first_compile_stdout"] = ""
                    record["first_compile_stderr"] = replace_summary
                    record["first_compile_summary"] = replace_summary
                elif attempt_index == 2:
                    record["second_compile_rc"] = None
                    record["second_compile_stdout"] = ""
                    record["second_compile_stderr"] = replace_summary
                    record["second_compile_summary"] = replace_summary
                record["reason"] = replace_summary
                break

            compile_summary = trim_build_errors(
                f"{compile_so}\n{compile_se}",
                target_file=target_file,
            )
            if compile_rc == 0:
                print(f"[compile] {child_key}: local build succeeded", flush=True)
            else:
                print(f"[compile] {child_key}: local build failed with rc={compile_rc}", flush=True)
                failure_headline = _first_nonempty_line(compile_summary)
                if failure_headline:
                    print(f"[compile] {child_key}: build issue: {failure_headline}", flush=True)
                if _looks_like_local_build_issue(compile_summary or ""):
                    print(
                        "[compile] detected likely local build/configuration or permission issue; "
                        "repair inference may not help until the environment is fixed",
                        flush=True,
                    )
                if _looks_like_missing_generated_header_issue(compile_summary or ""):
                    reason = (
                        "missing generated thrift headers in the active fbthrift build tree; "
                        "this is a build bootstrap issue, not a source repair issue"
                    )
                    print(f"[compile] {child_key}: {reason}", flush=True)
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

            patched_text = load_text(patched_copy)

            if compile_rc == 0:
                success = True
                final_text = patched_text
                final_path.write_text(final_text, encoding="utf-8")
                break

            if _looks_like_missing_generated_header_issue(compile_summary or ""):
                record["reason"] = (
                    "missing generated thrift headers in the active fbthrift build tree; "
                    "stop repair and fix/build generator targets first"
                )
                break

            if attempt_index >= total_attempts:
                break

            remaining_before_repair = _remaining_budget_seconds(deadline_monotonic)
            if remaining_before_repair is not None and remaining_before_repair <= 0.0:
                record["reason"] = (
                    f"compile/repair wall-clock budget exhausted before repair attempt {attempt_index + 1}"
                )
                print(f"[compile] {child_key}: {record['reason']}", flush=True)
                break
            print(
                f"[compile] {child_key}: requesting repair attempt {attempt_index + 1}/{total_attempts}",
                flush=True,
            )
            repaired_patch_path = artifact_dir / f"{child_key}_candidate_patch_attempt_{attempt_index + 1}.diff"
            try:
                current_patch_text = repair_patch(
                    model,
                    task=task,
                    reference_microbenchmark=reference_microbenchmark,
                    source_microbenchmark=source_microbenchmark,
                    target_file=target_file,
                    base_file_text=original_text,
                    current_file_text=patched_text,
                    local_excerpt="",
                    retrieved_contexts=[],
                    bad_patch="",
                    errors=compile_summary or "Build failed after patch applied.",
                    repair_mode="compile",
                )
            except RuntimeError as exc:
                record["reason"] = f"repair model gave up: {exc}"
                print(f"[compile] {child_key}: {record['reason']}", flush=True)
                break
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
        print(
            f"[compile] {child_key}: compile failed after {len(compile_attempts)} attempt(s) "
            f"while adjusting {source_microbenchmark or reference_microbenchmark or '(unknown)'}",
            flush=True,
        )

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
        default=20,
        help="Maximum total patch/apply/build attempts for one generated variant, including repair retries.",
    )
    parser.add_argument(
        "--wall-clock-budget-s",
        type=float,
        default=None,
        help="Optional wall-clock budget for the whole compile/repair loop.",
    )
    args = parser.parse_args()

    model = create_text_model(REPAIR_MODEL_NAME)

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
        wall_clock_budget_s=args.wall_clock_budget_s,
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

    # Persist cumulative token usage for cross-process aggregation
    from llm_provider import save_token_usage
    token_usage_path = COMPILE_LOGS_DIR / f"{args.example_id}.token_usage.json"
    save_token_usage(token_usage_path)


if __name__ == "__main__":
    main()
