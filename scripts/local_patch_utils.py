from __future__ import annotations

import difflib
import re
from pathlib import Path

_PROMPT_OMITTED_BLOCK_PREFIX = "__PROMPT_OMITTED_DISABLED_APPENDIX_"


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def normalize_generated_patch(text: str) -> str:
    stripped = _strip_code_fences(text)
    if stripped and not stripped.endswith("\n"):
        stripped += "\n"
    return stripped


def normalize_generated_file(text: str) -> str:
    stripped = _strip_code_fences(text)
    if stripped and not stripped.endswith("\n"):
        stripped += "\n"
    return stripped


def _normalize_benchmark_name(name: str) -> str:
    return " ".join(str(name).split()).strip()


def _ignored_benchmark_name(name: str) -> bool:
    normalized = _normalize_benchmark_name(name)
    return not normalized or normalized == "-"


def _extract_explicit_benchmark_names(text: str) -> set[str]:
    names: set[str] = set()
    for match in re.finditer(
        r"\b(?:BENCHMARK|BENCHMARK_RELATIVE|BENCHMARK_MULTI|BENCHMARK_COUNTERS|FBBENCHMARK)\s*\(\s*([A-Za-z_][A-Za-z0-9_:]*)",
        text,
    ):
        name = _normalize_benchmark_name(match.group(1))
        if not _ignored_benchmark_name(name):
            names.add(name)
    # BENCHMARK_NAMED_PARAM(func, suffix, ...) registers as "func(suffix)" in --bm_list.
    # Also covers BENCHMARK_RELATIVE_NAMED_PARAM, BENCHMARK_NAMED_PARAM_MULTI,
    # BENCHMARK_RELATIVE_NAMED_PARAM_MULTI, BENCHMARK_PARAM, BENCHMARK_RELATIVE_PARAM.
    for match in re.finditer(
        r"\b(?:BENCHMARK(?:_RELATIVE)?_NAMED_PARAM(?:_MULTI)?)\s*\(\s*([A-Za-z_]\w*)\s*,\s*(\w+)",
        text,
    ):
        name = _normalize_benchmark_name(f"{match.group(1)}({match.group(2)})")
        if not _ignored_benchmark_name(name):
            names.add(name)
    for match in re.finditer(
        r"\b(?:BENCHMARK(?:_RELATIVE)?_PARAM)\s*\(\s*([A-Za-z_]\w*)\s*,\s*(\w+)",
        text,
    ):
        name = _normalize_benchmark_name(f"{match.group(1)}({match.group(2)})")
        if not _ignored_benchmark_name(name):
            names.add(name)
    for match in re.finditer(
        r"\baddBenchmark\s*\(\s*(?:__FILE__|[^,]+)\s*,\s*\"([^\"\n]+)\"",
        text,
        flags=re.S,
    ):
        name = _normalize_benchmark_name(match.group(1))
        if not _ignored_benchmark_name(name):
            names.add(name)
    return names


def _added_explicit_benchmark_names(original_text: str, updated_text: str) -> set[str]:
    return _extract_explicit_benchmark_names(updated_text) - _extract_explicit_benchmark_names(original_text)


def _changed_updated_line_groups(original_text: str, updated_text: str) -> list[list[str]]:
    original_lines = str(original_text).splitlines()
    updated_lines = str(updated_text).splitlines()
    matcher = difflib.SequenceMatcher(a=original_lines, b=updated_lines)
    groups: list[list[str]] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in {"insert", "replace"} and j1 < j2:
            groups.append(updated_lines[j1:j2])
    return groups


def _is_block_comment_line(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith(("/*", "*/")):
        return True
    # A line starting with '*' is a block comment continuation only if
    # followed by whitespace, '/', or '*' (or is just '*' alone).
    # Lines like '*ptr = value;' are pointer dereferences, not comments.
    if stripped.startswith("*") and (len(stripped) == 1 or stripped[1] in (" ", "\t", "/", "*")):
        return True
    return False


def _is_cpp_control_header(line: str) -> bool:
    stripped = line.strip()
    return bool(
        stripped.endswith("{")
        and stripped.startswith(("if ", "for ", "while ", "switch ", "else", "do ", "try", "catch "))
    )


def _is_registration_boilerplate_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("//") or _is_block_comment_line(stripped):
        return True
    if stripped.startswith("#include"):
        return True
    if stripped in {"__FILE__,", "__FILE__"}:
        return True
    if stripped.startswith("std::move("):
        return True
    if re.fullmatch(r"[{}()[\],;]+", stripped):
        return True
    if stripped in {"});", "));", "}))", "}));", "};"}:
        return True
    if stripped.startswith(("addBenchmark(", "BENCHMARK(", "BENCHMARK_RELATIVE(", "BENCHMARK_MULTI(", "FBBENCHMARK(")):
        return True
    if "addBenchmark(" in stripped:
        return True
    if re.fullmatch(r'"[^"\n]*"\s*,?', stripped):
        return True
    if stripped.startswith(("std::function<", "folly::Function<")):
        return True
    if stripped.startswith(("template <", "namespace ")):
        return True
    if re.fullmatch(r"return\s+(?:iters|n|iterations|numIters|num_iters)\s*;", stripped):
        return True
    if not _is_cpp_control_header(stripped) and stripped.endswith("{"):
        # Lambda closures are always registration boilerplate.
        if "[](" in stripped:
            return True
        # Benchmark macro continuation lines like "iters) {" are boilerplate,
        # but standalone function definitions like "void myHelper(int x) {"
        # are NOT boilerplate.  Distinguish by checking whether the line begins
        # with a C++ type or qualifier keyword.
        if re.search(r"\)\s*(?:const\s*)?(?:noexcept\s*)?\{$", stripped) and not re.match(
            r"(?:"
            r"(?:void|bool|char|short|int|long|float|double|auto|inline|static"
            r"|virtual|explicit|constexpr|unsigned|signed|size_t|uint\d+_t|int\d+_t)\b"
            r"|std::|folly::|typename\s|template\s"
            r")",
            stripped,
        ):
            return True
    return False


def _normalized_group_logic(group_lines: list[str]) -> str:
    logic_lines: list[str] = []
    for line in group_lines:
        stripped = re.sub(r"//.*$", "", line).strip()
        if _is_registration_boilerplate_line(stripped):
            continue
        logic_lines.append(stripped)
    return " ".join(part for part in logic_lines if part).strip()


def _compact_code(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def _single_call_expression(fragment: str) -> str | None:
    normalized = _compact_code(fragment)
    if not normalized or normalized.count(";") != 1:
        return None
    if normalized.startswith("return"):
        normalized = normalized[len("return"):]
    if any(keyword in normalized for keyword in ("if(", "for(", "while(", "switch(", "catch(", "case", "default:")):
        return None
    if not re.fullmatch(r"[A-Za-z_~][A-Za-z0-9_:<>,*&\[\]\.-]*\([^;]*\);", normalized):
        return None
    return normalized


def _is_exact_existing_call_expression(fragment: str, original_text: str) -> bool:
    normalized = _single_call_expression(fragment)
    if not normalized:
        return False
    return normalized in _compact_code(original_text)


def _registration_only_alias_reason(original_text: str, updated_text: str) -> str | None:
    if not original_text:
        return None
    original_names = _extract_explicit_benchmark_names(original_text)
    updated_names = _extract_explicit_benchmark_names(updated_text)
    added_names = updated_names - original_names
    if not added_names and original_names == updated_names:
        return None

    changed_groups = _changed_updated_line_groups(original_text, updated_text)
    saw_trivial_forwarder = False
    for group in changed_groups:
        normalized_logic = _normalized_group_logic(group)
        if not normalized_logic:
            continue
        if _single_call_expression(normalized_logic):
            saw_trivial_forwarder = True
            continue
        return None

    if not saw_trivial_forwarder:
        return None
    return (
        "response added a benchmark registration but no new timed-path logic; "
        "this looks like a registration-only alias or bare wrapper"
    )


def _looks_like_benchmark_output_appendix(block: str) -> bool:
    text = str(block)
    if "#if 0" not in text or "#endif" not in text:
        return False
    if len(text) < 1200:
        return False

    benchmarkish_patterns = [
        r"relative\s+time/iter\s+iters/s",
        r"\b[A-Za-z0-9_]+:\s+k=\d+",
        r"\b[A-Za-z0-9_]+:\s+k=2\^\d+",
        r"\bCPU\b",
    ]
    hits = 0
    for pattern in benchmarkish_patterns:
        if re.search(pattern, text):
            hits += 1
    return hits >= 2


def prepare_prompt_file_view(text: str) -> tuple[str, dict[str, str]]:
    """Replace large disabled benchmark-result appendices with placeholders for prompting.

    The prompt should stay focused on editable code, but the returned file still needs to
    preserve omitted appendices. This helper emits a prompt-safe view plus a mapping that can
    be rehydrated on model output.
    """

    source = str(text)
    pattern = re.compile(r"(?ms)^[ \t]*#if 0\b.*?^[ \t]*#endif[ \t]*\n?")
    omitted: dict[str, str] = {}
    rendered: list[str] = []
    last = 0
    omitted_count = 0

    for match in pattern.finditer(source):
        block = match.group(0)
        is_trailing = match.end() >= int(len(source) * 0.70)
        if not (is_trailing and _looks_like_benchmark_output_appendix(block)):
            continue
        omitted_count += 1
        placeholder = (
            f"/* {_PROMPT_OMITTED_BLOCK_PREFIX}{omitted_count}\n"
            "   Omitted disabled benchmark-output appendix preserved unchanged outside the edited region.\n"
            "   Keep this placeholder exactly unchanged in the returned file unless the task explicitly requires editing inside it.\n"
            "*/\n"
        )
        omitted[placeholder] = block
        rendered.append(source[last:match.start()])
        rendered.append(placeholder)
        last = match.end()

    rendered.append(source[last:])
    prompt_view = "".join(rendered)
    if prompt_view and not prompt_view.endswith("\n"):
        prompt_view += "\n"
    return prompt_view, omitted


def restore_prompt_file_omissions(text: str, omitted_blocks: dict[str, str]) -> str:
    restored = str(text)
    trailing_blocks: list[str] = []
    for placeholder, original in omitted_blocks.items():
        if placeholder in restored:
            restored = restored.replace(placeholder, original)
        else:
            trailing_blocks.append(original)
    if trailing_blocks:
        if restored and not restored.endswith("\n"):
            restored += "\n"
        restored += "".join(
            block if block.endswith("\n") else f"{block}\n" for block in trailing_blocks
        )
    if restored and not restored.endswith("\n"):
        restored += "\n"
    return restored


def repair_truncated_tail(generated_text: str, original_text: str) -> str:
    """If the LLM output is truncated (missing main() or significantly shorter),
    graft the tail of the original file onto the generated output.

    This handles the common case where Gemini's output token limit causes the
    file to be cut off before main() and late benchmark registrations.
    Returns the generated text unchanged if no grafting is needed.
    """
    orig_has_main = bool(re.search(r"\bint\s+main\s*\(", original_text))
    gen_has_main = bool(re.search(r"\bint\s+main\s*\(", generated_text))

    if not orig_has_main:
        return generated_text  # nothing to graft

    orig_lines = original_text.count("\n")
    gen_lines = generated_text.count("\n")

    # Heuristic: if the generated file has main() and is at least 40% of
    # original size, it's probably not truncated.
    if gen_has_main and gen_lines >= orig_lines * 0.4:
        return generated_text

    # Find the last substantive line in the generated output — look for a
    # function, benchmark registration, or closing brace that we can match
    # in the original to find where the LLM stopped.
    gen_stripped_lines = [l.rstrip() for l in generated_text.splitlines()]
    orig_stripped_lines = [l.rstrip() for l in original_text.splitlines()]

    # Find the best anchor: search backwards from end of generated output
    # for a non-empty line that also appears in the original.
    graft_from_orig_line = None
    for search_back in range(min(30, len(gen_stripped_lines))):
        idx = len(gen_stripped_lines) - 1 - search_back
        if idx < 0:
            break
        anchor = gen_stripped_lines[idx].strip()
        if not anchor or anchor in ("{", "}"):
            continue
        # Find this line in the original, searching from the middle onwards
        # (the tail is more likely to match near the end of the original).
        search_start = max(0, len(orig_stripped_lines) // 3)
        for oi in range(search_start, len(orig_stripped_lines)):
            if orig_stripped_lines[oi].strip() == anchor:
                graft_from_orig_line = oi + 1  # graft everything AFTER match
                break
        if graft_from_orig_line is not None:
            break

    if graft_from_orig_line is None:
        # Fallback: find main() in the original and graft from a few lines
        # before it to capture any preceding benchmark registrations.
        main_match = None
        for oi, line in enumerate(orig_stripped_lines):
            if re.search(r"\bint\s+main\s*\(", line):
                main_match = oi
                break
        if main_match is not None:
            # Include ~20 lines before main() for context
            graft_from_orig_line = max(0, main_match - 20)
        else:
            return generated_text  # give up

    tail = "\n".join(original_text.splitlines()[graft_from_orig_line:])
    if not tail.strip():
        return generated_text

    result = generated_text.rstrip() + "\n\n// ── auto-grafted tail from original file ──\n" + tail
    if not result.endswith("\n"):
        result += "\n"

    grafted_lines = result.count("\n")
    print(
        f"[truncation-repair] grafted {grafted_lines - gen_lines} lines from original "
        f"(line {graft_from_orig_line}+) to fix truncated output "
        f"({gen_lines} -> {grafted_lines} lines)",
        flush=True,
    )
    return result


def build_unified_diff(original_text: str, updated_text: str, target_file: str) -> str:
    diff = difflib.unified_diff(
        original_text.splitlines(keepends=True),
        updated_text.splitlines(keepends=True),
        fromfile=f"a/{target_file}",
        tofile=f"b/{target_file}",
        n=3,
    )
    rendered = "".join(diff)
    if rendered and not rendered.endswith("\n"):
        rendered += "\n"
    return rendered


def _split_camel_token(token: str) -> list[str]:
    return re.findall(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|\d+", token)


def microbenchmark_search_tokens(name: str) -> list[str]:
    raw_parts = [part for part in re.split(r"[^A-Za-z0-9]+", name) if part]
    tokens: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        lowered = token.lower()
        if len(token) < 2 or lowered in seen:
            return
        seen.add(lowered)
        tokens.append(token)

    compact_name = " ".join(name.split())
    if compact_name:
        add(compact_name)

    for part in raw_parts:
        add(part)
        for subpart in _split_camel_token(part):
            add(subpart)

    if "(" in name and ")" in name:
        base = name.split("(", 1)[0].strip()
        inner = name[name.find("(") + 1 : name.rfind(")")].strip()
        if base:
            add(base)
            add(f"benchmark{base}")
        if inner:
            add(inner)

    first_word = raw_parts[0] if raw_parts else ""
    if first_word:
        add(f"benchmark{first_word}")

    return tokens


def _priority_tokens(name: str) -> list[str]:
    priorities: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        lowered = token.lower()
        if len(token) < 2 or lowered in seen:
            return
        seen.add(lowered)
        priorities.append(token)

    compact_name = " ".join(name.split())
    if "(" in compact_name and ")" in compact_name:
        base = compact_name.split("(", 1)[0].strip()
        inner = compact_name[compact_name.find("(") + 1 : compact_name.rfind(")")].strip()
        add(base)
        add(f"benchmark{base}")
        add(inner)
        return priorities

    raw_parts = [part for part in re.split(r"[^A-Za-z0-9]+", compact_name) if part]
    if raw_parts:
        add(raw_parts[0])
        add(f"benchmark{raw_parts[0]}")
        if len(raw_parts) > 1:
            add(raw_parts[1])
        if "_" in compact_name and len(raw_parts) > 2:
            add(raw_parts[-1])
    return priorities


def _line_match_score(line: str, tokens: list[str], priority_tokens: list[str]) -> float:
    lowered = line.lower()
    score = 0.0
    for token in tokens:
        if token.lower() in lowered:
            score += 1.0
    for token in priority_tokens:
        if token.lower() in lowered:
            score += 4.0
    stripped = line.strip()
    if "BENCHMARK" in line:
        score += 0.75
    if "addBenchmark" in line:
        score += 0.5
    if stripped.startswith("#define "):
        score += 0.25
    if stripped.startswith("void ") or stripped.startswith("static "):
        score += 0.25
    return score


def _select_anchor_lines(lines: list[str], source_microbenchmark: str, max_hits: int = 3) -> list[int]:
    tokens = microbenchmark_search_tokens(source_microbenchmark)
    priority_tokens = _priority_tokens(source_microbenchmark)
    scored: list[tuple[float, int]] = []
    for idx, line in enumerate(lines):
        score = _line_match_score(line, tokens, priority_tokens)
        if score > 0.0:
            scored.append((score, idx))

    if not scored:
        fallback_tokens = microbenchmark_search_tokens(Path(source_microbenchmark).name)
        fallback_priority_tokens = _priority_tokens(Path(source_microbenchmark).name)
        for idx, line in enumerate(lines):
            score = _line_match_score(line, fallback_tokens, fallback_priority_tokens)
            if score > 0.0:
                scored.append((score, idx))

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[int] = []
    for _, idx in scored:
        if all(abs(idx - existing) > 12 for existing in chosen):
            chosen.append(idx)
        if len(chosen) >= max_hits:
            break
    if chosen:
        return sorted(chosen)

    for idx, line in enumerate(lines):
        if "BENCHMARK" in line or "addBenchmark" in line:
            return [idx]
    return [max(0, min(len(lines) - 1, 0))]


def build_local_source_excerpt(
    file_text: str,
    source_microbenchmark: str,
    *,
    window: int = 24,
    max_chars: int = 12000,
) -> str:
    prompt_view, _ = prepare_prompt_file_view(file_text)
    lines = prompt_view.splitlines()
    if not lines:
        return ""

    anchors = _select_anchor_lines(lines, source_microbenchmark)
    spans: list[tuple[int, int]] = []
    for idx in anchors:
        start = max(0, idx - window)
        end = min(len(lines), idx + window + 1)
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))

    rendered: list[str] = []
    for start, end in spans:
        rendered.append(f"[EXCERPT lines {start + 1}-{end}]")
        rendered.extend(lines[start:end])

    text = "\n".join(rendered).strip()
    if len(text) <= max_chars:
        return text

    compact_rendered: list[str] = []
    reduced_window = max(8, window // 2)
    spans = []
    for idx in anchors:
        start = max(0, idx - reduced_window)
        end = min(len(lines), idx + reduced_window + 1)
        if spans and start <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    for start, end in spans:
        compact_rendered.append(f"[EXCERPT lines {start + 1}-{end}]")
        compact_rendered.extend(lines[start:end])
    text = "\n".join(compact_rendered).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n..."


def render_retrieved_context(
    retrieved_contexts: list[dict],
    *,
    target_file: str = "",
    source_microbenchmark: str = "",
    max_contexts: int = 4,
    max_chars: int = 12000,
) -> str:
    if not retrieved_contexts:
        return (
            "No retrieved context is available for this attempt. "
            "Use the current target file as the authoritative style, API, and registration reference."
        )

    target_file = str(target_file).strip()
    source_tokens = microbenchmark_search_tokens(source_microbenchmark)
    scored: list[tuple[float, int, dict]] = []
    for idx, ctx in enumerate(retrieved_contexts):
        text = str(ctx.get("text", ""))
        lowered = text.lower()
        score = 0.0
        if target_file and target_file.lower() in lowered:
            score += 8.0
        for token in source_tokens:
            if token.lower() in lowered:
                score += 1.0
        if "KIND: benchmark_src_block" in text or "KIND: benchmark_src_file" in text:
            score += 1.5
        scored.append((score, idx, ctx))

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = [ctx for _, _, ctx in scored[: max(1, max_contexts)]]
    rendered: list[str] = []
    total_chars = 0
    for i, ctx in enumerate(chosen, start=1):
        text = str(ctx.get("text", "")).strip()
        block = f"[CONTEXT {i}]\n{text}"
        next_total = total_chars + len(block) + (2 if rendered else 0)
        if rendered and next_total > max_chars:
            break
        rendered.append(block)
        total_chars = next_total
    if not rendered and retrieved_contexts:
        text = str(retrieved_contexts[0].get("text", "")).strip()
        rendered.append(f"[CONTEXT 1]\n{text[:max_chars].rstrip()}")
    return "\n\n".join(rendered)


def _detected_callback_storage_type(file_text: str) -> str:
    prompt_view, _ = prepare_prompt_file_view(file_text)
    if "std::function<" in prompt_view:
        return "std::function"
    if "folly::Function<" in prompt_view:
        return "folly::Function"
    return ""


def _looks_like_generator_registration_file(file_text: str) -> bool:
    prompt_view, _ = prepare_prompt_file_view(file_text)
    lines = prompt_view.splitlines()
    if not lines:
        return False
    return (
        any("addBenchmark(__FILE__" in line for line in lines)
        and any("#define " in line for line in lines)
        and any("testOrder" in line or "tests[" in line for line in lines)
    )


def _defined_macro_names(file_text: str) -> set[str]:
    return set(re.findall(r"(?m)^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", file_text))


def _generator_family_expansion_reason(original_text: str, updated_text: str) -> str | None:
    if not _looks_like_generator_registration_file(original_text):
        return None

    original_lines = {line.strip() for line in original_text.splitlines() if line.strip()}
    defined_macros = _defined_macro_names(original_text)
    benign_macros = {
        "BENCHMARK",
        "BENCHMARK_RELATIVE",
        "BENCHMARK_DRAW_LINE",
        "BENCHMARK_MULTI",
        "FBBENCHMARK",
        "TEST",
        "TEST_F",
        "CHECK",
        "DCHECK",
        "LOG",
    }

    for line in updated_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped in original_lines:
            continue
        macro_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\([^)]*\)\s*;?$", stripped)
        if macro_match:
            macro_name = macro_match.group(1)
            if macro_name in defined_macros and macro_name not in benign_macros:
                return (
                    f"response adds macro-expanded family registration `{stripped}` in a generated benchmark file; "
                    "add one concrete `addBenchmark(...)` entry with one unique benchmark name instead"
                )
        if "testOrder.push_back(" in stripped or re.search(r"\btests\s*\[", stripped):
            return (
                "response adds table or family expansion entries in a generated benchmark file; "
                "add one concrete `addBenchmark(...)` entry with one unique benchmark name instead"
            )
    return None


def build_registration_idiom_guidance(
    file_text: str,
    source_microbenchmark: str = "",
) -> str:
    prompt_view, _ = prepare_prompt_file_view(file_text)
    lines = prompt_view.splitlines()
    if not lines:
        return ""

    storage_type = _detected_callback_storage_type(file_text)
    has_add_benchmark = any("addBenchmark(" in line for line in lines)
    generator_like = _looks_like_generator_registration_file(file_text)
    family_name = " ".join(str(source_microbenchmark).split()).strip()

    if generator_like:
        sentences = [
            "This file appears to register benchmarks indirectly through a table or macro pipeline and a final addBenchmark loop."
        ]
        if family_name:
            sentences.append(
                f"For `{family_name}`, preserve that registration pipeline instead of inventing a standalone registration style."
            )
        if storage_type:
            sentences.append(
                f"Preserve the existing callback storage type `{storage_type}` and the existing final `addBenchmark(...)` call shape."
            )
        sentences.append(
            "Do not add a new macro-family invocation, `testOrder.push_back(...)`, or `tests[...]` table assignment that expands into many benchmarks."
        )
        sentences.append(
            "When the evaluator requires exactly one new benchmark, add one concrete `addBenchmark(...)` entry with one unique benchmark name, keep the existing generated family unchanged, and do not reuse an existing `--bm_list` benchmark name."
        )
        if storage_type == "std::function":
            sentences.append(
                "Do not replace `std::function` with `folly::Function`, and do not add an extra wrapper lambda unless the original file already uses that pattern."
            )
        if storage_type == "folly::Function":
            sentences.append(
                "Do not replace `folly::Function` with `std::function` or another callable wrapper unless the original file already mixes those patterns."
            )
        return " ".join(sentences)

    if has_add_benchmark and storage_type:
        return (
            f"This file uses direct `addBenchmark(...)` registrations with `{storage_type}`-style callable storage nearby. "
            f"Preserve that local registration idiom and callback storage type while adding exactly one new benchmark."
        )

    return ""


def build_edit_location_guidance(
    file_text: str,
    source_microbenchmark: str,
) -> str:
    prompt_view, _ = prepare_prompt_file_view(file_text)
    lines = prompt_view.splitlines()
    if not lines:
        return ""

    source_name = " ".join(str(source_microbenchmark).split()).strip()
    if not source_name:
        return ""

    family_base = source_name
    for separator in ("(", ":"):
        if separator in family_base:
            family_base = family_base.split(separator, 1)[0].strip()
    source_lower = source_name.lower()
    family_lower = family_base.lower()

    def _is_codeish(line: str) -> bool:
        stripped = line.strip()
        return bool(stripped) and not stripped.startswith("//")

    exact_hits = [line.strip() for line in lines if _is_codeish(line) and source_lower in line.lower()]
    anchor_hits = [
        line.strip()
        for line in lines
        if _is_codeish(line)
        and family_lower
        and family_lower in line.lower()
        and ("BENCHMARK" in line or "addBenchmark" in line or "add" in line)
    ]
    generator_like = any("addBenchmark(__FILE__" in line for line in lines) and bool(anchor_hits)

    if not exact_hits and anchor_hits:
        anchor = anchor_hits[0]
        if generator_like:
            return (
                f"This file appears to generate the {family_base} benchmark family programmatically rather than defining "
                f"a standalone `{source_name}` benchmark block. Add exactly one concrete `addBenchmark(...)` registration adjacent to "
                f"`{anchor}`, keep the existing generated family unchanged, and do not add a new macro-family invocation or table entry that expands into many benchmarks."
            )
        return (
            f"The named source benchmark `{source_name}` does not appear as a standalone implementation block. "
            f"Add exactly one new benchmark registration adjacent to `{anchor}` instead of inventing a new placement site elsewhere in the file."
        )

    if exact_hits:
        anchor = exact_hits[0]
        return (
            f"Keep the edit adjacent to the existing `{source_name}` code or registration site, for example near `{anchor}`. "
            "Do not move the change into an unrelated benchmark family."
        )

    return (
        f"Add exactly one new benchmark registration for a variant derived from `{source_name}` close to its existing family registration, "
        "and avoid modifying unrelated benchmark families."
    )


def trim_build_errors(
    text: str,
    *,
    target_file: str = "",
    max_lines: int = 120,
    max_chars: int = 6000,
) -> str:
    lines = text.splitlines()
    if not lines:
        return ""

    target_name = Path(target_file).name if target_file else ""
    relevant_indices: set[int] = set()
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if "error:" in lowered or "fatal error:" in lowered:
            for neighbor in range(max(0, idx - 2), min(len(lines), idx + 5)):
                relevant_indices.add(neighbor)
        elif target_name and target_name in line:
            for neighbor in range(max(0, idx - 1), min(len(lines), idx + 2)):
                relevant_indices.add(neighbor)
        elif "gmake[" in line or line.startswith("make: ***"):
            relevant_indices.add(idx)

    trimmed_lines: list[str] = []
    if relevant_indices:
        last_index = -2
        for idx in sorted(relevant_indices):
            if idx > last_index + 1 and trimmed_lines:
                trimmed_lines.append("...")
            trimmed_lines.append(lines[idx])
            last_index = idx
    else:
        trimmed_lines = [
            line
            for line in lines
            if "CMake Warning (dev)" not in line
            and "This warning is for project developers." not in line
        ]
        trimmed_lines = trimmed_lines[-max_lines:]

    trimmed = "\n".join(trimmed_lines).strip()
    if len(trimmed) > max_chars:
        trimmed = trimmed[-max_chars:]
    return trimmed


def build_compile_repair_guidance(
    errors: str,
    file_text: str,
    *,
    source_microbenchmark: str = "",
) -> str:
    lowered = str(errors).lower()
    storage_type = _detected_callback_storage_type(file_text)
    guidance: list[str] = []

    if "benchmark.h" in lowered and (
        "function(const function&) = delete" in lowered
        or "use of deleted function" in lowered
        or "no match for call to" in lowered
    ):
        guidance.append(
            "This looks like a benchmark-registration callable mismatch rather than a missing include or general syntax error."
        )
        registration_guidance = build_registration_idiom_guidance(
            file_text,
            source_microbenchmark=source_microbenchmark,
        )
        if registration_guidance:
            guidance.append(registration_guidance)
        if storage_type == "std::function":
            guidance.append(
                "Keep the original `std::function`-based registration shape. Avoid replacing it with `folly::Function` or wrapping stored callables in a new mutable lambda."
            )
        elif storage_type == "folly::Function":
            guidance.append(
                "Keep the original `folly::Function`-based registration shape. Avoid swapping the callable wrapper unless the original file already did so."
            )
        guidance.append(
            "If the file uses a final registration loop, add or repair one family entry only and keep that loop structurally unchanged."
        )

    if "operator new" in lowered and "std_function" in lowered:
        guidance.append(
            "Do not paper over this with `<new>` unless the original file already needed it; in this pattern it is usually a side effect of the wrong callable wrapper or constness."
        )

    return "\n".join(f"- {line}" for line in guidance if line)


def _looks_like_source_start(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith(("/*", "//", "#")):
        return True
    common_starts = (
        "template",
        "namespace",
        "using ",
        "typedef ",
        "struct ",
        "class ",
        "enum ",
        "union ",
        "extern ",
        "static ",
        "inline ",
        "constexpr ",
        "consteval ",
        "constinit ",
        "const ",
        "volatile ",
        "auto ",
        "void ",
        "int ",
        "char ",
        "bool ",
        "float ",
        "double ",
        "size_t ",
        "typename ",
        "BENCHMARK",
        "FBBENCHMARK",
        "TEST",
        "static_assert",
        "module ",
        "export ",
    )
    if stripped.startswith(common_starts):
        return True
    if re.match(r"^[A-Z_][A-Z0-9_]*(\(|\s)", stripped):
        return True
    return False


def validate_generated_source_output(
    text: str,
    *,
    original_text: str = "",
    forbidden_benchmark_names: set[str] | None = None,
    require_new_benchmark_name: bool = False,
) -> str | None:
    lowered = text.lower()
    prompt_markers = [
        "local source excerpt from the current target file:",
        "retrieved context:",
        "current patch:",
        "patch/apply/build errors:",
        "return only the complete updated source file contents",
        "return only the complete corrected source file contents",
        "required internal reasoning protocol",
        "what_is_wrong",
        "why_it_is_wrong",
        "how_to_fix_it",
        "protected_metrics",
        "[context ",
        "[excerpt lines",
        "task:",
    ]
    for marker in prompt_markers:
        if marker in lowered:
            return f"response echoed non-source prompt marker: {marker}"

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not _looks_like_source_start(stripped):
            preview = stripped[:120]
            return f"response begins with non-source prose: {preview}"
        break

    main_count = len(re.findall(r"\bint\s+main\s*\(", text))
    if main_count > 1:
        return f"response contains {main_count} definitions of main()"

    # ── Truncation detection ────────────────────────────────────────────
    if original_text:
        orig_lines = original_text.count("\n")
        gen_lines = text.count("\n")
        # If the original has a main() but the generated file dropped it,
        # the LLM almost certainly truncated the output.
        orig_has_main = bool(re.search(r"\bint\s+main\s*\(", original_text))
        if orig_has_main and main_count == 0:
            return (
                f"response is missing main() — the original file has main() but "
                f"the generated output ({gen_lines} lines) dropped it. "
                f"The original file is {orig_lines} lines. "
                f"You MUST preserve the main() function from the original file."
            )
        # Reject drastically shorter files (likely LLM output truncation).
        # Threshold: generated must be at least 40% of original length.
        if orig_lines > 200 and gen_lines < orig_lines * 0.4:
            return (
                f"response appears truncated: {gen_lines} lines vs "
                f"{orig_lines} lines in the original ({gen_lines * 100 // max(orig_lines, 1)}%). "
                f"The complete file must be returned. If the file is too large, "
                f"preserve ALL existing benchmark registrations and main()."
            )

    if original_text:
        family_expansion_reason = _generator_family_expansion_reason(original_text, text)
        if family_expansion_reason:
            return family_expansion_reason
        added_benchmark_names = _added_explicit_benchmark_names(original_text, text)
        alias_reason = _registration_only_alias_reason(original_text, text)
        if alias_reason:
            return alias_reason
        if require_new_benchmark_name and not added_benchmark_names:
            return "response did not add a new explicit benchmark registration with a unique benchmark name"
        if forbidden_benchmark_names:
            forbidden = {
                _normalize_benchmark_name(name)
                for name in forbidden_benchmark_names
                if not _ignored_benchmark_name(name)
            }
            duplicates = sorted(name for name in added_benchmark_names if name in forbidden)
            if duplicates:
                return (
                    f"response reuses existing `--bm_list` benchmark name `{duplicates[0]}`; "
                    "choose a unique new benchmark name"
                )

    return None
