from __future__ import annotations

import re
from pathlib import Path


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
    lines = file_text.splitlines()
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
