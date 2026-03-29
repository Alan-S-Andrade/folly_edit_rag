from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from config.settings import LLM_MAX_OUTPUT_TOKENS, LLM_PROVIDER, LOCATION, PROJECT_ID

# ── Global token usage accumulator ──────────────────────────────────────────
# Thread-safe counters that record cumulative token usage across all LLM
# calls within this process.  Persisted to a JSON file on request so that
# parent processes (cloud_tuning.py) can aggregate across sub-processes.

_usage_lock = threading.Lock()
_cumulative_usage: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "call_count": 0,
}


def get_cumulative_token_usage() -> dict[str, int]:
    """Return a snapshot of the cumulative token usage."""
    with _usage_lock:
        return dict(_cumulative_usage)


def _record_token_usage(input_tokens: int, output_tokens: int) -> None:
    with _usage_lock:
        _cumulative_usage["input_tokens"] += input_tokens
        _cumulative_usage["output_tokens"] += output_tokens
        _cumulative_usage["total_tokens"] += input_tokens + output_tokens
        _cumulative_usage["call_count"] += 1


def save_token_usage(path: str | Path) -> None:
    """Write cumulative token usage to a JSON file for cross-process aggregation."""
    usage = get_cumulative_token_usage()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(usage), encoding="utf-8")


class _BedrockModelAdapter:
    def __init__(self, model_name: str) -> None:
        self._model_name = str(model_name).strip()
        try:
            import boto3
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "boto3 is required for Bedrock mode. Install it in the current Python environment."
            ) from exc

        api_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or os.environ.get("BEDROCK_API_KEY")
        if api_key:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or LOCATION or "us-east-1"
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def generate_content(self, prompt: str) -> SimpleNamespace:
        response = self._client.converse(
            modelId=self._model_name,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig={"maxTokens": LLM_MAX_OUTPUT_TOKENS},
        )
        # Extract token usage from Bedrock response
        usage = (response or {}).get("usage") or {}
        input_tokens = int(usage.get("inputTokens") or 0)
        output_tokens = int(usage.get("outputTokens") or 0)
        _record_token_usage(input_tokens, output_tokens)

        parts: list[str] = []
        for block in (((response or {}).get("output") or {}).get("message") or {}).get("content", []) or []:
            text = block.get("text")
            if text:
                parts.append(str(text))
        result = SimpleNamespace(text="".join(parts))
        result.usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        return result


class _VertexModelAdapter:
    """Thin wrapper around VertexAI GenerativeModel that captures token usage."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def generate_content(self, prompt: str) -> Any:
        response = self._model.generate_content(prompt)
        # Extract token usage from Vertex response
        input_tokens = 0
        output_tokens = 0
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is not None:
            input_tokens = int(getattr(usage_metadata, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage_metadata, "candidates_token_count", 0) or 0)
        _record_token_usage(input_tokens, output_tokens)
        return response


def create_text_model(model_name: str) -> Any:
    provider = str(LLM_PROVIDER).strip().lower()
    if provider == "bedrock":
        return _BedrockModelAdapter(model_name)

    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    return _VertexModelAdapter(GenerativeModel(model_name))
