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


class _CopilotModelAdapter:
    """Routes LLM calls through the local Copilot API proxy (OpenAI-compatible)."""

    def __init__(self, model_name: str) -> None:
        self._model_name = str(model_name).strip()
        self._base_url = os.environ.get("COPILOT_API_URL", "http://127.0.0.1:4141")

    def generate_content(self, prompt: str) -> SimpleNamespace:
        import urllib.request
        import urllib.error

        url = f"{self._base_url}/v1/chat/completions"
        # Reasoning models (e.g. claude-opus-4.8) otherwise consume the entire
        # output-token budget on internal reasoning and return EMPTY content
        # (choices=[]). Passing an explicit reasoning_effort bounds the thinking
        # so the model still emits the requested file. "high" maximizes edit
        # quality while comfortably fitting within the budget for these files.
        body_payload = {
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max(int(LLM_MAX_OUTPUT_TOKENS), 32000),
        }
        _effort = os.environ.get("COPILOT_REASONING_EFFORT", "high").strip()
        if _effort and _effort.lower() != "none":
            body_payload["reasoning_effort"] = _effort
        payload = json.dumps(body_payload).encode("utf-8")

        import time as _time
        last_exc = None
        for _retry in range(4):
            try:
                _req = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(_req, timeout=600) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
                last_exc = exc
                _time.sleep(2 ** _retry)
        else:
            raise RuntimeError(f"Copilot API call failed after retries: {last_exc}") from last_exc

        text = ""
        choices = body.get("choices") or []
        if choices:
            text = (choices[0].get("message") or {}).get("content", "")

        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        _record_token_usage(input_tokens, output_tokens)

        result = SimpleNamespace(text=text)
        result.usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
        return result


class _BedrockModelAdapter:
    def __init__(self, model_name: str) -> None:
        self._model_name = str(model_name).strip()
        try:
            import boto3
            from botocore.config import Config as BotoConfig
            import botocore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "boto3 is required for Bedrock mode. Install it in the current Python environment."
            ) from exc

        bearer_token = (
            os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
            or os.environ.get("BEDROCK_API_KEY")
        )
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or LOCATION or "us-east-1"

        read_timeout = int(os.environ.get("BEDROCK_READ_TIMEOUT_SECONDS", "600"))

        if bearer_token:
            # Use bearer-token auth: skip SigV4 signing and inject the token
            # as an Authorization header on every request.
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=BotoConfig(
                    signature_version=botocore.UNSIGNED,
                    read_timeout=read_timeout,
                    retries={"max_attempts": 2},
                ),
            )
            _token = bearer_token  # capture for closure

            def _inject_bearer(request, **kwargs):
                request.headers["Authorization"] = f"Bearer {_token}"

            self._client.meta.events.register(
                "before-sign.bedrock-runtime.*",
                _inject_bearer,
            )
        else:
            # Fall back to standard IAM credential chain (SigV4).
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=BotoConfig(
                    read_timeout=read_timeout,
                    retries={"max_attempts": 2},
                ),
            )

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
    if provider == "copilot":
        return _CopilotModelAdapter(model_name)
    if provider == "bedrock":
        return _BedrockModelAdapter(model_name)

    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    return _VertexModelAdapter(GenerativeModel(model_name))
