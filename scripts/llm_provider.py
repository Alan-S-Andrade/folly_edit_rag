from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

from config.settings import LLM_MAX_OUTPUT_TOKENS, LLM_PROVIDER, LOCATION, PROJECT_ID


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
        parts: list[str] = []
        for block in (((response or {}).get("output") or {}).get("message") or {}).get("content", []) or []:
            text = block.get("text")
            if text:
                parts.append(str(text))
        return SimpleNamespace(text="".join(parts))


def create_text_model(model_name: str) -> Any:
    provider = str(LLM_PROVIDER).strip().lower()
    if provider == "bedrock":
        return _BedrockModelAdapter(model_name)

    import vertexai
    from vertexai.generative_models import GenerativeModel

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    return GenerativeModel(model_name)
