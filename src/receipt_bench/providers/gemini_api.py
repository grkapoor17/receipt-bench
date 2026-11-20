"""Gemini models reading the receipt image (google-genai SDK, JSON schema output)."""
import json
import time
from typing import Any

from ..images import prepare_image
from ..prompt import DIRECT_SYSTEM_PROMPT, USER_TEXT, direct_schema
from . import LLMResult

MAX_OUTPUT_TOKENS = 16000


def extract(image_bytes: bytes, model: str, client: Any = None) -> LLMResult:
    from google import genai  # optional dependency: pip install receipt-bench[gemini]
    from google.genai import errors, types

    client = client or genai.Client()  # reads GEMINI_API_KEY
    data, media_type = prepare_image(image_bytes)
    result = LLMResult(model=model, settings={"thinking": "default", "max_output_tokens": MAX_OUTPUT_TOKENS})
    started = time.time()
    try:
        response = client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=data, mime_type=media_type), USER_TEXT],
            config=types.GenerateContentConfig(
                system_instruction=DIRECT_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_json_schema=direct_schema(),
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
    except errors.APIError as e:
        result.latency_s = round(time.time() - started, 2)
        result.error = "{}: {}".format(type(e).__name__, e)
        return result

    result.latency_s = round(time.time() - started, 2)
    usage = response.usage_metadata
    if usage is not None:
        result.input_tokens = usage.prompt_token_count or 0
        # thinking tokens are billed as output
        result.output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    finish = response.candidates[0].finish_reason if response.candidates else None
    result.stop_reason = getattr(finish, "name", None) or (str(finish) if finish else None)
    if result.stop_reason != "STOP":
        result.error = "finish_reason: {}".format(result.stop_reason)
        return result
    try:
        result.out = json.loads(response.text)
    except (TypeError, ValueError):
        result.error = "bad_json"
    return result
