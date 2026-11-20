"""OpenAI models reading the receipt image (Responses API, strict JSON schema)."""
import base64
import json
import time
from typing import Any

from ..images import prepare_image
from ..prompt import DIRECT_SYSTEM_PROMPT, USER_TEXT, direct_schema
from . import LLMResult

MAX_OUTPUT_TOKENS = 16000


def extract(image_bytes: bytes, model: str, client: Any = None) -> LLMResult:
    import openai  # optional dependency: pip install receipt-bench[openai]

    client = client or openai.OpenAI()
    data, media_type = prepare_image(image_bytes)
    data_url = "data:{};base64,{}".format(media_type, base64.b64encode(data).decode("utf-8"))
    result = LLMResult(model=model, settings={"reasoning": "default", "image_detail": "high",
                                              "max_output_tokens": MAX_OUTPUT_TOKENS})
    started = time.time()
    try:
        response = client.responses.create(
            model=model,
            instructions=DIRECT_SYSTEM_PROMPT,
            input=[{"role": "user", "content": [
                {"type": "input_image", "image_url": data_url, "detail": "high"},
                {"type": "input_text", "text": USER_TEXT},
            ]}],
            text={"format": {"type": "json_schema", "name": "receipt", "schema": direct_schema(), "strict": True}},
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
    except openai.APIError as e:
        result.latency_s = round(time.time() - started, 2)
        result.error = "{}: {}".format(type(e).__name__, e)
        return result

    result.latency_s = round(time.time() - started, 2)
    result.stop_reason = response.status
    if response.usage is not None:
        result.input_tokens = response.usage.input_tokens
        result.output_tokens = response.usage.output_tokens
    if response.status != "completed":
        reason = getattr(response.incomplete_details, "reason", None)
        result.error = "status: {}{}".format(response.status, " ({})".format(reason) if reason else "")
        return result
    try:
        result.out = json.loads(response.output_text)
    except ValueError:
        result.error = "bad_json"
    return result
