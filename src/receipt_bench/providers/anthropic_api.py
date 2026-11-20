"""Claude reading the receipt image directly (Anthropic Messages API)."""
import base64
import json
import time
from typing import Any

from ..images import prepare_image
from ..prompt import DIRECT_SYSTEM_PROMPT, USER_TEXT, direct_schema
from . import LLMResult

MAX_TOKENS = 16000


def extract(image_bytes: bytes, model: str, client: Any = None) -> LLMResult:
    import anthropic  # optional dependency: pip install receipt-bench[claude]

    client = client or anthropic.Anthropic()
    data, media_type = prepare_image(image_bytes)
    # default effort for each model; no server-side fallbacks, so a refusal is
    # recorded as an error instead of silently being answered by another model
    result = LLMResult(model=model, settings={"effort": "default", "max_tokens": MAX_TOKENS})
    started = time.time()
    try:
        message = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=DIRECT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                             "data": base64.standard_b64encode(data).decode("utf-8")}},
                {"type": "text", "text": USER_TEXT},
            ]}],
            output_config={"format": {"type": "json_schema", "schema": direct_schema()}},
        )
    except anthropic.APIError as e:
        result.latency_s = round(time.time() - started, 2)
        result.error = "{}: {}".format(type(e).__name__, e)
        return result

    result.latency_s = round(time.time() - started, 2)
    result.stop_reason = message.stop_reason
    result.input_tokens = (message.usage.input_tokens + (message.usage.cache_read_input_tokens or 0)
                           + (message.usage.cache_creation_input_tokens or 0))
    result.output_tokens = message.usage.output_tokens
    if message.stop_reason != "end_turn":
        result.error = "stop_reason: {}".format(message.stop_reason)
        return result
    text = "".join(b.text for b in message.content if b.type == "text")
    try:
        result.out = json.loads(text)
    except ValueError:
        result.error = "bad_json"
    return result
