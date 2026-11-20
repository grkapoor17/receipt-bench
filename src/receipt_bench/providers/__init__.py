"""
One extract() per provider: image bytes in, the model's JSON reading out.

Every provider gets prompt.DIRECT_SYSTEM_PROMPT, the same image bytes and
prompt.direct_schema(), at the provider's default reasoning settings. Errors
are returned, not raised, so a benchmark run keeps going.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LLMResult:
    model: str
    out: Optional[Dict[str, Any]] = None  # parsed JSON, None on failure
    input_tokens: int = 0
    output_tokens: int = 0  # includes reasoning / thinking tokens
    latency_s: float = 0.0
    stop_reason: Optional[str] = None
    error: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)
