"""
USD prices used to put a cost on every benchmark call.

Standard (non-batch) per-1M-token rates, checked 2026-11-19 against the
providers' pricing pages. Runs also store raw token counts, so costs can be
recomputed if prices change. Gemini 4 Argon goes to $4 / $20 on 2026-12-01
and Gemini 3.8 Flash to $1.50 / $7.50 on 2027-01-01.
"""
from typing import Dict, Tuple

# model -> (input $/1M tokens, output $/1M tokens); output includes reasoning/thinking
PRICES: Dict[str, Tuple[float, float]] = {
    # Anthropic
    "claude-fable-5-2": (8.00, 40.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    # OpenAI
    "gpt-6.2-astra": (8.00, 40.00),
    "gpt-6.2-sol": (2.00, 10.00),
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6-luna": (0.10, 0.50),
    # Google
    "gemini-4-argon": (2.00, 10.00),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}

TEXTRACT_ANALYZE_EXPENSE_PER_PAGE = 0.01


def token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICES:
        raise KeyError("no price for {!r}; add it to receipt_bench.pricing.PRICES".format(model))
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
