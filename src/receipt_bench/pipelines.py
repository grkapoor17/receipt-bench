"""
End-to-end pipelines: image bytes in, Receipt out.

More pipelines (Claude direct, other vision models) will be added alongside these
so the benchmark can compare them on the same receipts.
"""
import time
from dataclasses import dataclass
from typing import Any, Optional

from . import claude, textract
from .schema import Receipt
from .validate import totals_mismatch


@dataclass
class PipelineResult:
    receipt: Receipt
    adds_up: bool
    latency_s: float
    used_fallback: bool = False  # a second model was called
    fallback: Optional[claude.RepairResult] = None


def textract_only(image_bytes: bytes, textract_client: Any = None) -> PipelineResult:
    started = time.time()
    receipt = textract.parse_expense(textract.analyze_expense(image_bytes, client=textract_client))
    return PipelineResult(
        receipt=receipt,
        adds_up=not totals_mismatch(receipt),
        latency_s=round(time.time() - started, 2),
    )


def textract_with_claude_repair(image_bytes: bytes, textract_client: Any = None,
                                claude_client: Any = None, model: str = claude.DEFAULT_MODEL,
                                effort: str = claude.DEFAULT_EFFORT) -> PipelineResult:
    """
    The production approach: Textract first; only if its reading fails the totals
    check, ask Claude to re-read the image, and keep Claude's reading only if it
    passes the same check. Otherwise the Textract reading stands.
    """
    started = time.time()
    first = textract.parse_expense(textract.analyze_expense(image_bytes, client=textract_client))
    if not totals_mismatch(first):
        return PipelineResult(receipt=first, adds_up=True, latency_s=round(time.time() - started, 2))

    fix = claude.repair(image_bytes, first, client=claude_client, model=model, effort=effort)
    receipt = fix.receipt if fix.reconciled and fix.receipt is not None else first
    return PipelineResult(
        receipt=receipt,
        adds_up=fix.reconciled,
        latency_s=round(time.time() - started, 2),
        used_fallback=True,
        fallback=fix,
    )
