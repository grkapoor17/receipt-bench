"""
Deterministic arithmetic checks on an extracted receipt.

No model grades a receipt here: a reading either adds up or it doesn't. These are
the same checks that decide, in production, whether a Textract scan gets a second
read from an LLM and whether that second read is accepted.
"""
from typing import Any, Dict

# items vs printed subtotal: must match to the cent
TOLERANCE = 0.01
# items - discounts + tax + tip vs printed total: off by less than 1 is fine
# (rounding on per-line tax, cash rounding, etc.)
TOTAL_TOLERANCE = 1.00
# currencies with small units, where totals are rounded to tens or hundreds
# (e.g. Indonesian receipts round to the nearest 100-500 rupiah)
TOTAL_TOLERANCE_BY_CURRENCY = {"IDR": 500.0, "VND": 1000.0, "KRW": 100.0}


def total_tolerance(receipt: Dict[str, Any]) -> float:
    return TOTAL_TOLERANCE_BY_CURRENCY.get((receipt.get("currency") or "").upper(), TOTAL_TOLERANCE)


def _num(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def items_sum(receipt: Dict[str, Any]) -> float:
    return sum(_num(i.get("total_amount")) for i in receipt.get("items") or [])


def discount_amount(receipt: Dict[str, Any], items_total: float) -> float:
    """Receipt-level discounts, capped so they can't take the items below zero."""
    if items_total <= 0:
        return 0.0
    total = sum(_num(d.get("amount")) for d in receipt.get("discounts") or [])
    return min(max(0.0, total), items_total)


def totals_mismatch(receipt: Dict[str, Any]) -> bool:
    """
    True when the reading doesn't reconcile:
    - the items must match the subtotal (when there is one), before or after
      receipt-level discounts, AND
    - discounted items + tax + tip must be within total_tolerance() of the total
      (1.00, or more for small-unit currencies such as IDR).
    """
    items_total = items_sum(receipt)
    after_discounts = items_total - discount_amount(receipt, items_total)
    total_net = receipt.get("total_net")
    if receipt.get("tax_included"):
        # prices include tax: the printed subtotal is either pre-tax (subtotal +
        # tax gives the items back) or already tax-inclusive (equals the items);
        # and items - discounts + tip must give the total
        if total_net is not None:
            net = _num(total_net)
            candidates = (net + _num(receipt.get("total_tax")), net)
            if min(abs(x - c) for x in (items_total, after_discounts) for c in candidates) >= TOLERANCE:
                return True
        total = after_discounts + _num(receipt.get("tip"))
        return abs(total - _num(receipt.get("total_amount"))) >= total_tolerance(receipt)
    if total_net is not None:
        net = _num(total_net)
        if min(abs(items_total - net), abs(after_discounts - net)) >= TOLERANCE:
            return True
    total = after_discounts + _num(receipt.get("total_tax")) + _num(receipt.get("tip"))
    return abs(total - _num(receipt.get("total_amount"))) >= total_tolerance(receipt)


def adds_up(receipt: Dict[str, Any]) -> bool:
    return not totals_mismatch(receipt)
