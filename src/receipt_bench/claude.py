"""
Claude "repair" pass: re-read a receipt image when a first-pass OCR reading fails
the totals check, using that reading as a layout hint.

Claude's own `adds_up` flag is recorded but never trusted; acceptance is decided
by validate.totals_mismatch on the post-processed result.
"""
import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .images import prepare_image
from .schema import Receipt
from .validate import totals_mismatch

DEFAULT_MODEL = "claude-opus-5-5"
# low effort keeps 40-line receipts fast; see the benchmark for the tradeoff
DEFAULT_EFFORT = "low"
DEFAULT_MAX_TOKENS = 16000


def _nullable_number() -> Dict[str, Any]:
    return {"anyOf": [{"type": "number"}, {"type": "null"}]}


def _labelled_amounts() -> Dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"label": {"type": "string"}, "amount": {"type": "number"}},
            "required": ["label", "amount"],
            "additionalProperties": False,
        },
    }


# structured-output schema for the model's reading (one entry per printed line)
RECEIPT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": _nullable_number(),
                    "unit_price": _nullable_number(),
                    "total_amount": {"type": "number"},
                },
                "required": ["description", "quantity", "unit_price", "total_amount"],
                "additionalProperties": False,
            },
        },
        "total_net": _nullable_number(),
        "total_tax": {"type": "number"},
        "tax_breakdown": _labelled_amounts(),
        "discounts": _labelled_amounts(),
        "tip": {"type": "number"},
        "total_amount": {"type": "number"},
        "supplier_name": {"type": "string"},
        "supplier_address": {"type": "string"},
        "date": {"type": "string"},
        "time": {"type": "string"},
        "currency": {"type": "string"},
        "adds_up": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": [
        "items", "total_net", "total_tax", "tax_breakdown", "discounts", "tip", "total_amount",
        "supplier_name", "supplier_address", "date", "time", "currency",
        "adds_up", "notes",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You transcribe photographed receipts into structured data for a bill-splitting app. \
People split the bill item by item using your output, so every price has to be what \
is printed on the receipt.

You get the receipt image and a first-pass reading from an OCR service. That reading \
failed an arithmetic check: its items, tax, tip and total don't reconcile. Common \
causes are a skewed or curled photo that shifts prices onto the wrong line, two lines \
merged into one, items dropped from a long receipt, a misread digit or decimal point, \
or a discount, fee, or subtotal row taken for an item. Use the reading as a hint about \
the layout, but trust the image.

How to fill the fields:
- items: one entry per purchased line, in printed order. total_amount is the line \
total as printed (not the unit price). quantity and unit_price only when printed, \
else null.
- On a skewed, curled or folded photo, a price can sit level with the item above or \
below its own. Pair prices with items by row order down the price column (the third \
price goes with the third priced item; modifier lines without a price don't count), \
not by what lines up horizontally. A repeated item \
normally has the same price each time it appears; if yours don't match, recheck the \
pairing.
- Where a line sits relative to the printed subtotal decides where it goes.
- Above the subtotal (included in it): fees like a bag fee or corkage are items, and \
discounts, comps and voids are items with a negative total_amount.
- Between the subtotal and the total: fees and surcharges (delivery fee, service fee, \
card surcharge, venue fee, non-cash adjustment) go in tax_breakdown under their printed \
label, because the app splits those proportionally, like tax. Promos, coupons, credits \
and other order-level discounts go in discounts under their printed label, with a \
positive amount. With no printed subtotal, treat order-level promos and credits the \
same way.
- A fee or surcharge printed as struck through, waived or FREE was not charged; leave \
it out.
- Subtotal, tax, tip and total rows are never items.
- total_net is the printed subtotal (null if none). total_amount is the printed grand \
total the customer paid.
- tax_breakdown lists each tax line, and each below-subtotal fee, with its printed \
label. Leave out a "total tax" row that only sums the lines above it. total_tax is the \
sum of tax_breakdown.
- tip covers tip, gratuity and service charge lines that the total includes. Do not \
count a suggested-tip table, or a tip line the customer left blank.
- currency is the ISO 4217 code; dates and times as printed, or "" if absent.

Never change a number to make the receipt add up. If, after reading carefully, the \
printed numbers still don't reconcile (the receipt itself is wrong, or part of it is \
unreadable or cut off), transcribe what is printed, set adds_up to false, and say what \
is wrong in notes. Set adds_up to true only when items match the subtotal and items - \
discounts + tax + tip match the total to the cent. notes stays short: what you corrected versus \
the first-pass reading, or why it doesn't reconcile."""


def textract_hint(first_pass: Dict[str, Any]) -> str:
    """The parts of the first-pass reading that help when re-reading the image."""
    keys = ("items", "total_net", "total_tax", "tax_breakdown", "tip", "total_amount", "currency")
    hint = {k: first_pass.get(k) for k in keys if k in first_pass}
    hint["items"] = [
        {k: i.get(k) for k in ("description", "quantity", "unit_price", "total_amount")}
        for i in (first_pass.get("items") or [])
    ]
    return json.dumps(hint, indent=1)


# ------------------------------------------------------------------ post-processing


def merge_duplicate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Combine repeated lines for the same item at the same unit price into one row:
    a bar tab with 11 separate "Cola 2.95" lines becomes one row of quantity 11.

    The key is the name (ignoring case and spacing) plus the unit price, so the
    same drink at happy-hour and full price stays two rows and no price is
    averaged. Only positive lines merge; discounts, comps and $0 lines stay as
    printed. An item with its own discount line right below it also stays as
    printed, so the pair isn't split apart. Rows keep first-appearance order.
    """
    merged: Dict[Tuple[str, float], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for n, item in enumerate(items):
        qty = item["quantity"] or 1.0
        has_own_discount = n + 1 < len(items) and items[n + 1]["total_amount"] < 0
        if item["total_amount"] <= 0 or has_own_discount:
            rows.append(item)
            continue
        key = (" ".join(item["description"].lower().split()), round(item["total_amount"] / qty, 2))
        row = merged.get(key)
        if row is None:
            row = dict(item)
            merged[key] = row
            rows.append(row)
            continue
        row["quantity"] += qty
        row["total_amount"] = round(row["total_amount"] + item["total_amount"], 2)
        row["unit_price"] = row["total_amount"] / row["quantity"]
    return rows


def to_receipt(out: Dict[str, Any], first_pass: Optional[Dict[str, Any]] = None,
               merge: bool = True) -> Receipt:
    """
    Turn the model's reading into a Receipt.

    merge=False keeps one row per printed line (useful for grading line items
    against per-line labels).
    """
    first_pass = first_pass or {}
    items = []
    for i in out["items"]:
        qty = i.get("quantity") or 1.0
        unit = i.get("unit_price")
        items.append({
            "description": i["description"].strip(),
            "quantity": qty,
            "unit_price": unit if unit is not None else (i["total_amount"] / qty if qty else 0.0),
            "total_amount": i["total_amount"],
        })
    if merge:
        items = merge_duplicate_items(items)

    # a negative "tax" line is a promo filed in the wrong place: make it a discount
    discounts = [
        {"label": d["label"] or "Discount", "amount": abs(d["amount"])}
        for d in out.get("discounts") or [] if d.get("amount")
    ]
    discounts += [
        {"label": t["label"] or "Discount", "amount": abs(t["amount"])}
        for t in out["tax_breakdown"] if (t.get("amount") or 0) < 0
    ]
    # sum the printed lines ourselves: the model sometimes lists a fee in the
    # breakdown and then leaves it out of its own total_tax
    tax_breakdown = [t for t in out["tax_breakdown"] if (t.get("amount") or 0) > 0]
    total_tax = round(sum(t["amount"] for t in tax_breakdown), 2) if tax_breakdown else max(out["total_tax"], 0.0)
    # no printed subtotal: derive one, as the Textract parser does
    total_net = out["total_net"]
    if total_net is None:
        total_net = max(out["total_amount"] - total_tax - out["tip"], 0.0)

    return {
        "items": items,
        "total_net": total_net,
        "total_tax": total_tax,
        "tax_breakdown": tax_breakdown,
        "discounts": discounts,
        "tip": out["tip"],
        "total_amount": out["total_amount"],
        "supplier_name": out["supplier_name"] or first_pass.get("supplier_name", ""),
        "supplier_address": out["supplier_address"] or first_pass.get("supplier_address", ""),
        "date": out["date"] or first_pass.get("date", ""),
        "time": out["time"] or first_pass.get("time", ""),
        "currency": out["currency"] or first_pass.get("currency", "USD"),
    }


# ------------------------------------------------------------------ API call


@dataclass
class RepairResult:
    receipt: Optional[Receipt]  # None when the call failed or returned no usable reading
    reconciled: bool  # the post-processed receipt passes validate.totals_mismatch
    model_adds_up: Optional[bool] = None  # what the model claimed
    notes: str = ""
    latency_s: float = 0.0
    model: str = ""
    stop_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None  # the model's JSON, before post-processing


def repair(image_bytes: bytes, first_pass: Dict[str, Any], client: Any = None,
           model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT,
           max_tokens: int = DEFAULT_MAX_TOKENS) -> RepairResult:
    """
    Ask Claude to re-read the receipt, given the first-pass reading that didn't add up.
    API errors are returned in RepairResult.error rather than raised, so a benchmark
    run keeps going.
    """
    import anthropic  # optional dependency: pip install receipt-bench[claude]

    if client is None:
        client = anthropic.Anthropic()
    data, media_type = prepare_image(image_bytes)

    started = time.time()
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.standard_b64encode(data).decode("utf-8"),
                        },
                    },
                    {"type": "text", "text": "First-pass OCR reading:\n" + textract_hint(first_pass)},
                ],
            }],
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": RECEIPT_SCHEMA},
            },
        )
    except anthropic.APIError as e:
        return RepairResult(receipt=None, reconciled=False, model=model,
                            latency_s=round(time.time() - started, 2),
                            error="{}: {}".format(type(e).__name__, e))

    result = RepairResult(
        receipt=None, reconciled=False, model=model,
        latency_s=round(time.time() - started, 2),
        stop_reason=message.stop_reason,
        usage=message.usage.model_dump(),
    )
    if message.stop_reason != "end_turn":
        result.error = "stop_reason: {}".format(message.stop_reason)
        return result

    text = "".join(b.text for b in message.content if b.type == "text")
    try:
        out = json.loads(text)
    except ValueError:
        result.error = "bad_json"
        return result

    result.raw = out
    result.model_adds_up = out.get("adds_up")
    result.notes = out.get("notes", "")
    result.receipt = to_receipt(out, first_pass)
    result.reconciled = not totals_mismatch(result.receipt)
    return result
