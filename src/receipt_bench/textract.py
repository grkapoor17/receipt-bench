"""
AWS Textract AnalyzeExpense -> Receipt.

`parse_expense` is pure (raw response in, Receipt out), so it can run over cached
Textract responses without AWS credentials. `analyze_expense` makes the API call.
"""
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .schema import Receipt, empty_receipt

# ------------------------------------------------------------------ API


def analyze_expense(image_bytes: bytes, client: Any = None, region: str = "us-east-1") -> Dict[str, Any]:
    """Run AnalyzeExpense on an image (JPEG/PNG, up to 5 MB) and return the raw response."""
    if client is None:
        import boto3  # optional dependency: pip install receipt-bench[textract]

        client = boto3.client("textract", region_name=region)
    return client.analyze_expense(Document={"Bytes": image_bytes})


# ------------------------------------------------------------------ currency

_SYMBOL_RE = re.compile(r"[$€£¥₹₩₺₴₽]")

_SYMBOL_TO_CURRENCY = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₩": "KRW",
    "₺": "TRY",
    "₴": "UAH",
    "₽": "RUB",
}

# tax labels that strongly indicate a country
_TAX_LABEL_CURRENCY = {
    "CGST": "INR",
    "SGST": "INR",
    "IGST": "INR",
    "CESS": "INR",
    "CENTRAL GST": "INR",
    "STATE GST": "INR",
    "GST NO": "INR",
    "HST": "CAD",
    "PST": "CAD",
}

# vendor-address keywords (matched case-insensitively)
_ADDRESS_CURRENCY = {
    "india": "INR",
    "maharashtra": "INR",
    "delhi": "INR",
    "karnataka": "INR",
    "tamil nadu": "INR",
    "gujarat": "INR",
    "rajasthan": "INR",
    "uttar pradesh": "INR",
    "west bengal": "INR",
    "telangana": "INR",
    "kerala": "INR",
    "punjab": "INR",
    "united states": "USD",
    "usa": "USD",
    "united kingdom": "GBP",
    "canada": "CAD",
    "australia": "AUD",
    "japan": "JPY",
    "germany": "EUR",
    "france": "EUR",
    "italy": "EUR",
    "spain": "EUR",
    "netherlands": "EUR",
    "south korea": "KRW",
}

_INDIA_PIN_RE = re.compile(r"\b[1-9]\d{5}\b")  # 6-digit Indian PIN code


def detect_currency(doc: Dict[str, Any]) -> str:
    """Best guess at the currency of one ExpenseDocument; USD when nothing points elsewhere."""
    # 1. a currency symbol on a total
    for t in ("TOTAL", "SUBTOTAL", "AMOUNT_DUE"):
        match = _SYMBOL_RE.search(_summary_value(doc, [t]) or "")
        if match:
            return _SYMBOL_TO_CURRENCY.get(match.group(0), "USD")

    # 2. country-specific tax labels
    for f in doc.get("SummaryFields", []):
        label = ((f.get("LabelDetection") or {}).get("Text") or "").upper().strip()
        type_text = ((f.get("Type") or {}).get("Text") or "").upper().strip()
        for token in (label, type_text):
            for tax_key, currency in _TAX_LABEL_CURRENCY.items():
                if tax_key in token:
                    return currency

    # 3. vendor address keywords / Indian PIN code
    address = (_summary_value(doc, ["VENDOR_ADDRESS"]) or "").lower()
    if address:
        for keyword, currency in _ADDRESS_CURRENCY.items():
            if keyword in address:
                return currency
        if _INDIA_PIN_RE.search(address):
            return "INR"

    return "USD"


# ------------------------------------------------------------------ fields


def _to_float(val: Optional[str]) -> float:
    if val is None:
        return 0.0
    txt = re.sub(r"[^\d\.\-]", "", val)  # drop currency symbols, commas, spaces
    try:
        return float(txt)
    except ValueError:
        return 0.0


def _summary_value(doc: Dict[str, Any], type_names: List[str]) -> Optional[str]:
    """ValueDetection.Text of the first SummaryField whose type is in type_names."""
    for f in doc.get("SummaryFields", []):
        if (f.get("Type") or {}).get("Text", "") in type_names:
            return (f.get("ValueDetection") or {}).get("Text")
    return None


def _summary_values_sum(doc: Dict[str, Any], type_names: List[str]) -> float:
    """Sum of every SummaryField whose type is in type_names."""
    total = 0.0
    for f in doc.get("SummaryFields", []):
        if ((f.get("Type") or {}).get("Text", "") or "") in type_names:
            total += _to_float((f.get("ValueDetection") or {}).get("Text"))
    return total


_TAX_AGGREGATE_RE = re.compile(r"\btotal\b", re.IGNORECASE)


def _tax_breakdown(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Individual TAX lines, leaving out a printed "Total Tax" row that only sums
    the lines above it (otherwise CGST + SGST + Total Tax double-counts).
    """
    breakdown = []
    for f in doc.get("SummaryFields", []):
        if (f.get("Type") or {}).get("Text", "") != "TAX":
            continue
        label = ((f.get("LabelDetection") or {}).get("Text") or "").strip()
        amount = _to_float((f.get("ValueDetection") or {}).get("Text"))
        if amount > 0 and not _TAX_AGGREGATE_RE.search(label):
            breakdown.append({"label": label, "amount": amount})
    return breakdown


def _line_item_value(fields: List[Dict[str, Any]], wanted_type: str) -> Optional[str]:
    for f in fields:
        if (f.get("Type") or {}).get("Text", "") == wanted_type:
            return (f.get("ValueDetection") or {}).get("Text")
    return None


# ------------------------------------------------------------------ main


def parse_expense(response: Dict[str, Any]) -> Receipt:
    """Convert a raw AnalyzeExpense response into a Receipt (first document only)."""
    docs = response.get("ExpenseDocuments", [])
    if not docs:
        return empty_receipt()
    doc = docs[0]

    # items: lines with the same description are combined into one row
    combined: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"description": "", "quantity": 0.0, "unit_price": 0.0, "total_amount": 0.0}
    )
    for group in doc.get("LineItemGroups", []):
        for line in group.get("LineItems", []):
            fields = line.get("LineItemExpenseFields", [])
            description = (_line_item_value(fields, "ITEM") or "").strip()
            qty = _to_float(_line_item_value(fields, "QUANTITY") or "1") or 1.0
            line_total = _to_float(
                _line_item_value(fields, "PRICE") or _line_item_value(fields, "AMOUNT") or "0"
            )
            row = combined[description]
            row["description"] = description
            row["quantity"] += qty
            row["total_amount"] += line_total
            row["unit_price"] = row["total_amount"] / row["quantity"]
    items = list(combined.values())

    total_amount = _to_float(_summary_value(doc, ["TOTAL", "AMOUNT_DUE"]))
    tip = _summary_values_sum(doc, ["TIP", "GRATUITY", "SERVICE_CHARGE"])
    total_net = _to_float(_summary_value(doc, ["SUBTOTAL"]))
    tax_breakdown = _tax_breakdown(doc)
    total_tax = sum(t["amount"] for t in tax_breakdown) if tax_breakdown else _summary_values_sum(doc, ["TAX"])

    # fill in whichever of total / subtotal / tax is missing from the other two
    if total_amount == 0 and (total_net > 0 or total_tax > 0 or tip > 0):
        total_amount = total_net + total_tax + tip
    if total_net == 0 and total_amount > 0:
        total_net = max(total_amount - total_tax - tip, 0.0)
    if total_tax == 0 and total_amount > 0 and total_net > 0:
        total_tax = max(total_amount - total_net - tip, 0.0)

    return {
        "items": items,
        "total_net": total_net,
        "total_tax": total_tax,
        "tax_breakdown": tax_breakdown,
        "discounts": [],  # AnalyzeExpense has no discount field
        "tip": tip,
        "total_amount": total_amount,
        "supplier_name": _summary_value(doc, ["VENDOR_NAME"]) or "",
        "supplier_address": _summary_value(doc, ["VENDOR_ADDRESS"]) or "",
        "date": _summary_value(doc, ["INVOICE_RECEIPT_DATE"]) or "",
        "time": _summary_value(doc, ["TIME"]) or "",
        "currency": detect_currency(doc),
    }
