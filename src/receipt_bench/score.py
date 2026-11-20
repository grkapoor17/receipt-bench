"""
Score pipeline predictions against labelled receipts.

    python -m receipt_bench.score --labels data/cord/test --preds runs/textract_only/cord-test [--preds ...]

A prediction directory holds one <id>.json per receipt, either a bare Receipt or
{"receipt": Receipt | null, "latency_s": ..., "cost_usd": ..., "error": ...}.
A missing file or a null receipt counts as a failed extraction.

Per receipt:
- amounts (subtotal, tax, tip, total, discount total): correct to the cent
- vendor (fuzzy), date (format- and dd/mm-tolerant), currency (exact); only
  graded when the source labels the field and the label isn't blank
- line items: rows with the same normalised name are combined on both sides
  (repeated lines vs. one row with a quantity are the same reading), then
  matched one-to-one on amount + similar name -> precision / recall / F1.
  Rows that come to 0 (free modifiers) are ignored on both sides.
  amounts_f1 matches on amount alone (ignores how names were read).
- split_correct (headline): every item matched on name + amount and the grand
  total right, i.e. every person's share comes out right. Tax / tip / fee /
  discount classification doesn't matter here: a split shares those proportionally.
- correct (strict): split_correct plus every graded amount field right
- passes_check: the prediction passes validate.totals_mismatch
- silent_error: passes the check but isn't correct (the check can't catch it);
  the summary also reports false alarms (fails the check but was correct)
"""
import argparse
import difflib
import json
import math
import os
import re
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .datasets import load_labels
from .validate import totals_mismatch

AMOUNT_FIELDS = ["total_net", "total_tax", "tip", "total_amount", "discount_total"]
TEXT_FIELDS = ["supplier_name", "date", "currency"]
ALL_FIELDS = AMOUNT_FIELDS + TEXT_FIELDS
# which Receipt field a labelled_fields entry covers
_FIELD_SOURCE = {"discount_total": "discounts"}

ITEM_NAME_THRESHOLD = 0.5
VENDOR_THRESHOLD = 0.8

# ------------------------------------------------------------------ helpers


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def amounts_equal(a: Any, b: Any) -> bool:
    return round(_num(a), 2) == round(_num(b), 2)


def normalize_text(s: Any) -> str:
    # \W is Unicode-aware: keeps letters/digits in any script (双拼, Ñ), drops punctuation
    return " ".join(re.sub(r"[\W_]+", " ", str(s or "").lower()).split())


def similarity(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 1.0 if a == b else 0.0
    if a == b:
        return 1.0
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 3 and (" " + short + " ") in (" " + long_ + " "):
        return 0.9  # one is a whole-word part of the other ("GORKHALI" in "GORKHALI RESTAURANT")
    return difflib.SequenceMatcher(None, a, b).ratio()


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def _year(y: int) -> int:
    return y + 2000 if y < 100 else y


def date_candidates(s: Any) -> Set[Tuple[int, int, int]]:
    """Every (year, month, day) a printed date could mean; "12/07/26" is both 7 Dec and 12 Jul."""
    text = str(s or "").lower()
    out: Set[Tuple[int, int, int]] = set()

    def add(y: int, m: int, d: int) -> None:
        try:
            date(_year(y), m, d)
            out.add((_year(y), m, d))
        except ValueError:
            pass

    m = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", text)
    if m:
        add(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", text)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        add(y, a, b)
        add(y, b, a)
    m = re.search(r"\b([a-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{2,4})\b", text)
    if m and m.group(1) in _MONTHS:
        add(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))
    m = re.search(r"\b(\d{1,2})\s+([a-z]{3})[a-z]*\.?,?\s+(\d{2,4})\b", text)
    if m and m.group(2) in _MONTHS:
        add(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))
    return out


def dates_equal(gold: Any, pred: Any) -> bool:
    g, p = date_candidates(gold), date_candidates(pred)
    if g and p:
        return bool(g & p)
    return normalize_text(gold) == normalize_text(pred) != ""


def item_key(description: Any) -> str:
    """
    Name used to combine repeated rows. Bracketed modifiers are ignored, so
    "Taco Platter (Shrimp, Chicken)" and "Taco Platter" are the same dish.
    """
    text = str(description or "")
    base = normalize_text(re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text))
    return base or normalize_text(text)


def combine_items(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge rows with the same normalised name, summing amounts (first-seen order).
    Rows that come to 0 are dropped: a free modifier line ("Caramelized Onions
    0.00") or a comp that nets out doesn't change anyone's share.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    for i in items or []:
        key = item_key(i.get("description"))
        row = rows.setdefault(key, {"name": i.get("description") or "", "amount": 0.0})
        row["amount"] = round(row["amount"] + _num(i.get("total_amount")), 2)
    return [r for r in rows.values() if r["amount"] != 0]


def _f1(matched: int, n_gold: int, n_pred: int) -> Tuple[float, float, float]:
    p = matched / n_pred if n_pred else (1.0 if not n_gold else 0.0)
    r = matched / n_gold if n_gold else (1.0 if not n_pred else 0.0)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def match_items(gold: List[Dict[str, Any]], pred: List[Dict[str, Any]],
                use_names: bool = True) -> int:
    """Greedy one-to-one matching on equal amount, best name similarity first."""
    pairs = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if amounts_equal(g["amount"], p["amount"]):
                sim = similarity(g["name"], p["name"]) if use_names else 1.0
                if sim >= ITEM_NAME_THRESHOLD:
                    pairs.append((sim, gi, pi))
    used_g, used_p, matched = set(), set(), 0
    for _, gi, pi in sorted(pairs, reverse=True):
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            matched += 1
    return matched


# ------------------------------------------------------------------ scoring


def graded_fields(label: Dict[str, Any]) -> List[str]:
    """Fields this label can be graded on: labelled by the source and, for text, not blank."""
    labelled = label.get("labelled_fields")
    gold = label["receipt"]
    fields = []
    for f in ALL_FIELDS:
        if labelled is not None and _FIELD_SOURCE.get(f, f) not in labelled:
            continue
        if f in TEXT_FIELDS and not str(gold.get(f) or "").strip():
            continue  # not printed / not labelled for this receipt
        if f == "total_net" and gold.get("total_net") is None:
            continue  # no printed subtotal
        fields.append(f)
    return fields


def _value(receipt: Dict[str, Any], f: str) -> Any:
    if f == "discount_total":
        return sum(_num(d.get("amount")) for d in receipt.get("discounts") or [])
    return receipt.get(f)


def score_receipt(label: Dict[str, Any], pred: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    gold = label["receipt"]
    fields = graded_fields(label)
    row: Dict[str, Any] = {"id": label["id"], "failed": pred is None, "fields": {}}
    gold_items = combine_items(gold.get("items"))
    if pred is None:
        row["fields"] = {f: False for f in fields}
        row.update(items_matched=0, amounts_matched=0, n_gold_items=len(gold_items), n_pred_items=0,
                   items_f1=0.0, amounts_f1=0.0, correct=False, split_correct=False, passes_check=False,
                   silent_error=False, split_silent_error=False)
        return row

    for f in fields:
        g, p = _value(gold, f), _value(pred, f)
        if f in AMOUNT_FIELDS:
            row["fields"][f] = amounts_equal(g, p)
        elif f == "date":
            row["fields"][f] = dates_equal(g, p)
        elif f == "supplier_name":
            row["fields"][f] = similarity(g, p) >= VENDOR_THRESHOLD
        else:
            row["fields"][f] = normalize_text(g) == normalize_text(p)

    pred_items = combine_items(pred.get("items"))
    matched = match_items(gold_items, pred_items)
    amounts_matched = match_items(gold_items, pred_items, use_names=False)
    row.update(items_matched=matched, amounts_matched=amounts_matched,
               n_gold_items=len(gold_items), n_pred_items=len(pred_items),
               items_f1=_f1(matched, len(gold_items), len(pred_items))[2],
               amounts_f1=_f1(amounts_matched, len(gold_items), len(pred_items))[2])

    # a split is right when every amount is right and every item has its own price:
    # names may be misread a little (OCR typos) but a price on the wrong item
    # charges the wrong person, so items must match on name AND amount
    amounts_ok = all(row["fields"][f] for f in fields if f in AMOUNT_FIELDS)
    row["correct"] = amounts_ok and matched == len(gold_items) == len(pred_items)
    # what a bill split actually needs: every item's price on the right item, and
    # the right grand total. Tax, tip, fees and order discounts are shared
    # proportionally, so filing a surcharge as tip (or IVA as 0) moves no money.
    row["split_correct"] = row["fields"].get("total_amount", True) and matched == len(gold_items) == len(pred_items)
    # judge the check with the label's currency: currency detection is graded separately
    row["passes_check"] = not totals_mismatch(dict(pred, currency=gold.get("currency")))
    row["silent_error"] = row["passes_check"] and not row["correct"]
    row["split_silent_error"] = row["passes_check"] and not row["split_correct"]
    return row


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    k = max(0, math.ceil(q / 100 * len(values)) - 1)
    return round(values[k], 2)


def summarize(rows: List[Dict[str, Any]], runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    field_acc = {}
    for f in ALL_FIELDS:
        graded = [r["fields"][f] for r in rows if f in r["fields"]]
        if graded:
            field_acc[f] = {"accuracy": round(sum(graded) / len(graded), 4), "n": len(graded)}
    gold = sum(r["n_gold_items"] for r in rows)
    pred = sum(r["n_pred_items"] for r in rows)
    p, rc, f1 = _f1(sum(r["items_matched"] for r in rows), gold, pred)
    _, _, af1 = _f1(sum(r["amounts_matched"] for r in rows), gold, pred)
    latencies = [x["latency_s"] for x in runs if x.get("latency_s") is not None]
    costs = [x["cost_usd"] for x in runs if x.get("cost_usd") is not None]
    passed = sum(r["passes_check"] for r in rows)
    return {
        "n": n,
        "failed": sum(r["failed"] for r in rows),
        "split_correct": round(sum(r["split_correct"] for r in rows) / n, 4) if n else None,
        "correct": round(sum(r["correct"] for r in rows) / n, 4) if n else None,
        "passes_check": round(passed / n, 4) if n else None,
        # of the receipts that pass the check, how many would still split wrong
        "split_silent_error_rate": round(sum(r["split_silent_error"] for r in rows) / passed, 4) if passed else None,
        # of the receipts that pass the check, how many are still wrong in any field
        "silent_error_rate": round(sum(r["silent_error"] for r in rows) / passed, 4) if passed else None,
        # of the receipts that fail the check, how many were actually right
        "false_alarm_rate": (round(sum(r["correct"] and not r["passes_check"] for r in rows) / (n - passed), 4)
                             if n - passed else None),
        "fields": field_acc,
        "items": {"precision": round(p, 4), "recall": round(rc, 4), "f1": round(f1, 4),
                  "amounts_f1": round(af1, 4), "gold_rows": gold, "pred_rows": pred},
        "latency_s": {"p50": _percentile(latencies, 50), "p95": _percentile(latencies, 95), "n": len(latencies)},
        "cost_usd": {"mean": round(sum(costs) / len(costs), 5) if costs else None,
                     "total": round(sum(costs), 4) if costs else None, "n": len(costs)},
    }


def load_predictions(directory: str) -> Dict[str, Dict[str, Any]]:
    """{id: {"receipt": ..., "latency_s": ..., "cost_usd": ..., "error": ...}}"""
    preds = {}
    for name in os.listdir(directory):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(directory, name)) as f:
            data = json.load(f)
        if "receipt" not in data:  # a bare Receipt
            data = {"receipt": data}
        preds[name[:-5]] = data
    return preds


def score(labels_dir: str, preds_dir: str, skip_unreconciled: bool = True) -> Dict[str, Any]:
    labels = list(load_labels(labels_dir))
    skipped = [r["id"] for r in labels if skip_unreconciled and r.get("label_adds_up") is False]
    labels = [r for r in labels if r["id"] not in skipped]
    preds = load_predictions(preds_dir)
    rows, runs = [], []
    for label in labels:
        run = preds.get(label["id"], {})
        runs.append(run)
        rows.append(score_receipt(label, run.get("receipt")))
    return {"labels": labels_dir, "preds": preds_dir, "skipped_unreconciled_labels": skipped,
            "summary": summarize(rows, runs), "receipts": rows}


# ------------------------------------------------------------------ report


def _pct(x: Optional[float]) -> str:
    return "–" if x is None else "{:.1f}%".format(100 * x)


def markdown_table(reports: List[Dict[str, Any]], names: List[str]) -> str:
    def line(label: str, values: List[str]) -> str:
        return "| {} | {} |".format(label, " | ".join(values))

    s = [r["summary"] for r in reports]
    out = [line("", names), "|---|" + "---|" * len(names)]
    out.append(line("Receipts", [str(x["n"]) for x in s]))
    out.append(line("**Split-correct**", [_pct(x["split_correct"]) for x in s]))
    out.append(line("Fully correct", [_pct(x["correct"]) for x in s]))
    out.append(line("Passes totals check", [_pct(x["passes_check"]) for x in s]))
    out.append(line("Splits wrong despite passing", [_pct(x["split_silent_error_rate"]) for x in s]))
    out.append(line("Any field wrong despite passing", [_pct(x["silent_error_rate"]) for x in s]))
    out.append(line("Right despite failing", [_pct(x["false_alarm_rate"]) for x in s]))
    out.append(line("Line items F1", [_pct(x["items"]["f1"]) for x in s]))
    out.append(line("Line item amounts F1", [_pct(x["items"]["amounts_f1"]) for x in s]))
    for f in ALL_FIELDS:
        if any(f in x["fields"] for x in s):
            out.append(line(f, [_pct(x["fields"].get(f, {}).get("accuracy")) for x in s]))
    out.append(line("Failed", [str(x["failed"]) for x in s]))
    out.append(line("Latency p50 / p95 (s)", ["{} / {}".format(x["latency_s"]["p50"], x["latency_s"]["p95"])
                                              if x["latency_s"]["n"] else "–" for x in s]))
    out.append(line("Cost per receipt", ["${:.4f}".format(x["cost_usd"]["mean"]) if x["cost_usd"]["mean"] is not None
                                         else "–" for x in s]))
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Score predictions against labels.")
    parser.add_argument("--labels", required=True, help="dataset dir with labels/ and images/")
    parser.add_argument("--preds", required=True, action="append", help="prediction dir (repeat to compare)")
    parser.add_argument("--name", action="append", help="column name per --preds (default: dir name)")
    parser.add_argument("--json", help="write the full report(s) here")
    parser.add_argument("--keep-unreconciled", action="store_true",
                        help="also grade receipts whose own label doesn't add up")
    args = parser.parse_args(argv)

    reports = [score(args.labels, p, skip_unreconciled=not args.keep_unreconciled) for p in args.preds]
    names = args.name or [os.path.basename(os.path.normpath(p)) for p in args.preds]
    print(markdown_table(reports, names))
    skipped = reports[0]["skipped_unreconciled_labels"]
    if skipped:
        print("\nSkipped {} receipt(s) whose labels don't reconcile: {}".format(len(skipped), ", ".join(skipped)))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(zip(names, reports)), f, indent=1)


if __name__ == "__main__":
    main()
