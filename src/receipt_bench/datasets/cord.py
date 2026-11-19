"""
CORD v2 (Park et al., 2019): ~1,000 Indonesian receipts with line-item labels.
CC-BY-4.0. https://huggingface.co/datasets/naver-clova-ix/cord-v2

    python -m receipt_bench.datasets.cord --split test --out data/cord

downloads the split (pinned to REVISION) and writes it in the receipt-bench
layout (see receipt_bench.datasets). Needs pyarrow: pip install receipt-bench[data]

Label mapping (gt_parse -> Receipt):
- menu[] -> items (price is the line total; a priced `sub` modifier becomes its
  own row; a negative item `discountprice` becomes a negative row)
- sub_total.subtotal_price -> total_net, tax_price -> total_tax,
  service_price -> tip (service charges count as tip), discount_price -> discounts
- total.total_price -> total_amount
- tax_included is set when the label only reconciles with tax-inclusive pricing
  (common in CORD: prices include 10% PB1 tax)
CORD has no vendor, address, date or time labels, so those aren't graded.
"""
import argparse
import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from ..images import media_type
from ..schema import Receipt, empty_receipt
from ..validate import totals_mismatch

REPO = "naver-clova-ix/cord-v2"
# pinned so the benchmark doesn't shift under us
REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
FILES = {
    "test": ["data/test-00000-of-00001-9c204eb3f4e11791.parquet"],
    "validation": ["data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet"],
    "train": ["data/train-0000{}-of-00004-{}.parquet".format(i, h) for i, h in enumerate(
        ["b4aaeceff1d90ecb", "7dbbe248962764c5", "688fe1305a55e5cc", "2d0cd200555ed7fd"])],
}

LABELLED_FIELDS = ["items", "total_net", "total_tax", "tip", "discounts", "total_amount"]
EXT = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif"}

# ------------------------------------------------------------------ parsing

_DECIMALS = re.compile(r"[.,](\d{1,2})$")


def parse_amount(value: Any) -> Optional[float]:
    """
    Parse a CORD price string: "28.182" and "28,182" are 28182 (thousands
    separators); "99999.99" and "9.999,99" have two decimals; "Rp", "@", "*",
    spaces and brackets are ignored; "-" or "(...)" make it negative.
    Returns None when there is no number.
    """
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
        return parse_amount(value)
    s = str(value).strip()
    negative = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = re.sub(r"(?i)rp\.?", "", s)
    s = re.sub(r"[^\d.,]", "", s)
    if not re.search(r"\d", s):
        return None
    m = _DECIMALS.search(s)
    if m:
        whole = re.sub(r"[.,]", "", s[:m.start()]) or "0"
        number = float("{}.{}".format(whole, m.group(1)))
    else:
        number = float(re.sub(r"[.,]", "", s))
    return -number if negative else number


def parse_count(value: Any) -> Optional[float]:
    """"2", "x1", "1 x", "2.00", "1.00xITEMs" -> float."""
    if value is None:
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", str(value))
    return float(m.group(0).replace(",", ".")) if m else None


def _as_list(v: Any) -> List[Any]:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _name(v: Any) -> str:
    return " ".join(str(x) for x in _as_list(v)).strip()


def _total(totals: Dict[str, Any]) -> float:
    """
    total_price when labelled; otherwise the amount charged to a card or
    e-money, an unlabelled total line, or cash tendered minus change.
    """
    for key in ("total_price", "creditcardprice", "emoneyprice", "total_etc"):
        value = parse_amount(totals.get(key))
        if value is not None and value > 0:
            return value
    cash = parse_amount(totals.get("cashprice"))
    if cash:
        return cash - (parse_amount(totals.get("changeprice")) or 0.0)
    return 0.0


def to_receipt(gt_parse: Dict[str, Any]) -> Tuple[Receipt, bool]:
    """CORD gt_parse -> (Receipt, whether the label reconciles)."""
    receipt = empty_receipt(currency="IDR")
    items = []
    for m in _as_list(gt_parse.get("menu")):
        if not isinstance(m, dict):
            continue
        qty = parse_count(m.get("cnt")) or 1.0
        total = parse_amount(m.get("price"))
        unit = parse_amount(m.get("unitprice"))
        if total is None and unit is not None:
            total = unit * qty
        if total is None:
            continue  # unpriced line (e.g. a section header)
        items.append({"description": _name(m.get("nm")), "quantity": qty,
                      "unit_price": unit if unit is not None else (total / qty if qty else total),
                      "total_amount": total})
        for sub in _as_list(m.get("sub")):
            sub_price = parse_amount(sub.get("price")) if isinstance(sub, dict) else None
            if sub_price:
                sub_qty = parse_count(sub.get("cnt")) or 1.0
                items.append({"description": _name(sub.get("nm")), "quantity": sub_qty,
                              "unit_price": sub_price / sub_qty, "total_amount": sub_price})
        # item discounts are an amount ("-85,500") or a percentage ("100%")
        raw_discount = m.get("discountprice")
        item_discount = parse_amount(raw_discount)
        if item_discount and "%" in str(raw_discount):
            item_discount = total * item_discount / 100.0
        if item_discount:
            items.append({"description": "Discount", "quantity": 1.0,
                          "unit_price": -abs(item_discount), "total_amount": -abs(item_discount)})
    receipt["items"] = items

    sub_total = gt_parse.get("sub_total") or {}
    totals = gt_parse.get("total") or {}
    if isinstance(sub_total, list):
        sub_total = sub_total[0] if sub_total else {}
    if isinstance(totals, list):
        totals = totals[0] if totals else {}
    receipt["total_net"] = parse_amount(sub_total.get("subtotal_price"))
    receipt["total_tax"] = abs(parse_amount(sub_total.get("tax_price")) or 0.0)
    receipt["tip"] = abs(parse_amount(sub_total.get("service_price")) or 0.0)
    discount = abs(parse_amount(sub_total.get("discount_price")) or 0.0)
    receipt["discounts"] = [{"label": "Discount", "amount": discount}] if discount else []
    receipt["total_amount"] = _total(totals)

    if not totals_mismatch(receipt):
        return receipt, True
    inclusive = dict(receipt, tax_included=True)
    if not totals_mismatch(inclusive):
        return inclusive, True
    return receipt, False


# ------------------------------------------------------------------ build


def download(split: str, raw_dir: str) -> List[str]:
    os.makedirs(raw_dir, exist_ok=True)
    paths = []
    for rel in FILES[split]:
        dest = os.path.join(raw_dir, os.path.basename(rel))
        if not os.path.exists(dest):
            url = "https://huggingface.co/datasets/{}/resolve/{}/{}".format(REPO, REVISION, rel)
            print("downloading", url)
            urllib.request.urlretrieve(url, dest + ".part")
            os.replace(dest + ".part", dest)
        paths.append(dest)
    return paths


def build(split: str, out_dir: str) -> Dict[str, int]:
    import pyarrow.parquet as pq  # optional dependency: pip install receipt-bench[data]

    split_dir = os.path.join(out_dir, split)
    os.makedirs(os.path.join(split_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)
    n = reconciled = 0
    for path in download(split, os.path.join(out_dir, "raw")):
        for row in pq.read_table(path).to_pylist():
            rid = "cord-{}-{:04d}".format(split, n)
            gt = json.loads(row["ground_truth"])
            receipt, adds_up = to_receipt(gt["gt_parse"])
            image = row["image"]["bytes"]
            image_rel = "images/{}.{}".format(rid, EXT[media_type(image)])
            with open(os.path.join(split_dir, image_rel), "wb") as f:
                f.write(image)
            record = {
                "id": rid,
                "source": "cord-v2",
                "split": split,
                "image": image_rel,
                "receipt": receipt,
                "labelled_fields": LABELLED_FIELDS,
                "label_adds_up": adds_up,
                "raw": gt["gt_parse"],
            }
            with open(os.path.join(split_dir, "labels", rid + ".json"), "w") as f:
                json.dump(record, f, indent=1, ensure_ascii=False)
            n += 1
            reconciled += adds_up
    return {"receipts": n, "label_adds_up": reconciled}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=sorted(FILES), default="test")
    parser.add_argument("--out", default="data/cord")
    args = parser.parse_args(argv)
    summary = build(args.split, args.out)
    print("{receipts} receipts written, {label_adds_up} with labels that reconcile".format(**summary))


if __name__ == "__main__":
    main()
