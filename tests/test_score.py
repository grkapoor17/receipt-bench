import json

import pytest

from receipt_bench import score as sc


def receipt(items, total_net=None, total_tax=0.0, tip=0.0, total_amount=None, discounts=(), **extra):
    subtotal = round(sum(t for _, t in items), 2)
    r = {
        "items": [{"description": d, "quantity": 1.0, "unit_price": t, "total_amount": t} for d, t in items],
        "total_net": subtotal if total_net is None else total_net,
        "total_tax": total_tax, "tax_breakdown": [], "tip": tip,
        "discounts": [{"label": "promo", "amount": d} for d in discounts],
        "total_amount": round(subtotal - sum(discounts) + total_tax + tip, 2) if total_amount is None else total_amount,
        "supplier_name": "Corner Diner", "supplier_address": "", "date": "11/17/2026", "time": "", "currency": "USD",
    }
    r.update(extra)
    return r


def label(r, rid="r1", **extra):
    return dict({"id": rid, "receipt": r}, **extra)


GOLD = receipt([("Burger", 12.0), ("Fries", 4.5), ("Cola", 2.95), ("Cola", 2.95)], total_tax=1.98, tip=4.0)


# ------------------------------------------------------------------ pieces


@pytest.mark.parametrize("gold, pred, same", [
    ("11/17/2026", "2026-11-17", True),
    ("APRIL 15, 2026", "04/15/26", True),
    ("Sun, 5/31/2026 2:46 PM", "5/31/26", True),
    ("12/07/26", "07/12/2026", True),  # dd/mm vs mm/dd
    ("12/07/26", "12/08/26", False),
    ("", "", False),
])
def test_dates(gold, pred, same):
    assert sc.dates_equal(gold, pred) is same


@pytest.mark.parametrize("gold, pred, ok", [
    ("GORKHALI RESTAURANT BROOKLYN INC", "Gorkhali Restaurant", True),
    ("Trader Joe's", "TRADER JOES", True),
    ("The Wingbar", "Lydia's", False),
])
def test_vendor_similarity(gold, pred, ok):
    assert (sc.similarity(gold, pred) >= sc.VENDOR_THRESHOLD) is ok


def test_non_latin_names_are_kept_apart():
    rows = sc.combine_items([{"description": "双拼", "total_amount": 22.99},
                             {"description": "红烧豆腐", "total_amount": 17.99}])
    assert len(rows) == 2
    assert sc.similarity("双拼", "XXIH") < sc.ITEM_NAME_THRESHOLD
    assert sc.similarity("Jalapeño", "JALAPEÑO") == 1.0


def test_combine_items_merges_names_and_drops_zero_rows():
    rows = sc.combine_items([
        {"description": "Cola", "total_amount": 2.95}, {"description": "COLA ", "total_amount": 2.95},
        {"description": "Caramelized Onions", "total_amount": 0.0},
        {"description": "Wings", "total_amount": 12.0}, {"description": "Wings", "total_amount": -12.0},
    ])
    assert rows == [{"name": "Cola", "amount": 5.9}]


# ------------------------------------------------------------------ score_receipt


def test_perfect_prediction():
    row = sc.score_receipt(label(GOLD), json.loads(json.dumps(GOLD)))
    assert row["correct"] and row["passes_check"] and not row["silent_error"]
    assert row["items_f1"] == 1.0 and all(row["fields"].values())


def test_repeated_lines_vs_one_row_with_quantity_are_equivalent():
    pred = dict(GOLD, items=[{"description": "Burger", "total_amount": 12.0}, {"description": "Fries", "total_amount": 4.5},
                             {"description": "Cola", "quantity": 2, "total_amount": 5.9}])
    assert sc.score_receipt(label(GOLD), pred)["correct"]


def test_small_name_typos_are_fine():
    pred = json.loads(json.dumps(GOLD))
    pred["items"][0]["description"] = "BURGR"
    row = sc.score_receipt(label(GOLD), pred)
    assert row["items_f1"] == 1.0 and row["correct"]


def test_unreadable_name_is_not_correct():
    pred = json.loads(json.dumps(GOLD))
    pred["items"][0]["description"] = "QZXV"
    row = sc.score_receipt(label(GOLD), pred)
    assert row["items_f1"] < 1.0 and row["amounts_f1"] == 1.0
    assert not row["correct"]


def test_wrong_price_that_still_adds_up_is_a_silent_error():
    # two prices swapped onto the wrong items: every total still reconciles
    gold = receipt([("Burger", 12.0), ("Fries", 4.5)])
    pred = receipt([("Burger", 4.5), ("Fries", 12.0)])
    row = sc.score_receipt(label(gold), pred)
    assert row["passes_check"] and not row["correct"] and row["silent_error"]


def test_wrong_total_fails_the_check():
    pred = dict(GOLD, total_amount=99.0)
    row = sc.score_receipt(label(GOLD), pred)
    assert not row["fields"]["total_amount"] and not row["correct"] and not row["passes_check"]
    assert not row["silent_error"]


def test_failed_extraction():
    row = sc.score_receipt(label(GOLD), None)
    assert row["failed"] and not row["correct"] and row["items_f1"] == 0.0


def test_only_labelled_and_printed_fields_are_graded():
    gold = dict(GOLD, supplier_name="", date="")
    assert "supplier_name" not in sc.graded_fields(label(gold))
    cord_like = label(dict(GOLD, total_net=None), labelled_fields=["items", "total_tax", "total_amount"])
    assert sc.graded_fields(cord_like) == ["total_tax", "total_amount"]


def test_check_uses_the_label_currency():
    gold = receipt([("Nasi", 618490.0)], total_tax=61849.0, total_amount=680500.0, currency="IDR")
    pred = dict(gold, currency="USD")  # wrong currency shouldn't change whether the numbers reconcile
    row = sc.score_receipt(label(gold), pred)
    assert row["passes_check"] and not row["fields"]["currency"]


# ------------------------------------------------------------------ end to end


def _write(path, rid, obj):
    path.mkdir(parents=True, exist_ok=True)
    (path / (rid + ".json")).write_text(json.dumps(obj))


def test_score_directory_and_summary(tmp_path):
    labels, preds = tmp_path / "ds" / "labels", tmp_path / "preds"
    gold2 = receipt([("Burger", 12.0), ("Fries", 4.5)])
    _write(labels, "a", {"id": "a", "image": "images/a.png", "receipt": GOLD})
    _write(labels, "b", {"id": "b", "image": "images/b.png", "receipt": gold2})
    _write(labels, "c", {"id": "c", "image": "images/c.png", "receipt": gold2})
    _write(labels, "bad", {"id": "bad", "image": "images/bad.png", "receipt": gold2, "label_adds_up": False})
    _write(preds, "a", {"receipt": GOLD, "latency_s": 2.0, "cost_usd": 0.01})
    _write(preds, "b", dict(gold2, total_amount=50.0))  # bare receipt, wrong total
    # "c" missing -> failed

    report = sc.score(str(tmp_path / "ds"), str(preds))
    s = report["summary"]
    assert report["skipped_unreconciled_labels"] == ["bad"]
    assert (s["n"], s["failed"], s["correct"]) == (3, 1, round(1 / 3, 4))
    assert s["passes_check"] == round(1 / 3, 4) and s["silent_error_rate"] == 0.0
    assert s["false_alarm_rate"] == 0.0
    assert s["latency_s"]["p50"] == 2.0 and s["cost_usd"]["mean"] == 0.01
    table = sc.markdown_table([report], ["demo"])
    assert "| **Fully correct** | 33.3% |" in table
