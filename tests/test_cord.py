import json

import pytest

from receipt_bench.datasets import load_labels
from receipt_bench.datasets.cord import parse_amount, parse_count, to_receipt


@pytest.mark.parametrize("raw, expected", [
    ("28.182", 28182.0),
    ("28,182", 28182.0),
    ("1,591,600", 1591600.0),
    ("Rp 165000.00", 165000.0),
    ("Rp.39.600", 39600.0),
    ("9.999,99", 9999.99),
    ("99,999.99", 99999.99),
    ("@24.000", 24000.0),
    ("*46000", 46000.0),
    ("-2 600", -2600.0),
    ("-60.000", -60000.0),
    ("(0)", 0.0),
    (["6,818", "0"], 6818.0),
    ("-", None),
    (None, None),
])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize("raw, expected", [("2", 2.0), ("x1", 1.0), ("1 x", 1.0), ("2.00", 2.0),
                                           ("1.00xITEMs", 1.0), ("(2", 2.0), (None, None)])
def test_parse_count(raw, expected):
    assert parse_count(raw) == expected


def test_single_menu_entry_and_tax_on_top():
    receipt, adds_up = to_receipt({
        "menu": {"nm": "JASMINE MT ( L )", "cnt": "2", "price": "48,000"},
        "sub_total": {"subtotal_price": "48,000", "tax_price": "4,800"},
        "total": {"total_price": "52,800"},
    })
    assert adds_up and not receipt.get("tax_included")
    assert receipt["items"] == [{"description": "JASMINE MT ( L )", "quantity": 2.0,
                                 "unit_price": 24000.0, "total_amount": 48000.0}]
    assert (receipt["total_net"], receipt["total_tax"], receipt["total_amount"]) == (48000.0, 4800.0, 52800.0)
    assert receipt["currency"] == "IDR"


def test_tax_inclusive_prices_are_detected():
    receipt, adds_up = to_receipt({
        "menu": [{"nm": "CIRENG", "cnt": "1", "price": "29,500"}],
        "sub_total": {"subtotal_price": "26,818", "tax_price": "2,682"},
        "total": {"total_price": "29,500"},
    })
    assert adds_up and receipt["tax_included"]


def test_priced_modifiers_and_item_discounts_become_rows():
    receipt, adds_up = to_receipt({
        "menu": [
            {"nm": "MILK TEA", "cnt": "1", "price": "25,000",
             "sub": [{"nm": "Less Ice"}, {"nm": "Extra Jelly", "cnt": "1", "price": "4,000"}]},
            {"nm": "BLUS WANITA", "cnt": "1", "unitprice": "@120,000", "discountprice": "0%"},
            {"nm": "SHOPPING BAG", "cnt": "1", "unitprice": "@880", "discountprice": "100%"},
            {"nm": "NASI", "cnt": "1", "price": "10,000", "discountprice": "-2,000"},
        ],
        "total": {"total_price": "157,000"},
    })
    rows = [(i["description"], i["total_amount"]) for i in receipt["items"]]
    assert rows == [("MILK TEA", 25000.0), ("Extra Jelly", 4000.0), ("BLUS WANITA", 120000.0),
                    ("SHOPPING BAG", 880.0), ("Discount", -880.0), ("NASI", 10000.0), ("Discount", -2000.0)]
    assert adds_up


def test_service_charge_is_tip_and_discount_is_receipt_level():
    receipt, adds_up = to_receipt({
        "menu": [{"nm": "STEAK", "cnt": "1", "price": "200,000"}],
        "sub_total": {"subtotal_price": "200,000", "service_price": "10,000",
                      "discount_price": "-20,000", "tax_price": "19,000"},
        "total": {"total_price": "209,000"},
    })
    assert receipt["tip"] == 10000.0
    assert receipt["discounts"] == [{"label": "Discount", "amount": 20000.0}]
    assert adds_up


@pytest.mark.parametrize("totals, expected", [
    ({"cashprice": "50000", "changeprice": "14000"}, 36000.0),
    ({"cashprice": "*110000"}, 110000.0),
    ({"total_etc": "20.000", "cashprice": "50.000", "changeprice": "30.000"}, 20000.0),
    ({"creditcardprice": "3,112,800"}, 3112800.0),
])
def test_total_falls_back_when_total_price_is_missing(totals, expected):
    receipt, _ = to_receipt({"menu": [{"nm": "X", "price": "1"}], "total": totals})
    assert receipt["total_amount"] == expected


def test_unpriced_lines_are_skipped():
    receipt, _ = to_receipt({"menu": [{"nm": "Minuman"}, {"nm": "TEA", "price": "5,000"}]})
    assert [i["description"] for i in receipt["items"]] == ["TEA"]


def test_load_labels(tmp_path):
    (tmp_path / "labels").mkdir()
    for rid, status in (("a", "approved"), ("b", "excluded"), ("c", None)):
        record = {"id": rid, "image": "images/{}.png".format(rid), "receipt": {}}
        if status:
            record["review"] = {"status": status}
        (tmp_path / "labels" / (rid + ".json")).write_text(json.dumps(record))
    records = list(load_labels(str(tmp_path)))
    assert [r["id"] for r in records] == ["a", "c"]
    assert records[0]["image"] == str(tmp_path / "images" / "a.png")
    assert len(list(load_labels(str(tmp_path), include_excluded=True))) == 3
