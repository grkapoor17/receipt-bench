import json
from types import SimpleNamespace

import anthropic

from receipt_bench import claude

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def reading(lines, total_net="sum", total_tax=0.0, tip=0.0, total_amount=None, tax_breakdown=(),
            discounts=()):
    """A minimal model reading around (description, quantity, total) lines."""
    subtotal = round(sum(t for _, _, t in lines), 2)
    return {
        "items": [{"description": d, "quantity": q, "unit_price": None, "total_amount": t} for d, q, t in lines],
        "total_net": subtotal if total_net == "sum" else total_net,
        "total_tax": total_tax,
        "tax_breakdown": list(tax_breakdown),
        "discounts": list(discounts),
        "tip": tip,
        "total_amount": round(subtotal + total_tax + tip, 2) if total_amount is None else total_amount,
        "supplier_name": "", "supplier_address": "", "date": "", "time": "", "currency": "USD",
        "adds_up": True, "notes": "",
    }


def rows(receipt):
    return [(i["description"], i["quantity"], i["total_amount"]) for i in receipt["items"]]


# ------------------------------------------------------------------ merging


def test_repeated_lines_merge_into_quantities():
    r = claude.to_receipt(reading([("Cola", None, 2.95), ("IPA", None, 8.00), ("cola ", None, 2.95)]))
    assert rows(r) == [("Cola", 2.0, 5.9), ("IPA", 1.0, 8.0)]


def test_same_item_at_two_prices_stays_two_rows():
    r = claude.to_receipt(reading([("IPA", None, 5.00), ("IPA", None, 8.00), ("IPA", None, 5.00)]))
    assert rows(r) == [("IPA", 2.0, 10.0), ("IPA", 1.0, 8.0)]


def test_discount_lines_and_their_item_stay_as_printed():
    lines = [("Wings", None, 12.00), ("Trivia 10%", None, -1.20), ("Wings", None, 12.00), ("Comp", None, 0.0)]
    r = claude.to_receipt(reading(lines))
    assert rows(r) == [("Wings", 1.0, 12.0), ("Trivia 10%", 1.0, -1.2), ("Wings", 1.0, 12.0), ("Comp", 1.0, 0.0)]


def test_merge_false_keeps_every_line():
    r = claude.to_receipt(reading([("Cola", None, 2.95), ("Cola", None, 2.95)]), merge=False)
    assert len(r["items"]) == 2


def test_printed_quantity_is_kept_when_merging():
    r = claude.to_receipt(reading([("Taco", 3, 9.00), ("Taco", 1, 3.00)]))
    assert rows(r) == [("Taco", 4, 12.0)]
    assert r["items"][0]["unit_price"] == 3.0


# ------------------------------------------------------------------ fields


def test_negative_tax_line_becomes_a_discount():
    out = reading([("Pizza", None, 20.00)], total_tax=1.00, total_amount=16.00,
                  tax_breakdown=[{"label": "Tax", "amount": 1.00}, {"label": "Promo", "amount": -5.00}])
    r = claude.to_receipt(out)
    assert r["tax_breakdown"] == [{"label": "Tax", "amount": 1.00}]
    assert r["discounts"] == [{"label": "Promo", "amount": 5.00}]


def test_total_tax_is_summed_from_the_breakdown():
    out = reading([("Pizza", None, 20.00)], total_tax=1.00,
                  tax_breakdown=[{"label": "Tax", "amount": 1.00}, {"label": "Delivery fee", "amount": 2.99}])
    assert claude.to_receipt(out)["total_tax"] == 3.99


def test_missing_subtotal_is_derived():
    out = reading([("Pizza", None, 20.00)], total_net=None, total_tax=1.60, tip=4.00)
    assert claude.to_receipt(out)["total_net"] == 20.0


def test_blank_fields_fall_back_to_the_first_pass():
    r = claude.to_receipt(reading([("Pizza", None, 20.0)]), {"supplier_name": "Joe's", "date": "11/17/2026"})
    assert (r["supplier_name"], r["date"]) == ("Joe's", "11/17/2026")


def test_hint_keeps_only_layout_fields():
    hint = json.loads(claude.textract_hint({
        "items": [{"description": "Pizza", "quantity": 1, "unit_price": 20.0, "total_amount": 20.0, "x": 1}],
        "total_amount": 21.6, "supplier_address": "1 Main St",
    }))
    assert hint == {"items": [{"description": "Pizza", "quantity": 1, "unit_price": 20.0, "total_amount": 20.0}],
                    "total_amount": 21.6}


# ------------------------------------------------------------------ repair()


class FakeMessages:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.kwargs = reply, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.reply


def fake_client(text=None, stop_reason="end_turn", error=None):
    reply = SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text or "")],
        usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 1500, "output_tokens": 400}),
    )
    return SimpleNamespace(messages=FakeMessages(reply, error))


FIRST_PASS = {"items": [{"description": "Pizza", "quantity": 1, "unit_price": 2.0, "total_amount": 2.0}],
              "total_net": 20.0, "total_tax": 1.6, "tip": 0.0, "total_amount": 21.6}


def test_repair_accepts_a_reading_that_adds_up():
    client = fake_client(json.dumps(reading([("Pizza", None, 20.0)], total_tax=1.6)))
    result = claude.repair(PNG, FIRST_PASS, client=client)
    assert result.reconciled and result.error is None
    assert rows(result.receipt) == [("Pizza", 1.0, 20.0)]
    assert result.usage["output_tokens"] == 400
    sent = client.messages.kwargs
    assert sent["model"] == claude.DEFAULT_MODEL
    assert sent["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
    assert sent["output_config"]["format"]["schema"] is claude.RECEIPT_SCHEMA


def test_repair_ignores_the_models_own_adds_up_flag():
    out = reading([("Pizza", None, 20.0)], total_amount=30.0)  # model says adds_up=True; it doesn't
    result = claude.repair(PNG, FIRST_PASS, client=fake_client(json.dumps(out)))
    assert result.model_adds_up is True
    assert not result.reconciled
    assert result.receipt is not None


def test_repair_reports_truncation_and_bad_json():
    result = claude.repair(PNG, FIRST_PASS, client=fake_client("{", stop_reason="max_tokens"))
    assert (result.reconciled, result.error) == (False, "stop_reason: max_tokens")
    result = claude.repair(PNG, FIRST_PASS, client=fake_client("not json"))
    assert (result.reconciled, result.error) == (False, "bad_json")


def test_repair_returns_api_errors_instead_of_raising():
    error = anthropic.APIConnectionError(request=None)
    result = claude.repair(PNG, FIRST_PASS, client=fake_client(error=error))
    assert not result.reconciled
    assert result.error.startswith("APIConnectionError")
