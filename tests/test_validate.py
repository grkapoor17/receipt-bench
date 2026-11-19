from receipt_bench.validate import adds_up, totals_mismatch


def receipt(items, total_net, total_tax=0.0, tip=0.0, total_amount=None, discounts=()):
    return {
        "items": [{"description": f"item {n}", "total_amount": t} for n, t in enumerate(items)],
        "total_net": total_net,
        "total_tax": total_tax,
        "tip": tip,
        "discounts": [{"label": "promo", "amount": d} for d in discounts],
        "total_amount": total_amount,
    }


def test_exact_receipt_adds_up():
    assert adds_up(receipt([10.00, 5.50], 15.50, total_tax=1.38, tip=3.00, total_amount=19.88))


def test_items_off_from_subtotal_by_two_cents_fails():
    assert totals_mismatch(receipt([10.00, 5.52], 15.50, total_tax=1.38, total_amount=16.88))


def test_exactly_one_cent_off_depends_on_float_noise():
    # Documents current behaviour (same as production): the check compares floats
    # with >= 0.01, so a gap of exactly one cent is caught or missed depending on
    # how the amounts round in binary.
    assert adds_up(receipt([10.00, 5.51], 15.50, total_tax=1.38, total_amount=16.88))


def test_total_within_a_dollar_passes():
    # per-line tax rounding often leaves the printed total a few cents off
    assert adds_up(receipt([10.00], 10.00, total_tax=0.89, total_amount=10.92))


def test_total_off_by_a_dollar_fails():
    assert totals_mismatch(receipt([10.00], 10.00, total_tax=0.89, total_amount=11.89))


def test_subtotal_printed_after_discount():
    r = receipt([20.00, 10.00], 25.00, total_tax=2.00, total_amount=27.00, discounts=[5.00])
    assert adds_up(r)


def test_subtotal_printed_before_discount():
    r = receipt([20.00, 10.00], 30.00, total_tax=2.00, total_amount=27.00, discounts=[5.00])
    assert adds_up(r)


def test_discount_is_capped_at_the_items():
    r = receipt([10.00], None, total_amount=0.00, discounts=[50.00])
    assert adds_up(r)


def test_no_subtotal_skips_the_subtotal_check():
    assert adds_up(receipt([4.00, 6.00], None, total_tax=0.80, total_amount=10.80))


def test_tax_included_prices():
    # Mexico: items include 16% IVA; printed subtotal is pre-tax
    r = receipt([500.00, 500.00], 862.07, total_tax=137.93, total_amount=1000.00)
    assert totals_mismatch(r)  # read as tax-on-top, it doesn't add up
    r["tax_included"] = True
    assert adds_up(r)
    r["tip"] = 100.00
    assert totals_mismatch(r)  # a tip that isn't in the printed total
    r["total_amount"] = 1100.00
    assert adds_up(r)


def test_tax_included_with_wrong_tax_fails():
    r = receipt([500.00, 500.00], 862.07, total_tax=120.00, total_amount=1000.00)
    r["tax_included"] = True
    assert totals_mismatch(r)


def test_small_unit_currencies_allow_rounded_totals():
    # Indonesian receipts round the total to the nearest 100-500 rupiah
    r = receipt([618490.0], 618490.0, total_tax=61849.0, total_amount=680500.0)
    assert totals_mismatch(r)  # no currency: 161 off is too much
    r["currency"] = "IDR"
    assert adds_up(r)
    r["total_amount"] = 690000.0
    assert totals_mismatch(r)


def test_missing_fields_count_as_zero():
    assert adds_up({"items": [], "total_amount": None})
