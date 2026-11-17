from pytest import approx

from receipt_bench.textract import analyze_expense, detect_currency, parse_expense
from receipt_bench.validate import adds_up


def field(type_, value, label=None):
    f = {"Type": {"Text": type_}, "ValueDetection": {"Text": value}}
    if label is not None:
        f["LabelDetection"] = {"Text": label}
    return f


def line(item, price, qty=None):
    fields = [field("ITEM", item), field("PRICE", price)]
    if qty is not None:
        fields.append(field("QUANTITY", qty))
    return {"LineItemExpenseFields": fields}


def response(lines, summary):
    return {"ExpenseDocuments": [{
        "SummaryFields": summary,
        "LineItemGroups": [{"LineItems": lines}],
    }]}


def test_parses_a_simple_receipt():
    r = parse_expense(response(
        [line("Burger", "$12.00"), line("Fries", "$4.50")],
        [
            field("VENDOR_NAME", "Corner Diner"),
            field("INVOICE_RECEIPT_DATE", "11/17/2026"),
            field("SUBTOTAL", "$16.50"),
            field("TAX", "$1.46", "Sales Tax"),
            field("TIP", "$3.00"),
            field("TOTAL", "$20.96"),
        ],
    ))
    assert [(i["description"], i["total_amount"]) for i in r["items"]] == [("Burger", 12.0), ("Fries", 4.5)]
    assert (r["total_net"], r["total_tax"], r["tip"], r["total_amount"]) == (16.5, 1.46, 3.0, 20.96)
    assert r["supplier_name"] == "Corner Diner"
    assert r["date"] == "11/17/2026"
    assert r["currency"] == "USD"
    assert r["discounts"] == []
    assert adds_up(r)


def test_same_description_lines_combine():
    r = parse_expense(response(
        [line("Cola", "2.95"), line("Cola", "2.95"), line("Nachos", "9.00", qty="1")],
        [field("TOTAL", "14.90")],
    ))
    cola = r["items"][0]
    assert (cola["description"], cola["quantity"], cola["total_amount"]) == ("Cola", 2.0, 5.9)
    assert cola["unit_price"] == 2.95
    assert len(r["items"]) == 2


def test_total_tax_row_is_not_double_counted():
    r = parse_expense(response(
        [line("Thali", "₹400.00")],
        [
            field("SUBTOTAL", "₹400.00"),
            field("TAX", "10.00", "CGST 2.5%"),
            field("TAX", "10.00", "SGST 2.5%"),
            field("TAX", "20.00", "Total Tax"),
            field("TOTAL", "₹420.00"),
        ],
    ))
    assert r["tax_breakdown"] == [{"label": "CGST 2.5%", "amount": 10.0}, {"label": "SGST 2.5%", "amount": 10.0}]
    assert r["total_tax"] == 20.0
    assert r["currency"] == "INR"
    assert adds_up(r)


def test_service_charge_and_gratuity_sum_into_tip():
    r = parse_expense(response(
        [line("Pasta", "20.00")],
        [field("SUBTOTAL", "20.00"), field("SERVICE_CHARGE", "2.00"), field("GRATUITY", "1.00"),
         field("TOTAL", "23.00")],
    ))
    assert r["tip"] == 3.0


def test_missing_subtotal_and_tax_are_derived():
    r = parse_expense(response([line("Salad", "8.00")], [field("TOTAL", "8.70"), field("TAX", "0.70")]))
    assert r["total_net"] == approx(8.0)
    r = parse_expense(response([line("Salad", "8.00")], [field("SUBTOTAL", "8.00"), field("TOTAL", "8.70")]))
    assert r["total_tax"] == approx(0.7)


def test_missing_total_is_derived():
    r = parse_expense(response([line("Tea", "3.00")], [field("SUBTOTAL", "3.00"), field("TAX", "0.27")]))
    assert r["total_amount"] == approx(3.27)


def test_empty_response():
    r = parse_expense({"ExpenseDocuments": []})
    assert r["items"] == [] and r["total_amount"] == 0.0 and r["currency"] == "USD"


def test_currency_from_symbol_tax_label_and_address():
    assert detect_currency({"SummaryFields": [field("TOTAL", "€12,50")]}) == "EUR"
    assert detect_currency({"SummaryFields": [field("TAX", "1.30", "HST 13%")]}) == "CAD"
    assert detect_currency({"SummaryFields": [field("VENDOR_ADDRESS", "12 MG Road, Pune 411001")]}) == "INR"
    assert detect_currency({"SummaryFields": [field("VENDOR_ADDRESS", "Shoreditch, United Kingdom")]}) == "GBP"
    assert detect_currency({"SummaryFields": []}) == "USD"


def test_analyze_expense_sends_bytes():
    class FakeTextract:
        def analyze_expense(self, Document):
            self.document = Document
            return {"ExpenseDocuments": []}

    client = FakeTextract()
    assert analyze_expense(b"img", client=client) == {"ExpenseDocuments": []}
    assert client.document == {"Bytes": b"img"}
