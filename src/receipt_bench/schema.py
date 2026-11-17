"""
The receipt format every pipeline produces and every metric grades.

All amounts are plain floats in the receipt's own currency. Discounts are
receipt-level (order-level) reductions with a positive `amount`; a per-item
discount printed above the subtotal is an item with a negative total_amount.
"""
from typing import List, Optional, TypedDict


class Item(TypedDict):
    description: str
    quantity: float
    unit_price: float
    total_amount: float  # the line total as printed, not the unit price


class TaxLine(TypedDict):
    label: str
    amount: float


class Discount(TypedDict):
    label: str
    amount: float  # positive


class Receipt(TypedDict):
    items: List[Item]
    total_net: Optional[float]  # printed subtotal; None = no subtotal to check
    total_tax: float
    tax_breakdown: List[TaxLine]  # each tax line and each below-subtotal fee
    discounts: List[Discount]
    tip: float  # tip, gratuity and service charges included in the total
    total_amount: float  # the grand total the customer paid
    supplier_name: str
    supplier_address: str
    date: str  # as printed
    time: str  # as printed
    currency: str  # ISO 4217


def empty_receipt(currency: str = "USD") -> Receipt:
    return {
        "items": [],
        "total_net": 0.0,
        "total_tax": 0.0,
        "tax_breakdown": [],
        "discounts": [],
        "tip": 0.0,
        "total_amount": 0.0,
        "supplier_name": "",
        "supplier_address": "",
        "date": "",
        "time": "",
        "currency": currency,
    }
