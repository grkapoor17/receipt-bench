"""receipt-bench: receipt extraction pipelines and a benchmark to compare them."""
from .schema import Receipt, empty_receipt
from .textract import parse_expense
from .validate import adds_up, totals_mismatch

__version__ = "0.1.0"

__all__ = ["Receipt", "empty_receipt", "parse_expense", "adds_up", "totals_mismatch", "__version__"]
