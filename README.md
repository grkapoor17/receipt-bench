# receipt-bench

Receipt extraction pipelines, deterministic totals checks, and (in progress) a public benchmark comparing OCR and LLM approaches on accuracy, latency and cost.

The code is the extraction layer from [SplitNShare](https://splitnshare.ai), an iOS bill-splitting app. Users there split a bill item by item, so every price has to be right. The production pipeline:

1. **AWS Textract** (AnalyzeExpense) reads the receipt.
2. A **deterministic check** asks whether it adds up: items vs. the printed subtotal, and items − discounts + tax + tip vs. the printed total.
3. Only if it doesn't, **Claude** re-reads the image using Textract's output as a layout hint.
4. Claude's reading is accepted only if it passes **the same check**. The model's own claim that it adds up is recorded but never trusted.

## Status

| | |
|---|---|
| Textract parser, totals checks, Claude repair, two pipelines | ✅ |
| Dataset (CORD + SROIE + hand-labelled receipts) | 🚧 |
| More pipelines (Claude direct, GPT, Gemini, an open model) | 🚧 |
| Metrics: field accuracy, line-item F1, totals pass rate, p50/p95 latency, $/doc | 🚧 |
| CI eval gate on a 20-receipt subset | 🚧 |

## Install

```bash
pip install -e ".[all]"   # boto3 + anthropic + Pillow
```

## Usage

```python
from receipt_bench import pipelines

image = open("receipt.jpg", "rb").read()
result = pipelines.textract_with_claude_repair(image)

result.receipt         # items, subtotal, tax lines, discounts, tip, total, vendor, date, currency
result.adds_up         # passed the totals check
result.used_fallback   # Claude was called
result.latency_s
```

The parts work on their own too:

```python
from receipt_bench import parse_expense, adds_up
from receipt_bench.claude import repair

receipt = parse_expense(cached_textract_response)   # pure; no AWS call
if not adds_up(receipt):
    fix = repair(image, receipt)                    # RepairResult: receipt, reconciled, usage, latency, error
```

Credentials come from the usual places: AWS via the boto3 credential chain, Anthropic via `ANTHROPIC_API_KEY`.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pytest
```

The tests never call AWS or Anthropic.

## License

MIT
