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
| Datasets: CORD v2 loader; private set of 80 hand-checked real-world receipts | ✅ |
| More pipelines (Claude direct, GPT, Gemini, an open model) | 🚧 |
| Metrics: field accuracy, line-item F1, totals pass rate, p50/p95 latency, $/doc | 🚧 |
| CI eval gate on a 20-receipt subset | 🚧 |

## Install

```bash
pip install -e ".[all]"   # boto3 + anthropic + Pillow + pyarrow
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

## Datasets

Every dataset is written in the same layout (`labels/<id>.json` + `images/`), and each label lists the fields it actually labels, so a source without vendor or date labels isn't graded on them.

**CORD v2:** 1,000 Indonesian receipts with line items ([clovaai/cord](https://github.com/clovaai/cord), Park et al. 2019, CC-BY-4.0).

```bash
python -m receipt_bench.datasets.cord --split test --out data/cord   # 100 receipts
```

The loader normalises CORD's mixed number formats (`28.182`, `Rp 165000.00`, `9.999,99`), handles tax-inclusive pricing, and records whether each label reconciles. On the test split, 97 of 100 do; the other 3 are inconsistent in the source labels and are kept but marked.

**Private real-world set:** 80 receipts (US, Mexico, India) from the SplitNShare author's own bills, with item prices checked by a person when the bill was split. Labels record what is printed: tips added afterwards, handwritten amounts and cash-vs-card variants are noted, not labelled. These images contain personal details, so only aggregate results are published.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pytest
```

The tests never call AWS or Anthropic.

## License

MIT. CORD v2 is © its authors, CC-BY-4.0; it is downloaded at runtime and not redistributed here.
