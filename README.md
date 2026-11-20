# receipt-bench

A benchmark of 10 receipt-extraction pipelines (AWS Textract, Claude, GPT and Gemini) on 148 labelled receipts, scored on whether every person's share of the bill comes out right, plus latency and cost. Also the extraction code itself: parsers, deterministic totals checks and an LLM repair step.

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
| Runner: Textract, production repair, Claude / OpenAI / Gemini reading the image directly | ✅ |
| Scorer: split-correct, fully-correct, field accuracy, line-item F1, totals-check pass / miss / false-alarm rates, latency, cost | ✅ |
| First full results (November 2026) | ✅ |
| CI eval gate on a 20-receipt subset | 🚧 |

## Results (November 2026)

148 graded receipts: 80 real-world receipts from the US, Mexico and India (private set), and the first 70 receipts of the CORD v2 test split (Indonesia; 2 skipped because their source labels don't add up). One run per pipeline, standard API pricing, each provider at its default reasoning settings.

**Split-correct**: every line item has the right price on the right item, and the grand total is right, so every person's share comes out right.

| Pipeline | Split-correct, all 148 | Real-world (80) | CORD (68) | Wrong but passed the totals check | Median latency | Cost / receipt |
|---|---|---|---|---|---|---|
| Gemini 3.8 Flash | **96.6%** | 96.2% | 97.1% | 5 | 10.3 s | $0.0069 |
| Claude Opus 5.5 | **95.9%** | 95.0% | 97.1% | 6 | 6.8 s | $0.0306 |
| GPT-6.2 Sol | **95.9%** | 95.0% | 97.1% | 4 | 7.2 s | $0.0099 |
| Gemini 4 Argon | 95.3% | 93.8% | 97.1% | 7 | 9.2 s | $0.0089 |
| Claude Sonnet 5.5 | 91.9% | 93.8% | 89.7% | 12 | 5.7 s | $0.0163 |
| GPT-6 Luna | 91.2% | 91.2% | 91.2% | 10 | 9.3 s | $0.0007 |
| Gemini 3.5 Flash-Lite | 91.2% | 90.0% | 92.7% | 7 | 3.0 s | $0.0017 |
| Claude Haiku 5.5 | 89.9% | 90.0% | 89.7% | 13 | 5.6 s | $0.0010 |
| Textract + Claude Opus 5.5 repair (production) | 81.1% | 91.2% | 69.1% | 25 | 3.6 s | $0.0201 |
| Textract only | 66.9% | 75.0% | 57.4% | 21 | 3.0 s | $0.0100 |

What stands out:

- **Some wrong readings add up perfectly, so a totals check can't catch them.** On Indonesian receipts, `28.182` means 28,182 rupiah. Reading it as 28.182 shrinks every number by the same factor of 1,000, and the receipt still reconciles. That accounts for 17 of production's 21 CORD misses. A tip printed only on page two of a payment-app receipt is missed the same way.
- **Cheap models fail differently.** Haiku's errors mostly still add up (silent errors), while Flash-Lite's mostly trip the totals check (false alarms that trigger an unnecessary second read).
- **The models audited the labels.** Wherever at least 5 of the 8 LLMs agreed on a value that differed from the label, the photo was checked. That found 7 label errors, all fixed before these numbers.

Caveats: the prompt was adapted from a production prompt tuned on Claude; 80 and 68 receipts are small samples (one receipt is about 1.3 points); some real-world labels started from the production pipeline's own output, which, if anything, flatters production; latency includes up to 4 concurrent requests per pipeline.

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

## Running

```bash
python -m receipt_bench.run --dry-run --dataset data/cord/test --pipeline claude:claude-haiku-5-5   # plan + cost estimate, no calls
python -m receipt_bench.run --dataset data/cord/test --pipeline textract --pipeline openai:gpt-6.2-sol --max-cost 5
```

Pipelines: `textract`, `textract+repair` (the production setup), and `claude:<model>`, `openai:<model>`, `gemini:<model>`, which read the image directly. Every LLM gets the same prompt (`prompt.py`), the same image bytes and the same JSON schema, at its provider's default reasoning settings, with no automatic fallback to another model. Each call records latency, token usage and cost (`pricing.py`). Textract responses are cached so the two Textract pipelines pay once per page. Runs resume where they stopped, and `--max-cost` caps the spend. Keys come from the environment or a `.env` file.

## Scoring

```bash
python -m receipt_bench.score --labels data/cord/test --preds runs/a --preds runs/b   # side-by-side table
```

A prediction directory holds one `<id>.json` per receipt (a Receipt, or `{"receipt", "latency_s", "cost_usd"}`). The headline metric is strict, because a bill split needs every price on the right item:

- **Split-correct** (headline): every line item matches on price *and* name, and the grand total is right, so every person's share comes out right. Small OCR typos in a name are fine; a price on the wrong item is not. How tax, tip, fees and order discounts are classified doesn't count here, because a split shares those proportionally.
- **Fully correct** (strict): split-correct, and also every labelled amount field right to the cent (subtotal, tax, tip, discounts).
- **Line items**: rows with the same name are combined on both sides first (three "Cola 2.95" lines and one "Cola ×3 8.85" row are the same reading), and $0 modifier lines are ignored. Then precision / recall / F1, plus an amounts-only F1.
- **The totals check, as a detector**: *passes totals check*; *splits wrong despite passing* (errors the check can't see, so the ones that reach users); *right despite failing* (false alarms that trigger an unnecessary LLM re-read).
- **Fields**: vendor (fuzzy), date (format- and dd/mm-tolerant), currency. A field is graded only when the dataset labels it and the receipt prints it.
- Receipts whose own label doesn't reconcile are skipped (`--keep-unreconciled` to include them).

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pytest
```

The tests never call AWS or Anthropic.

## License

MIT. CORD v2 is © its authors, CC-BY-4.0; it is downloaded at runtime and not redistributed here.
