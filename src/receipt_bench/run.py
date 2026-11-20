"""
Run pipelines over datasets and save one prediction file per receipt.

    python -m receipt_bench.run --pipeline claude:claude-haiku-5-5 --dataset data/cord/test --dry-run
    python -m receipt_bench.run --pipeline textract --pipeline textract+repair \\
        --dataset data/own --dataset data/cord/test --max-cost 5

Pipelines:
  textract                 AWS Textract AnalyzeExpense only
  textract+repair          production: Textract, then Claude Opus 5.5 (low effort)
                           re-reads receipts that fail the totals check
  claude:<model>           Claude reads the image directly
  openai:<model>           OpenAI model reads the image directly
  gemini:<model>           Gemini model reads the image directly

Output: <runs>/<pipeline>/<dataset>/<id>.json, scored by receipt_bench.score.
Existing results are skipped, so an interrupted run resumes (--retry-errors
redoes failed ones). Textract responses are cached under <cache>/textract/, so
textract and textract+repair pay for each page once.

--max-cost stops submitting new receipts once the run's spend passes the cap.
--dry-run prints what would run and an estimated cost, and calls nothing.
API keys come from the environment or a .env file in the working directory.
"""
import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from . import claude, textract
from .datasets import load_labels
from .images import prepare_image
from .pricing import PRICES, TEXTRACT_ANALYZE_EXPENSE_PER_PAGE, token_cost
from .providers import LLMResult
from .validate import totals_mismatch

REPAIR_MODEL = "claude-opus-5-5"
REPAIR_EFFORT = "low"
# rough per-receipt token use for --dry-run estimates (from production logs, rounded up)
EST_INPUT_TOKENS, EST_OUTPUT_TOKENS = 5000, 1500
EST_REPAIR_INPUT, EST_REPAIR_OUTPUT, EST_REPAIR_SHARE = 7000, 1000, 0.3

PROVIDERS = {"claude", "openai", "gemini"}

# ------------------------------------------------------------------ helpers


def load_dotenv(path: str = ".env") -> List[str]:
    """Set KEY=VALUE lines from a .env file (without overriding the environment). Returns the keys set."""
    if not os.path.exists(path):
        return []
    loaded = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    return loaded


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+_-]+", "-", text).strip("-")


def dataset_name(path: str) -> str:
    """data/own -> own, data/cord/test -> cord-test"""
    parts = os.path.normpath(path).split(os.sep)
    if "data" in parts:
        parts = parts[parts.index("data") + 1:]
    return "-".join(parts[-2:]) if len(parts) > 1 else parts[-1]


def parse_pipeline(spec: str):
    if spec in ("textract", "textract+repair"):
        return spec, None
    provider, _, model = spec.partition(":")
    if provider not in PROVIDERS or not model:
        raise ValueError("unknown pipeline {!r} (textract, textract+repair, claude:<model>, "
                         "openai:<model>, gemini:<model>)".format(spec))
    if model not in PRICES:
        raise ValueError("no price for {!r}; add it to receipt_bench.pricing.PRICES".format(model))
    return provider, model


def estimate_cost(spec: str, n: int) -> float:
    kind, model = parse_pipeline(spec)
    if kind == "textract":
        return n * TEXTRACT_ANALYZE_EXPENSE_PER_PAGE
    if kind == "textract+repair":
        repair = token_cost(REPAIR_MODEL, EST_REPAIR_INPUT, EST_REPAIR_OUTPUT) * EST_REPAIR_SHARE
        return n * (TEXTRACT_ANALYZE_EXPENSE_PER_PAGE + repair)
    return n * token_cost(model, EST_INPUT_TOKENS, EST_OUTPUT_TOKENS)


# ------------------------------------------------------------------ pipelines


class TextractCache:
    """Raw AnalyzeExpense responses keyed by image hash, so each page is billed once."""

    def __init__(self, directory: str, client: Any = None):
        self.dir = os.path.join(directory, "textract")
        os.makedirs(self.dir, exist_ok=True)
        self.client = client
        self.lock = threading.Lock()

    def get(self, image_bytes: bytes) -> Dict[str, Any]:
        """{"response", "latency_s", "cached": bool}; latency is from the original live call."""
        path = os.path.join(self.dir, hashlib.sha256(image_bytes).hexdigest() + ".json")
        if os.path.exists(path):
            with open(path) as f:
                return dict(json.load(f), cached=True)
        if self.client is None:
            with self.lock:
                if self.client is None:
                    import boto3

                    self.client = boto3.client("textract", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        # same shrinking the LLM pipelines get: AnalyzeExpense rejects images over
        # 5 MB sent as bytes (a few CORD PNGs are 10-12 MB)
        data, _ = prepare_image(image_bytes)
        started = time.time()
        response = textract.analyze_expense(data, client=self.client)
        entry = {"response": response, "latency_s": round(time.time() - started, 2)}
        with open(path + ".part", "w") as f:
            json.dump(entry, f, default=str)
        os.replace(path + ".part", path)
        return dict(entry, cached=False)


def _from_llm(result: LLMResult) -> Dict[str, Any]:
    receipt = None
    if result.out is not None:
        receipt = claude.to_receipt(result.out)
        receipt["tax_included"] = bool(result.out.get("tax_included"))
    return {
        "receipt": receipt,
        "latency_s": result.latency_s,
        "cost_usd": token_cost(result.model, result.input_tokens, result.output_tokens),
        "error": result.error,
        "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
        "stop_reason": result.stop_reason,
        "settings": dict(result.settings, model=result.model),
        "model_notes": (result.out or {}).get("notes", ""),
    }


def make_pipeline(spec: str, cache: TextractCache,
                  extractors: Optional[Dict[str, Callable[..., LLMResult]]] = None,
                  repair_fn: Callable[..., Any] = claude.repair) -> Callable[[bytes], Dict[str, Any]]:
    kind, model = parse_pipeline(spec)

    if kind == "textract":
        def run_textract(image: bytes) -> Dict[str, Any]:
            hit = cache.get(image)
            receipt = textract.parse_expense(hit["response"])
            return {"receipt": receipt, "latency_s": hit["latency_s"],
                    "cost_usd": TEXTRACT_ANALYZE_EXPENSE_PER_PAGE, "error": None,
                    "textract_cached": hit["cached"]}
        return run_textract

    if kind == "textract+repair":
        def run_repair(image: bytes) -> Dict[str, Any]:
            hit = cache.get(image)
            first = textract.parse_expense(hit["response"])
            out = {"receipt": first, "latency_s": hit["latency_s"], "cost_usd": TEXTRACT_ANALYZE_EXPENSE_PER_PAGE,
                   "error": None, "used_fallback": False, "textract_cached": hit["cached"]}
            if not totals_mismatch(first):
                return out
            fix = repair_fn(image, first, model=REPAIR_MODEL, effort=REPAIR_EFFORT)
            usage = fix.usage or {}
            in_tokens = (usage.get("input_tokens") or 0) + (usage.get("cache_read_input_tokens") or 0) \
                + (usage.get("cache_creation_input_tokens") or 0)
            out_tokens = usage.get("output_tokens") or 0
            out.update(
                used_fallback=True,
                fallback_reconciled=fix.reconciled,
                fallback_error=fix.error,
                latency_s=round(hit["latency_s"] + fix.latency_s, 2),
                cost_usd=TEXTRACT_ANALYZE_EXPENSE_PER_PAGE + token_cost(REPAIR_MODEL, in_tokens, out_tokens),
                usage={"input_tokens": in_tokens, "output_tokens": out_tokens},
                settings={"repair_model": REPAIR_MODEL, "repair_effort": REPAIR_EFFORT},
            )
            # production keeps Claude's reading only if it passes the same check
            if fix.reconciled and fix.receipt is not None:
                out["receipt"] = fix.receipt
            return out
        return run_repair

    if extractors and kind in extractors:
        extract = extractors[kind]
    elif kind == "claude":
        from .providers.anthropic_api import extract
    elif kind == "openai":
        from .providers.openai_api import extract
    else:
        from .providers.gemini_api import extract

    def run_llm(image: bytes) -> Dict[str, Any]:
        return _from_llm(extract(image, model))
    return run_llm


# ------------------------------------------------------------------ runner


def run(spec: str, dataset_dir: str, runs_dir: str = "runs", cache_dir: str = "cache",
        limit: Optional[int] = None, concurrency: int = 4, max_cost: Optional[float] = None,
        retry_errors: bool = False, pipeline: Optional[Callable[[bytes], Dict[str, Any]]] = None,
        log: Callable[[str], None] = print) -> Dict[str, Any]:
    out_dir = os.path.join(runs_dir, slug(spec), dataset_name(dataset_dir))
    os.makedirs(out_dir, exist_ok=True)
    labels = list(load_labels(dataset_dir))[:limit]

    todo = []
    for record in labels:
        path = os.path.join(out_dir, record["id"] + ".json")
        if os.path.exists(path):
            if not retry_errors:
                continue
            with open(path) as f:
                if json.load(f).get("error") is None:
                    continue
        todo.append(record)

    pipeline = pipeline or make_pipeline(spec, TextractCache(cache_dir))
    spent, done, failed, stopped = 0.0, 0, 0, False
    lock = threading.Lock()

    def one(record: Dict[str, Any]) -> Dict[str, Any]:
        with open(record["image"], "rb") as f:
            image = f.read()
        try:
            result = pipeline(image)
        except Exception as e:  # a crash on one receipt is recorded, not fatal
            result = {"receipt": None, "latency_s": None, "cost_usd": 0.0, "error": "{}: {}".format(type(e).__name__, e)}
        result.update(id=record["id"], pipeline=spec, created_at=datetime.now(timezone.utc).isoformat())
        path = os.path.join(out_dir, record["id"] + ".json")
        with open(path + ".part", "w") as f:
            json.dump(result, f, indent=1, ensure_ascii=False, default=str)
        os.replace(path + ".part", path)
        return result

    log("{} on {}: {} receipts to run ({} already done)".format(spec, dataset_name(dataset_dir), len(todo),
                                                               len(labels) - len(todo)))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = set()
        queue = list(todo)
        while queue or pending:
            while queue and len(pending) < concurrency and not stopped:
                with lock:
                    if max_cost is not None and spent >= max_cost:
                        stopped = True
                        break
                pending.add(pool.submit(one, queue.pop(0)))
            if not pending:
                break
            finished = next(as_completed(pending))
            pending.discard(finished)
            result = finished.result()
            with lock:
                spent += result.get("cost_usd") or 0.0
                done += 1
                failed += result.get("error") is not None
            if done % 10 == 0 or not (queue or pending):
                log("  {}/{} done, {} errors, ${:.3f} spent".format(done, len(todo), failed, spent))
    if stopped:
        log("  stopped: spend reached --max-cost ${:.2f}; rerun to resume".format(max_cost))
    return {"pipeline": spec, "dataset": dataset_dir, "out_dir": out_dir, "ran": done, "errors": failed,
            "cost_usd": round(spent, 4), "stopped_at_max_cost": stopped}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run extraction pipelines over benchmark datasets.")
    parser.add_argument("--pipeline", action="append", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--limit", type=int, help="only the first N receipts of each dataset")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-cost", type=float, help="stop each pipeline/dataset run once its spend passes this (USD)")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    loaded = load_dotenv()
    if loaded:
        print("loaded from .env:", ", ".join(sorted(loaded)))
    for spec in args.pipeline:
        parse_pipeline(spec)  # fail fast on typos

    if args.dry_run:
        total = 0.0
        for spec in args.pipeline:
            for ds in args.dataset:
                n = len(list(load_labels(ds))[:args.limit])
                est = estimate_cost(spec, n)
                total += est
                print("{:<32} {:<12} {:>4} receipts  ~${:.2f}".format(spec, dataset_name(ds), n, est))
        print("estimated total ~${:.2f} (before skipping already-finished receipts)".format(total))
        return

    summaries = [run(spec, ds, runs_dir=args.runs_dir, cache_dir=args.cache_dir, limit=args.limit,
                     concurrency=args.concurrency, max_cost=args.max_cost, retry_errors=args.retry_errors)
                 for spec in args.pipeline for ds in args.dataset]
    print("\ntotal spent ${:.3f}".format(sum(s["cost_usd"] for s in summaries)))


if __name__ == "__main__":
    main()
