import json
import os
from types import SimpleNamespace

import pytest

from receipt_bench import run as rb
from receipt_bench.claude import RepairResult
from receipt_bench.pricing import token_cost
from receipt_bench.providers import LLMResult

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def field(t, v):
    return {"Type": {"Text": t}, "ValueDetection": {"Text": v}}


def expense(total):
    return {"ExpenseDocuments": [{
        "SummaryFields": [field("SUBTOTAL", "20.00"), field("TAX", "1.60"), field("TOTAL", total)],
        "LineItemGroups": [{"LineItems": [{"LineItemExpenseFields": [field("ITEM", "Pizza"), field("PRICE", "20.00")]}]}],
    }]}


class FakeTextract:
    def __init__(self, total="21.60"):
        self.calls, self.total = 0, total

    def analyze_expense(self, Document):
        self.calls += 1
        return expense(self.total)


READING = {
    "items": [{"description": "Pizza", "quantity": None, "unit_price": None, "total_amount": 20.0}],
    "total_net": 20.0, "total_tax": 1.6, "tax_breakdown": [], "discounts": [], "tip": 0.0, "total_amount": 21.6,
    "supplier_name": "Joe's", "supplier_address": "", "date": "", "time": "", "currency": "USD",
    "adds_up": True, "notes": "", "tax_included": False,
}


def make_dataset(tmp_path, n=3):
    ds = tmp_path / "data" / "mini"
    (ds / "labels").mkdir(parents=True)
    (ds / "images").mkdir()
    for i in range(n):
        rid = "r{}".format(i)
        (ds / "images" / (rid + ".png")).write_bytes(PNG + bytes([i]))
        (ds / "labels" / (rid + ".json")).write_text(json.dumps({"id": rid, "image": "images/{}.png".format(rid), "receipt": {}}))
    return str(ds)


# ------------------------------------------------------------------ helpers


def test_load_dotenv_never_overrides(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nexport RB_A='one'\nRB_B=\"two\"\nRB_C=three\n\nnot a line\n")
    monkeypatch.setenv("RB_C", "already")
    monkeypatch.delenv("RB_A", raising=False)
    monkeypatch.delenv("RB_B", raising=False)
    assert sorted(rb.load_dotenv(str(env))) == ["RB_A", "RB_B"]
    assert (os.environ["RB_A"], os.environ["RB_B"], os.environ["RB_C"]) == ("one", "two", "already")
    for k in ("RB_A", "RB_B"):
        monkeypatch.delenv(k)


def test_names_and_specs():
    assert rb.dataset_name("data/own") == "own"
    assert rb.dataset_name("data/cord/test") == "cord-test"
    assert rb.parse_pipeline("openai:gpt-6.2-sol") == ("openai", "gpt-6.2-sol")
    with pytest.raises(ValueError):
        rb.parse_pipeline("mistral:large")
    with pytest.raises(ValueError):
        rb.parse_pipeline("claude:claude-unknown")
    assert rb.estimate_cost("textract", 100) == pytest.approx(1.0)


# ------------------------------------------------------------------ pipelines


def test_textract_is_billed_once_across_pipelines(tmp_path):
    client = FakeTextract()
    cache = rb.TextractCache(str(tmp_path / "cache"), client=client)
    first = rb.make_pipeline("textract", cache)(PNG)
    second = rb.make_pipeline("textract+repair", cache)(PNG)
    assert client.calls == 1
    assert first["textract_cached"] is False and second["textract_cached"] is True
    assert second["used_fallback"] is False  # Textract added up
    assert first["receipt"]["total_amount"] == 21.6


def test_repair_runs_only_on_a_mismatch_and_is_kept_only_if_it_reconciles(tmp_path):
    cache = rb.TextractCache(str(tmp_path / "cache"), client=FakeTextract(total="27.60"))
    fixed = dict(READING)
    good = RepairResult(receipt=dict(fixed, items=[{"description": "Pizza", "quantity": 1.0, "unit_price": 20.0,
                                                    "total_amount": 20.0}]),
                        reconciled=True, latency_s=5.0, usage={"input_tokens": 7000, "output_tokens": 1000})
    out = rb.make_pipeline("textract+repair", cache, repair_fn=lambda *a, **k: good)(PNG)
    assert out["used_fallback"] and out["receipt"]["total_amount"] == 21.6
    assert out["cost_usd"] == pytest.approx(0.01 + token_cost("claude-opus-5-5", 7000, 1000))

    bad = RepairResult(receipt=None, reconciled=False, latency_s=1.0, error="bad_json", usage={})
    out = rb.make_pipeline("textract+repair", cache, repair_fn=lambda *a, **k: bad)(PNG)
    assert out["used_fallback"] and out["receipt"]["total_amount"] == 27.6 and out["fallback_error"] == "bad_json"


def test_llm_pipeline_converts_the_reading_and_prices_it(tmp_path):
    reading = dict(READING, tax_included=True)
    fake = lambda image, model: LLMResult(model=model, out=reading, input_tokens=5000, output_tokens=800, latency_s=3.2)
    out = rb.make_pipeline("gemini:gemini-3.8-flash", rb.TextractCache(str(tmp_path)), extractors={"gemini": fake})(PNG)
    assert out["receipt"]["tax_included"] is True and out["receipt"]["total_amount"] == 21.6
    assert out["cost_usd"] == pytest.approx(token_cost("gemini-3.8-flash", 5000, 800))
    assert out["latency_s"] == 3.2 and out["error"] is None


# ------------------------------------------------------------------ runner


def test_run_writes_resumes_and_records_errors(tmp_path):
    ds = make_dataset(tmp_path)
    calls = []

    def pipeline(image):
        calls.append(image)
        if image.endswith(b"\x01"):
            raise RuntimeError("boom")
        return {"receipt": READING, "latency_s": 1.0, "cost_usd": 0.01, "error": None}

    runs = str(tmp_path / "runs")
    summary = rb.run("claude:claude-haiku-5-5", ds, runs_dir=runs, pipeline=pipeline, concurrency=2, log=lambda m: None)
    assert (summary["ran"], summary["errors"]) == (3, 1)
    out_dir = os.path.join(runs, "claude-claude-haiku-5-5", "mini")
    assert summary["out_dir"] == out_dir
    saved = json.load(open(os.path.join(out_dir, "r1.json")))
    assert saved["receipt"] is None and saved["error"].startswith("RuntimeError")

    rb.run("claude:claude-haiku-5-5", ds, runs_dir=runs, pipeline=pipeline, log=lambda m: None)
    assert len(calls) == 3  # everything already done
    rb.run("claude:claude-haiku-5-5", ds, runs_dir=runs, pipeline=pipeline, retry_errors=True, log=lambda m: None)
    assert len(calls) == 4  # only the failed one again


def test_max_cost_stops_new_submissions(tmp_path):
    ds = make_dataset(tmp_path, n=6)
    pipeline = lambda image: {"receipt": READING, "latency_s": 1.0, "cost_usd": 1.0, "error": None}
    summary = rb.run("claude:claude-haiku-5-5", ds, runs_dir=str(tmp_path / "runs"), pipeline=pipeline,
                     concurrency=1, max_cost=2.0, log=lambda m: None)
    assert summary["stopped_at_max_cost"] and summary["ran"] == 2


# ------------------------------------------------------------------ providers (fake clients)


def test_anthropic_provider(monkeypatch):
    from receipt_bench.providers import anthropic_api
    sent = {}

    def create(**kw):
        sent.update(kw)
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(READING))],
                               usage=SimpleNamespace(input_tokens=4000, output_tokens=600,
                                                     cache_read_input_tokens=None, cache_creation_input_tokens=None))
    result = anthropic_api.extract(PNG, "claude-haiku-5-5", client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert result.out["total_amount"] == 21.6 and (result.input_tokens, result.output_tokens) == (4000, 600)
    assert "tax_included" in sent["output_config"]["format"]["schema"]["required"]
    assert "fallbacks" not in sent


def test_openai_provider_incomplete_is_an_error():
    from receipt_bench.providers import openai_api
    sent = {}

    def create(**kw):
        sent.update(kw)
        return SimpleNamespace(status="incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                               output_text="", usage=SimpleNamespace(input_tokens=3000, output_tokens=16000))
    result = openai_api.extract(PNG, "gpt-6.2-sol", client=SimpleNamespace(responses=SimpleNamespace(create=create)))
    assert result.out is None and result.error == "status: incomplete (max_output_tokens)"
    assert sent["text"]["format"]["strict"] is True
    assert sent["input"][0]["content"][0]["image_url"].startswith("data:image/png;base64,")


def test_gemini_provider_counts_thinking_as_output():
    from receipt_bench.providers import gemini_api
    response = SimpleNamespace(
        text=json.dumps(READING),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
        usage_metadata=SimpleNamespace(prompt_token_count=1300, candidates_token_count=500, thoughts_token_count=900),
    )
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: response))
    result = gemini_api.extract(PNG, "gemini-3.8-flash", client=client)
    assert result.out["supplier_name"] == "Joe's" and (result.input_tokens, result.output_tokens) == (1300, 1400)
