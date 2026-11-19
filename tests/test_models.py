"""Per-role models: Lab Ops can use a faster model than the planner/coder, and falls back to executor."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.models import resolve  # noqa: E402


def env(monkeypatch, **kv):
    for k in ("LLM_MODEL_LAB_OPS", "LLM_BASE_URL_LAB_OPS", "LLM_BASE_URL_EXECUTOR", "LLM_BASE_URL_PLANNER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "http://m:8009/v1")
    monkeypatch.setenv("LLM_MODEL_EXECUTOR", "muse-spark-open")
    monkeypatch.setenv("LLM_MODEL_PLANNER", "muse-spark-open")
    for k, v in kv.items():
        monkeypatch.setenv(k, v)


def test_lab_ops_falls_back_to_executor(monkeypatch):
    env(monkeypatch)
    assert resolve("lab_ops") == ("muse-spark-open", "http://m:8009/v1")


def test_lab_ops_own_model_and_url(monkeypatch):
    env(monkeypatch, LLM_MODEL_LAB_OPS="gemma-4-31B-it", LLM_BASE_URL_LAB_OPS="http://m:8003/v1")
    assert resolve("lab_ops") == ("gemma-4-31B-it", "http://m:8003/v1")
    assert resolve("executor") == ("muse-spark-open", "http://m:8009/v1")
    assert resolve("planner") == ("muse-spark-open", "http://m:8009/v1")


def test_lab_ops_own_model_shared_url(monkeypatch):
    env(monkeypatch, LLM_MODEL_LAB_OPS="other")
    assert resolve("lab_ops") == ("other", "http://m:8009/v1")
