import requests

from core.ollama_client import generate_answer


def test_generate_answer_reports_timeout_with_model_name(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.Timeout("Read timed out")

    monkeypatch.setattr(requests, "post", fake_post)

    result = generate_answer("What is soil texture?", "Soil texture is determined by the mix of sand, silt, and clay.", "qwen2.5:latest")

    assert "qwen2.5:latest" in result
    assert "timed out" in result.lower()
    assert "ollama pull" in result.lower()


def test_generate_answer_falls_back_to_general_knowledge_when_context_is_missing(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "Paris is the capital of France."}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    result = generate_answer("What is the capital of France?", "", "qwen2.5:latest")

    assert result == "Paris is the capital of France."
    assert "general knowledge" in captured["json"]["prompt"].lower()
    assert "question" in captured["json"]["prompt"].lower()
