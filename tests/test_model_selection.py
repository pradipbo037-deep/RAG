from core.model_selector import pick_best_model


def test_pick_qwen_preferred_over_llama():
    models = ["llama3.2:latest", "qwen2.5:latest"]
    result = pick_best_model(models)
    assert result == "qwen2.5:latest"


def test_pick_llama_when_qwen_missing():
    models = ["llama3.1:8b"]
    result = pick_best_model(models)
    assert result == "llama3.1:8b"


def test_pick_model_returns_none_if_missing():
    assert pick_best_model([]) is None
