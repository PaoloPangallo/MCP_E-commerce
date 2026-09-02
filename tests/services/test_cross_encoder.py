from app.services.rag import cross_encoder


def test_cross_encoder_is_loaded_only_when_reranking(monkeypatch):
    constructed = []

    class FakeCrossEncoder:
        def __init__(self, model_name):
            constructed.append(model_name)

        def predict(self, pairs):
            return [0.2, 0.9]

    cross_encoder._get_model.cache_clear()
    monkeypatch.setattr(cross_encoder, "CrossEncoder", FakeCrossEncoder)

    ranked = cross_encoder.cross_rerank(
        "telefono",
        [{"title": "modello economico"}, {"title": "modello premium"}],
    )

    assert constructed == ["cross-encoder/ms-marco-MiniLM-L-6-v2"]
    assert [item["title"] for item in ranked] == ["modello premium", "modello economico"]
