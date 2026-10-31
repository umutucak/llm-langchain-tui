from types import SimpleNamespace

import pytest

from llmtui.models_config import EmbeddingsEntry, build_embeddings
from scripts.ingest import embed_and_insert_batches


def test_build_embeddings_has_bounded_defaults():
    entry = EmbeddingsEntry(
        name="test",
        base_url="http://localhost:8000/v1",
        api="openai-completions",
        api_key="test",
        model="embed",
        batch_size=2,
        concurrency=1,
    )

    embeddings = build_embeddings(entry)

    assert embeddings.client._client.timeout == 120.0
    assert embeddings.max_retries == 2


def test_build_embeddings_preserves_timeout_overrides():
    entry = EmbeddingsEntry(
        name="test",
        base_url="http://localhost:8000/v1",
        api="openai-completions",
        api_key="test",
        model="embed",
        batch_size=2,
        concurrency=1,
        params={"timeout": 5, "max_retries": 0},
    )

    embeddings = build_embeddings(entry)

    assert embeddings.client._client.timeout == 5
    assert embeddings.max_retries == 0


def test_embed_and_insert_batches_fails_when_worker_fails():
    calls = []

    def add_documents(batch):
        calls.append(batch)
        if batch == ["bad"]:
            raise RuntimeError("connection closed")

    store = SimpleNamespace(add_documents=add_documents)

    with pytest.raises(RuntimeError, match="index is incomplete"):
        embed_and_insert_batches(store, [["first"], ["bad"]], workers=1)

    assert calls == [["first"], ["bad"]]
