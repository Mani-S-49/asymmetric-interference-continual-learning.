from contextlib import contextmanager
from types import SimpleNamespace

import torch

from phase2.similarity import EmbeddingSimilarityReplayTrainer


class _TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 2)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
                [4.0, 0.0],
                [0.0, 3.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [1.0, 2.0],
            ]))

    def forward(self, input_ids, attention_mask, use_cache):
        return SimpleNamespace(
            last_hidden_state=self.embedding(input_ids)
        )


class _TinyPeftModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _TinyBackbone()
        self.adapters_disabled = False

    def get_base_model(self):
        return SimpleNamespace(model=self.backbone)

    def get_input_embeddings(self):
        return self.backbone.embedding

    @contextmanager
    def disable_adapter(self):
        self.adapters_disabled = True
        try:
            yield
        finally:
            self.adapters_disabled = False


class _TinyTokenizer:
    def __call__(self, prompts, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2], [3, 0]]),
            "attention_mask": torch.tensor([[1, 1], [1, 0]]),
        }


class _TinyTrainer:
    model_name = "tiny-test"
    max_length = 8

    def __init__(self):
        self.model = _TinyPeftModel()
        self.tokenizer = _TinyTokenizer()


def test_encoder_uses_masked_mean_and_restores_model_state():
    base_trainer = _TinyTrainer()
    base_trainer.model.train()
    trainer = EmbeddingSimilarityReplayTrainer(
        base_trainer,
        capacity=2,
        replay_seed=1,
        training_seed=42,
        embedding_batch_size=2,
    )

    embeddings = trainer._encode([{"prompt": "first"}, {"prompt": "second"}])

    assert len(embeddings) == 2
    assert embeddings[0] == (0.4472135901451111, 0.8944271802902222)
    assert embeddings[1] == (1.0, 0.0)
    assert base_trainer.model.training
    assert not base_trainer.model.adapters_disabled
