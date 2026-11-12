import pytest

from phase2.lr_adjust import LRAdjustTrainer


class _BaseTrainer:
    model_name = "tiny"
    seed = 42

    def __init__(self, learning_rate=2e-4):
        self.learning_rate = learning_rate
        self.model = object()
        self.tokenizer = object()
        self.rates_seen = []

    def load_model_and_tokenizer(self):
        return None

    def train_stage(self, dataset, stage_name):
        self.rates_seen.append(self.learning_rate)
        return {"stage_name": stage_name}


def test_lr_adjust_decays_before_each_stage_and_honors_floor():
    base = _BaseTrainer()
    trainer = LRAdjustTrainer(base, gamma=0.5, minimum_learning_rate=5e-5)

    for index in range(5):
        trainer.train_stage([], f"stage_{index}_en")

    assert base.rates_seen == pytest.approx(
        [1e-4, 5e-5, 5e-5, 5e-5, 5e-5]
    )
    assert [item["learning_rate"] for item in trainer.learning_rates] == pytest.approx(
        base.rates_seen
    )


@pytest.mark.parametrize(
    ("gamma", "minimum_learning_rate"),
    [(1.0, None), (0.0, None), (0.5, 3e-4), (True, None)],
)
def test_lr_adjust_rejects_invalid_schedule(gamma, minimum_learning_rate):
    with pytest.raises(ValueError):
        LRAdjustTrainer(
            _BaseTrainer(),
            gamma=gamma,
            minimum_learning_rate=minimum_learning_rate,
        )
