from phase2.derpp import (
    DERPPDataCollator,
    candidate_first_token_ids,
    derpp_loss,
    make_derpp_features,
    score_candidate_first_token_logits,
)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
    ):
        tokens = [ord(token[0]) % 7 + 1 for token in text.split()]
        if add_special_tokens:
            tokens = [8] + tokens
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return {"input_ids": tokens}


def _example():
    return {"prompt": "premise hypothesis", "target": " entailment"}


def test_candidate_first_token_ids_and_features_keep_distillation_data():
    tokenizer = _Tokenizer()
    candidate_ids = candidate_first_token_ids(tokenizer)
    assert len(candidate_ids) == 3

    features = make_derpp_features(
        [
            _example(),
            {**_example(), "teacher_logits": [0.2, -0.3, 0.1]},
        ],
        tokenizer,
        max_length=16,
        candidate_token_ids=candidate_ids,
    )
    assert features[0]["is_replay"] is False
    assert features[1]["is_replay"] is True
    batch = DERPPDataCollator(tokenizer)(features)
    assert batch["teacher_logits"].shape == (2, 3)
    assert batch["is_replay"].tolist() == [False, True]


def test_derpp_loss_combines_current_replay_and_candidate_distillation():
    import torch
    import torch.nn.functional as functional

    logits = torch.randn(2, 4, 6, requires_grad=True)
    labels = torch.tensor([[-100, -100, 1, 2], [-100, -100, 3, 4]])
    prompt_lengths = torch.tensor([2, 2])
    teacher_logits = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]])
    is_replay = torch.tensor([False, True])

    loss = derpp_loss(
        logits,
        labels,
        prompt_lengths,
        teacher_logits,
        is_replay,
        [1, 3, 5],
        alpha=0.5,
        beta=0.5,
    )
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    per_token = functional.cross_entropy(
        shifted_logits.reshape(-1, 6),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(2, 3)
    ce_by_example = per_token.sum(1) / shifted_labels.ne(-100).sum(1)
    student_dark_logits = logits[1, 1, [1, 3, 5]].float()
    expected = (
        ce_by_example[0]
        + 0.5 * ce_by_example[1]
        + 0.5 * functional.mse_loss(student_dark_logits, teacher_logits[1])
    )
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_derpp_loss_without_replay_is_supervised_causal_lm_loss():
    import torch

    logits = torch.randn(1, 3, 5, requires_grad=True)
    labels = torch.tensor([[-100, 2, 4]])
    loss = derpp_loss(
        logits,
        labels,
        torch.tensor([1]),
        torch.zeros(1, 3),
        torch.tensor([False]),
        [1, 2, 3],
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_teacher_score_helper_restores_model_training_state():
    import torch

    class _Model:
        training = True

        def __init__(self):
            self.embedding = type(
                "Embedding",
                (),
                {"weight": torch.zeros(1, 1)},
            )()

        def get_input_embeddings(self):
            return self.embedding

        def eval(self):
            self.training = False

        def train(self, training):
            self.training = training

        def __call__(self, *, input_ids, attention_mask, logits_to_keep):
            logits = torch.zeros(
                input_ids.shape[0],
                len(logits_to_keep),
                16,
            )
            for index in range(len(logits_to_keep)):
                logits[:, index, :] = index
            return type("Output", (), {"logits": logits})()

    model = _Model()
    scores = score_candidate_first_token_logits(
        model,
        _Tokenizer(),
        [_example()],
        max_length=16,
    )
    assert len(scores) == 1
    assert len(scores[0]) == 3
    assert model.training is True

