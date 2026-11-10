import json

from phase2.run_baselines import _dataset_from_jsonl, format_xnli_example


def test_format_xnli_example_uses_phase1_prompt_template():
    example = {"premise": "A cat sat.", "hypothesis": "A cat was sitting.", "label": 0}

    formatted = format_xnli_example(example)

    assert formatted["target"] == " entailment"
    assert "Premise: A cat sat." in formatted["prompt"]
    assert "Hypothesis: A cat was sitting." in formatted["prompt"]


def test_dataset_from_jsonl_creates_prompt_target_columns(tmp_path):
    dataset_path = tmp_path / "mini_en.train.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"example": {"premise": "p1", "hypothesis": "h1", "label": 0}}),
                json.dumps({"example": {"premise": "p2", "hypothesis": "h2", "label": 2}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = _dataset_from_jsonl(dataset_path, limit=2)

    assert len(dataset) == 2
    assert set(dataset.column_names) == {"prompt", "target"}
    assert dataset[0]["target"] == " entailment"
    assert dataset[1]["target"] == " contradiction"
