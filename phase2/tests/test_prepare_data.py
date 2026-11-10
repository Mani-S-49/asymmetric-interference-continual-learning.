import json

from phase2.prepare_data import write_jsonl_records


def test_write_jsonl_records_preserves_indices_and_hash(tmp_path):
    rows = [
        {"premise": "p0", "hypothesis": "h0", "label": 0},
        {"premise": "p1", "hypothesis": "h1", "label": 2},
    ]
    path = tmp_path / "en.train.jsonl"

    result = write_jsonl_records(
        rows,
        [1, 0],
        language="en",
        split="train",
        output_path=path,
    )

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["source_index"] for record in records] == [1, 0]
    assert records[0]["example"]["premise"] == "p1"
    assert result["rows"] == 2
    assert len(result["sha256"]) == 64