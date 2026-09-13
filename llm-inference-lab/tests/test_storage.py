import csv

from llm_lab.storage import append_result


def test_append_result_adds_rows_and_new_columns(tmp_path):
    path = tmp_path / "results.csv"
    append_result({"model": "a", "speed": 1}, path)
    append_result({"model": "b", "speed": 2, "memory": 3}, path)
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[1]["memory"] == "3"

