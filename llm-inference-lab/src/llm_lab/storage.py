from __future__ import annotations

import csv
import json
from pathlib import Path


def append_result(result: dict, csv_path: str | Path) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {k: v for k, v in result.items() if k not in {"response", "token_latencies_ms"}}
    existing_rows: list[dict] = []
    fieldnames = list(row)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            fieldnames = list(dict.fromkeys([*(reader.fieldnames or []), *fieldnames]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(row)


def save_run_json(result: dict, directory: str | Path) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    stamp = result["timestamp_utc"].replace(":", "-")
    output = path / f"run_{stamp}_{result['size']}_{result['precision']}.json"
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return output

