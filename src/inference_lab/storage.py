"""Atomic final artifacts plus a flushed JSONL journal for partial recovery."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_parquet(path, rows):
    path = Path(path)
    temporary = path.with_suffix(".parquet.tmp")
    # Empty tables remain valid Parquet artifacts and do not imply zero observations.
    table = (
        pa.Table.from_pylist(rows)
        if rows
        else pa.table({"timestamp_ms": pa.array([], pa.float64())})
    )
    pq.write_table(table, temporary)
    temporary.replace(path)


def read_parquet(path):
    return pq.read_table(path).to_pylist()


class Journal:
    def __init__(self, path):
        self.file = Path(path).open("a")

    def append(self, row):
        self.file.write(json.dumps(row, allow_nan=False) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()
