"""
Benchmark datasets.

Every dataset is written to disk in the same layout, so the scorer doesn't care
where a receipt came from:

    <dir>/labels/<id>.json   {"id", "source", "image", "receipt", "labelled_fields", ...}
    <dir>/images/<id>.<ext>  ("image" in the label is relative to <dir>)

`labelled_fields` lists the Receipt fields the source actually labels; the scorer
only grades those (CORD, for example, has no vendor or date labels).
"""
import json
import os
from typing import Any, Dict, Iterator


def load_labels(directory: str, include_excluded: bool = False) -> Iterator[Dict[str, Any]]:
    """Yield label records from <directory>/labels, sorted by id, with "image" made absolute."""
    labels_dir = os.path.join(directory, "labels")
    for name in sorted(os.listdir(labels_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(labels_dir, name)) as f:
            record = json.load(f)
        if not include_excluded and record.get("review", {}).get("status") == "excluded":
            continue
        record["image"] = os.path.join(directory, record["image"])
        yield record
