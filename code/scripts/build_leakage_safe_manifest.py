#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from audit_dataset_split import rows_from_manifest, sample_sha256


def fallback_group_key(row: dict[str, Any]) -> str:
    file_name = str(row.get("file_name") or Path(str(row.get("path", ""))).name).lower()
    return f"filename:{file_name}"


def build(manifest: dict[str, Any], seed: int, test_fraction: float) -> dict[str, Any]:
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    key_sources = defaultdict(int)
    for _old_split, label, row in rows_from_manifest(manifest):
        digest, source = sample_sha256(row, rehash_existing=False)
        key = f"sha256:{digest}" if digest else fallback_group_key(row)
        groups[key].append((label, row))
        key_sources[source if digest else "filename_fallback"] += 1

    canonical_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    duplicate_rows_removed = 0
    for key, members in groups.items():
        labels = {label for label, _row in members}
        if len(labels) != 1:
            raise ValueError(f"Conflicting labels for group {key}: {sorted(labels)}")
        label = next(iter(labels))
        canonical = dict(members[0][1])
        canonical["split_group_key"] = key
        canonical_by_label[label].append(canonical)
        duplicate_rows_removed += len(members) - 1

    rng = random.Random(seed)
    output = {"train": defaultdict(list), "test": defaultdict(list)}
    for label, rows in sorted(canonical_by_label.items()):
        rows = sorted(rows, key=lambda row: row["split_group_key"])
        rng.shuffle(rows)
        test_count = round(len(rows) * test_fraction)
        test_keys = {row["split_group_key"] for row in rows[:test_count]}
        for row in rows:
            split = "test" if row["split_group_key"] in test_keys else "train"
            output[split][label].append(row)

    train = {label: rows for label, rows in output["train"].items()}
    test = {label: rows for label, rows in output["test"].items()}
    return {
        "config": {
            "split_policy": "sha256_or_filename_grouped_stratified",
            "seed": seed,
            "test_fraction": test_fraction,
            "warning": (
                "filename fallback is an exact-name grouping proxy, not a substitute for "
                "content hashing or package/version metadata"
            ),
        },
        "summary": {
            "source_rows": sum(len(rows) for rows in manifest["train"].values())
            + sum(len(rows) for rows in manifest["test"].values()),
            "unique_groups": len(groups),
            "duplicate_rows_removed": duplicate_rows_removed,
            "group_key_source_counts": dict(key_sources),
            "train_total": sum(len(rows) for rows in train.values()),
            "test_total": sum(len(rows) for rows in test.values()),
            "train": {label: len(rows) for label, rows in train.items()},
            "test": {label: len(rows) for label, rows in test.items()},
        },
        "train": train,
        "test": test,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be between zero and one")

    result = build(json.loads(args.manifest.read_text()), args.seed, args.test_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
