#!/usr/bin/env python3
"""Build a group-disjoint PyG dataset from one or more graph-artifact roots.

The original CICMalDroid manifest was split before duplicate APK identifiers were
removed.  This builder deliberately ignores the old split assignment: it pools
all artifacts, groups them by the source-file hash embedded in the artifact file
name, keeps one canonical artifact per group, and only then performs a seeded,
label-stratified train/test split.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Iterable

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_pyg_dataset import build_data_object, validate_data_object  # noqa: E402


SOURCE_HASH_RE = re.compile(r"^(?P<source_hash>[0-9a-fA-F]{64}|[0-9a-fA-F]{32})__")


@dataclass(frozen=True)
class ArtifactRecord:
    path: Path
    root: Path
    old_split: str
    label_name: str
    label: int
    group_id: str
    hash_algorithm: str

    @property
    def portable_path(self) -> str:
        return f"{self.root.name}/{self.path.relative_to(self.root)}"


def parse_record(path: Path, root: Path) -> ArtifactRecord:
    relative = path.relative_to(root)
    if len(relative.parts) < 3:
        raise ValueError(f"Unexpected artifact path layout: {path}")
    old_split, label_name = relative.parts[0], relative.parts[1]
    if old_split not in {"train", "test"}:
        raise ValueError(f"Unexpected original split in {path}: {old_split}")
    if label_name not in {"benign", "malware"}:
        raise ValueError(f"Unexpected label directory in {path}: {label_name}")
    match = SOURCE_HASH_RE.match(path.name)
    if match is None:
        raise ValueError(f"Artifact name has no 32/64-hex source identifier: {path}")
    group_id = match.group("source_hash").lower()
    return ArtifactRecord(
        path=path,
        root=root,
        old_split=old_split,
        label_name=label_name,
        label=1 if label_name == "malware" else 0,
        group_id=group_id,
        hash_algorithm="sha256" if len(group_id) == 64 else "md5",
    )


def discover_records(roots: Iterable[Path]) -> list[ArtifactRecord]:
    records: list[ArtifactRecord] = []
    for root in sorted(path.resolve() for path in roots):
        if not root.is_dir():
            raise FileNotFoundError(f"Artifact root does not exist: {root}")
        records.extend(parse_record(path, root) for path in sorted(root.glob("*/*/*.pt")))
    if not records:
        raise RuntimeError("No graph artifacts found under the supplied roots.")
    return records


def group_records(records: Iterable[ArtifactRecord]) -> dict[str, list[ArtifactRecord]]:
    grouped: dict[str, list[ArtifactRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)
    conflicts = {
        group_id: rows
        for group_id, rows in grouped.items()
        if len({row.label for row in rows}) != 1
    }
    if conflicts:
        preview = ", ".join(sorted(conflicts)[:5])
        raise RuntimeError(f"Found {len(conflicts)} source groups with label conflicts: {preview}")
    return dict(grouped)


def stratified_group_split(
    grouped: dict[str, list[ArtifactRecord]], test_ratio: float, seed: int
) -> dict[str, str]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test-ratio must be strictly between 0 and 1")
    by_label: dict[int, list[str]] = defaultdict(list)
    for group_id, rows in grouped.items():
        by_label[rows[0].label].append(group_id)

    assignments: dict[str, str] = {}
    for label, group_ids in sorted(by_label.items()):
        shuffled = sorted(group_ids)
        random.Random(seed + label * 1_000_003).shuffle(shuffled)
        test_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * test_ratio)))
        test_ids = set(shuffled[:test_count])
        assignments.update(
            {group_id: ("test" if group_id in test_ids else "train") for group_id in shuffled}
        )
    return assignments


def choose_canonical(rows: list[ArtifactRecord]) -> ArtifactRecord:
    # Prefer an old-train artifact only to make selection stable if shard naming changes.
    # The new split assignment is made independently and does not use this preference.
    return min(rows, key=lambda row: (row.old_split != "train", row.portable_path))


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_split(
    split: str,
    group_ids: list[str],
    grouped: dict[str, list[ArtifactRecord]],
    output_root: Path,
    expected_feature_dim: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    dataset = []
    rejected: list[dict[str, object]] = []
    label_counts: Counter[str] = Counter()
    node_total = 0
    edge_total = 0

    for index, group_id in enumerate(group_ids, start=1):
        record = choose_canonical(grouped[group_id])
        artifact = torch.load(record.path, map_location="cpu", weights_only=False)
        artifact_label = int(artifact["label"])
        if artifact_label != record.label:
            raise RuntimeError(
                f"Label mismatch between path and payload for {record.portable_path}: "
                f"{record.label} != {artifact_label}"
            )
        data = build_data_object(artifact)
        reason = validate_data_object(data, expected_feature_dim)
        if reason is not None:
            rejected.append(
                {
                    "split": split,
                    "group_id": group_id,
                    "source_artifact": record.portable_path,
                    "reason": reason,
                }
            )
            continue

        # These fields make the split independently auditable from the serialized
        # dataset.  Do not retain machine-specific absolute source paths.
        data.leakage_group_id = group_id
        data.source_hash_algorithm = record.hash_algorithm
        data.source_artifact = record.portable_path
        data.original_split = record.old_split
        data.sample_path = ""
        dataset.append(data)
        label_counts[record.label_name] += 1
        node_total += int(data.num_nodes_original)
        edge_total += int(data.num_edges_original)
        if index % 1000 == 0:
            print(f"{split}: loaded {index}/{len(group_ids)} groups", flush=True)

    output_path = output_root / f"{split}_dataset.pt"
    temporary = output_path.with_suffix(".pt.tmp")
    torch.save(dataset, temporary)
    temporary.replace(output_path)
    count = len(dataset)
    stats = {
        "count": count,
        "rejected": len(rejected),
        "label_counts": dict(sorted(label_counts.items())),
        "feature_dim": expected_feature_dim,
        "avg_nodes": node_total / count if count else 0.0,
        "avg_edges": edge_total / count if count else 0.0,
        "path": output_path.name,
        "sha256": file_sha256(output_path),
        "size_bytes": output_path.stat().st_size,
    }
    print(f"{split}: wrote {count} graphs to {output_path} (rejected={len(rejected)})")
    return stats, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--feature-dim", type=int, default=768)
    args = parser.parse_args()

    records = discover_records(args.artifact_root)
    grouped = group_records(records)
    assignments = stratified_group_split(grouped, args.test_ratio, args.seed)
    canonical = {group_id: choose_canonical(rows) for group_id, rows in grouped.items()}

    train_ids = sorted(group_id for group_id, split in assignments.items() if split == "train")
    test_ids = sorted(group_id for group_id, split in assignments.items() if split == "test")
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise AssertionError(f"Internal error: {len(overlap)} groups cross the new split")

    old_cross_split = sum(
        len({row.old_split for row in rows}) > 1 for rows in grouped.values()
    )
    manifest_rows = [
        {
            "group_id": group_id,
            "hash_algorithm": canonical[group_id].hash_algorithm,
            "label": canonical[group_id].label,
            "label_name": canonical[group_id].label_name,
            "split": assignments[group_id],
            "source_artifact": canonical[group_id].portable_path,
            "original_splits": sorted({row.old_split for row in grouped[group_id]}),
            "artifact_copies": len(grouped[group_id]),
        }
        for group_id in sorted(grouped)
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "split_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "policy": "pool_all_artifacts_then_group_by_source_hash_then_stratified_split",
            "seed": args.seed,
            "test_ratio": args.test_ratio,
            "artifact_roots": [path.name for path in sorted(args.artifact_root)],
            "rows": manifest_rows,
        },
    )

    split_stats = {}
    all_rejected: list[dict[str, object]] = []
    for split, group_ids in (("train", train_ids), ("test", test_ids)):
        split_stats[split], rejected = build_split(
            split, group_ids, grouped, args.output_root, args.feature_dim
        )
        all_rejected.extend(rejected)

    rejected_path = args.output_root / "rejected_graphs.jsonl"
    rejected_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rejected),
        encoding="utf-8",
    )
    stats = {
        "schema_version": 1,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "input_artifact_count": len(records),
        "unique_group_count": len(grouped),
        "duplicate_artifacts_removed": len(records) - len(grouped),
        "duplicate_group_count": sum(len(rows) > 1 for rows in grouped.values()),
        "old_cross_split_group_count": old_cross_split,
        "label_conflict_group_count": 0,
        "new_cross_split_group_count": len(overlap),
        "source_hash_algorithms": dict(
            sorted(Counter(row.hash_algorithm for row in canonical.values()).items())
        ),
        "manifest": {
            "path": manifest_path.name,
            "sha256": file_sha256(manifest_path),
        },
        "splits": split_stats,
        "total_rejected": len(all_rejected),
    }
    write_json(args.output_root / "stats.json", stats)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
