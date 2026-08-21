from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch_geometric.data import Data

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_leakage_safe_pyg_dataset import (  # noqa: E402
    ArtifactRecord,
    group_records,
    parse_record,
    stratified_group_split,
)
from train_gnn import audit_group_disjoint_rows  # noqa: E402


def make_record(group_id: str, label: int, split: str, suffix: str) -> ArtifactRecord:
    label_name = "malware" if label else "benign"
    root = Path("/artifacts")
    return ArtifactRecord(
        path=root / split / label_name / f"{group_id}__{suffix}.pt",
        root=root,
        old_split=split,
        label_name=label_name,
        label=label,
        group_id=group_id,
        hash_algorithm="sha256" if len(group_id) == 64 else "md5",
    )


def test_parser_accepts_32_and_64_hex_source_hashes() -> None:
    root = Path("/artifacts")
    for group_id, algorithm in (("a" * 32, "md5"), ("b" * 64, "sha256")):
        row = parse_record(root / "train" / "benign" / f"{group_id}__1234.pt", root)
        assert row.group_id == group_id
        assert row.hash_algorithm == algorithm


def test_split_is_group_disjoint_and_deterministic() -> None:
    rows = []
    for label in (0, 1):
        for index in range(10):
            group_id = f"{label}{index:031x}"
            rows.append(make_record(group_id, label, "train", f"a{index}"))
            if index < 3:
                rows.append(make_record(group_id, label, "test", f"b{index}"))
    grouped = group_records(rows)
    first = stratified_group_split(grouped, test_ratio=0.2, seed=42)
    second = stratified_group_split(grouped, test_ratio=0.2, seed=42)
    assert first == second
    assert sum(split == "test" for split in first.values()) == 4
    assert set(first) == set(grouped)


def test_label_conflict_fails_closed() -> None:
    group_id = "c" * 64
    with pytest.raises(RuntimeError, match="label conflicts"):
        group_records(
            [
                make_record(group_id, 0, "train", "one"),
                make_record(group_id, 1, "test", "two"),
            ]
        )


def data_row(group_id: str) -> Data:
    row = Data(x=torch.ones((1, 2)), edge_index=torch.empty((2, 0), dtype=torch.long))
    row.leakage_group_id = group_id
    return row


def test_serialized_dataset_audit_fails_on_cross_split_overlap() -> None:
    with pytest.raises(ValueError, match="cross_split_overlap=1"):
        audit_group_disjoint_rows([data_row("shared")], [data_row("shared")])


def test_serialized_dataset_audit_accepts_disjoint_groups() -> None:
    result = audit_group_disjoint_rows(
        [data_row("train-a"), data_row("train-b")], [data_row("test-a")]
    )
    assert result["cross_split_group_count"] == 0
