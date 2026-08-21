from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


restoration = load_script("run_mevgt_restoration", "scripts/run_mevgt_restoration.py")
control = load_script(
    "analyze_restoration_random_nonhotspot_control",
    "scripts/analyze_restoration_random_nonhotspot_control.py",
)
benchmark = load_script(
    "run_mevgt_node_latent_benchmark",
    "scripts/run_mevgt_node_latent_benchmark.py",
)


def test_recovery_ratio_rejects_samples_without_lost_score() -> None:
    assert restoration.recovery_ratio(0.8, 0.8, 0.9) is None
    assert restoration.recovery_ratio(0.7, 0.8, 0.9) is None
    assert restoration.recovery_ratio(0.8, 0.4, 0.6) == pytest.approx(0.5)


def test_control_uses_preselected_hotspot_not_posthoc_best() -> None:
    report = {
        "schema_version": 3,
        "restoration_mode": "independent_local_node_activation",
        "sample_count": 1,
        "samples": [
            {
                "preselected_hotspot_stage": "gnn:0",
                "hotspot_target_node_recovery_ratio": 0.4,
                "random_layer_target_node_recovery_ratio": 0.9,
                "hotspot_random_node_recovery_ratio": 0.1,
            }
        ],
    }

    summary = control.summarize(report, seed=0)

    assert summary["hotspot_target_node"]["mean"] == pytest.approx(0.4)
    assert summary["random_layer_same_target_node"]["mean"] == pytest.approx(0.9)
    assert summary["hotspot_layer_random_node"]["mean"] == pytest.approx(0.1)
    assert summary["paired_hotspot_minus_random_layer"]["mean"] == pytest.approx(-0.5)
    assert summary["hotspot_stage_counts"] == {"gnn:0": 1}


def test_control_rejects_legacy_progressive_trace() -> None:
    with pytest.raises(ValueError, match="schema-v3"):
        control.summarize({"samples": []}, seed=0)


def test_reference_sampling_is_reproducible_and_not_head_order() -> None:
    summaries = [{"reference_index": index} for index in range(100)]
    first = benchmark.select_reference_summaries(
        summaries, 12, seed=42, mode="deterministic_random"
    )
    second = benchmark.select_reference_summaries(
        summaries, 12, seed=42, mode="deterministic_random"
    )

    assert first == second
    assert first != summaries[:12]
    assert len({row["reference_index"] for row in first}) == 12


def test_random_control_excludes_every_preselected_hotspot() -> None:
    report = {
        "schema_version": 3,
        "restoration_mode": "independent_local_node_activation",
        "sample_count": 1,
        "samples": [
            {
                "preselected_hotspot_stage": "gnn:0",
                "random_nonhotspot_stage": "gnn:2",
                "hotspot_target_node_recovery_ratio": 0.4,
                "random_layer_target_node_recovery_ratio": 0.2,
                "hotspot_random_node_recovery_ratio": 0.05,
            }
        ],
    }

    summary = control.summarize(report, seed=0)
    assert summary["random_layer_same_target_node"]["mean"] == pytest.approx(0.2)


def test_shared_manifest_selection_uses_stable_sample_ids() -> None:
    summaries = [
        {"reference_index": 7, "sample_id": "a"},
        {"reference_index": 2, "sample_id": "b"},
    ]
    selected = benchmark.select_reference_summaries_from_manifest(
        summaries, {"sample_ids": ["b", "a"]}
    )
    assert [row["sample_id"] for row in selected] == ["b", "a"]
    assert [row["reference_index"] for row in selected] == [2, 7]
