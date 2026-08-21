from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregate_script = load_script(
    "aggregate_apk_validation_gates", "scripts/aggregate_apk_validation_gates.py"
)
case_script = load_script("run_apk_validation_case", "scripts/run_apk_validation_case.py")
ioc_script = load_script("apk_ioc", "scripts/apk_ioc.py")


def evidence_file(tmp_path: Path, name: str, passed: bool) -> str:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"pass": passed}), encoding="utf-8")
    return str(path)


def row(tmp_path: Path, operator: str, sample_id: str, *, detector_flip: bool = True):
    values = {
        "edit_applied_pass": True,
        "repack_pass": True,
        "art_install_pass": True,
        "smoke_pass": True,
        "behavior_ioc_pass": True,
        "detector_flip": detector_flip,
    }
    evidence = {
        key: evidence_file(tmp_path, f"{operator}-{sample_id}-{key}", value)
        for key, value in values.items()
    }
    detector_path = Path(evidence["detector_flip"])
    detector_path.write_text(json.dumps({
        "pass": detector_flip,
        "detector": {
            "checkpoints": [
                {
                    "seed": seed,
                    "original_malware": True,
                    "modified_malware": not detector_flip,
                }
                for seed in (42, 43, 44)
            ]
        },
    }), encoding="utf-8")
    return {
        "operator": operator,
        "sample_id": sample_id,
        **values,
        "evidence": evidence,
    }


def test_aggregator_checks_evidence_contents_not_just_paths(tmp_path: Path) -> None:
    candidate = row(tmp_path, "carrier_rewrite", "a")
    bad_path = Path(candidate["evidence"]["smoke_pass"])
    bad_path.write_text(json.dumps({"pass": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="Evidence/result mismatch"):
        aggregate_script.validate_row(candidate, root=ROOT)


def test_aggregator_can_require_budgeted_detector_evidence(tmp_path: Path) -> None:
    candidate = row(tmp_path, "carrier_rewrite", "budgeted")
    detector_path = Path(candidate["evidence"]["detector_flip"])
    payload = json.loads(detector_path.read_text())
    payload["detector"]["per_seed_timeout_seconds"] = 300
    for checkpoint in payload["detector"]["checkpoints"]:
        checkpoint["timeout_seconds"] = 300
        checkpoint["elapsed_seconds"] = 1.25
    detector_path.write_text(json.dumps(payload), encoding="utf-8")
    validated = aggregate_script.validate_row(
        candidate, root=ROOT, required_score_timeout=300
    )
    assert validated["sample_id"] == "budgeted"

    del payload["detector"]["checkpoints"][0]["elapsed_seconds"]
    detector_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duration"):
        aggregate_script.validate_row(
            candidate, root=ROOT, required_score_timeout=300
        )


def test_aggregator_rejects_duplicate_cases(tmp_path: Path) -> None:
    candidate = aggregate_script.validate_row(
        row(tmp_path, "wide_void", "same"), root=ROOT
    )
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_script.aggregate([candidate, candidate], min_cases_per_operator=0)


def test_cumulative_gate_prevents_flip_only_success(tmp_path: Path) -> None:
    candidate = row(tmp_path, "carrier_rewrite", "a")
    candidate["behavior_ioc_pass"] = False
    candidate["evidence"]["behavior_ioc_pass"] = evidence_file(
        tmp_path, "carrier-a-failed-ioc", False
    )
    validated = aggregate_script.validate_row(candidate, root=ROOT)
    assert validated["detector_flip"] is True
    assert validated["valid_evasion"] is False


def test_dex_payload_manifest_detects_noop_and_multidex_change(tmp_path: Path) -> None:
    original = tmp_path / "original.apk"
    same = tmp_path / "same.apk"
    changed = tmp_path / "changed.apk"
    for path, second in ((original, b"two"), (same, b"two"), (changed, b"changed")):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("classes.dex", b"one")
            archive.writestr("classes2.dex", second)
            archive.writestr("META-INF/signature", path.name)
    assert case_script.dex_payload_manifest(original) == case_script.dex_payload_manifest(same)
    assert case_script.dex_payload_manifest(original) != case_script.dex_payload_manifest(changed)


def test_cohort_attachment_requires_exact_two_operator_coverage(tmp_path: Path) -> None:
    validated = [
        aggregate_script.validate_row(row(tmp_path, operator, "a"), root=ROOT)
        for operator in ("carrier_rewrite", "wide_void")
    ]
    payload = aggregate_script.aggregate(validated, min_cases_per_operator=1)
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps({
        "schema_version": 1,
        "cohort_id": "cohort-a",
        "selection_rule": "detector_blind",
        "sampling_seed": 2027,
        "detector_predictions_used_for_selection": False,
        "rows": [{"sample_id": "a"}],
    }), encoding="utf-8")
    aggregate_script.attach_cohort(payload, cohort_path)
    assert payload["cohort"]["case_count"] == 1
    assert payload["cohort"]["detector_predictions_used_for_selection"] is False


def test_detector_evidence_requires_checkpoint_and_boolean_labels(tmp_path: Path) -> None:
    path = tmp_path / "detector.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy": "all_three_clean_malware_to_modified_benign",
                "checkpoints": [
                    {
                        "seed": seed,
                        "checkpoint_sha256": f"hash-{seed}",
                        "original_malware": True,
                        "modified_malware": False,
                    }
                    for seed in (42, 43, 44)
                ],
            }
        ),
        encoding="utf-8",
    )
    flipped, payload = case_script.verify_detector_evidence(path)
    assert flipped is True
    assert len(payload["checkpoints"]) == 3


def test_ioc_spec_cannot_be_empty(tmp_path: Path) -> None:
    path = tmp_path / "ioc.json"
    path.write_text(
        json.dumps({"schema_version": 1, "mode": "runtime_tokens", "required_runtime_tokens": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty"):
        case_script.load_ioc_spec(path)


def test_static_ioc_gate_is_count_preserving_and_nontrivial() -> None:
    required = {"sms": 2, "network": 1}
    assert ioc_script.static_iocs_preserved(required, {"sms": 2, "network": 4})
    assert not ioc_script.static_iocs_preserved(required, {"sms": 1, "network": 4})
    assert not ioc_script.static_iocs_preserved({}, {})
