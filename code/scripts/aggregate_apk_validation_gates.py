#!/usr/bin/env python3
"""Validate per-APK evidence and aggregate cumulative deployment gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


OPERATORS = ("carrier_rewrite", "wide_void")
GATES = (
    "edit_applied_pass",
    "repack_pass",
    "art_install_pass",
    "smoke_pass",
    "behavior_ioc_pass",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("APK gate evidence must be a JSON list or an object with rows")
    return rows


def validate_evidence_path(
    path_value: Any, *, root: Path, label: str, expected_pass: bool
) -> str:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"Missing {label} evidence path")
    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty {label} evidence file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Evidence file is not valid JSON: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("pass"), bool):
        raise ValueError(f"Evidence file must contain an explicit boolean pass field: {path}")
    if payload["pass"] is not expected_pass:
        raise ValueError(
            f"Evidence/result mismatch for {label}: row={expected_pass}, evidence={payload['pass']}"
        )
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return path.name


def validate_row(
    row: dict[str, Any], *, root: Path, required_score_timeout: int | None = None
) -> dict[str, Any]:
    operator = row.get("operator")
    if operator not in OPERATORS:
        raise ValueError(f"Unsupported APK operator: {operator!r}")
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Every APK gate row requires a stable sample_id")

    evidence = row.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"APK gate row {sample_id} has no evidence mapping")
    normalized_evidence = {}
    for gate in GATES:
        if not isinstance(row.get(gate), bool):
            raise ValueError(f"APK gate row {sample_id} must explicitly set {gate}")
        normalized_evidence[gate] = validate_evidence_path(
            evidence.get(gate), root=root, label=f"{sample_id}:{gate}",
            expected_pass=bool(row[gate]),
        )
    if not isinstance(row.get("detector_flip"), bool):
        raise ValueError(f"APK gate row {sample_id} must explicitly set detector_flip")
    normalized_evidence["detector_flip"] = validate_evidence_path(
        evidence.get("detector_flip"), root=root, label=f"{sample_id}:detector_flip",
        expected_pass=bool(row["detector_flip"]),
    )
    detector_path = Path(str(evidence["detector_flip"]))
    if not detector_path.is_absolute():
        detector_path = root / detector_path
    detector_payload = json.loads(detector_path.read_text(encoding="utf-8"))
    detector_by_seed: dict[str, bool] = {}
    detector_record = detector_payload.get("detector", {})
    checkpoints = detector_record.get("checkpoints", [])
    if row["detector_flip"] and not checkpoints:
        raise ValueError(f"Positive detector flip for {sample_id} lacks three-checkpoint evidence")
    if checkpoints:
        seeds = {int(checkpoint["seed"]) for checkpoint in checkpoints}
        if seeds != {42, 43, 44}:
            raise ValueError(f"Detector evidence for {sample_id} lacks seeds 42/43/44")
        detector_by_seed = {
            str(int(checkpoint["seed"])): bool(
                checkpoint["original_malware"] and not checkpoint["modified_malware"]
            )
            for checkpoint in checkpoints
        }
        if bool(row["detector_flip"]) != all(detector_by_seed.values()):
            raise ValueError(f"All-three detector policy mismatch for {sample_id}")
        if required_score_timeout is not None:
            if detector_record.get("per_seed_timeout_seconds") != required_score_timeout:
                raise ValueError(f"Detector evidence for {sample_id} lacks the fixed scoring budget")
            for checkpoint in checkpoints:
                elapsed = checkpoint.get("elapsed_seconds")
                if checkpoint.get("timeout_seconds") != required_score_timeout:
                    raise ValueError(f"Checkpoint budget mismatch for {sample_id}")
                if not isinstance(elapsed, (int, float)) or not 0 <= elapsed <= required_score_timeout:
                    raise ValueError(f"Invalid checkpoint duration for {sample_id}")
    elif required_score_timeout is not None and "error" not in detector_payload:
        raise ValueError(f"Unbudgeted detector evidence for {sample_id}")

    cumulative = True
    cumulative_pass = {}
    for gate in GATES:
        cumulative = cumulative and bool(row[gate])
        cumulative_pass[gate] = cumulative
    valid_evasion = cumulative and bool(row["detector_flip"])
    return {
        "operator": operator,
        "sample_id": sample_id,
        **{gate: bool(row[gate]) for gate in GATES},
        "detector_flip": bool(row["detector_flip"]),
        "cumulative_pass": cumulative_pass,
        "valid_evasion": valid_evasion,
        "detector_by_seed": detector_by_seed,
        "evidence": normalized_evidence,
    }


def aggregate(rows: list[dict[str, Any]], *, min_cases_per_operator: int) -> dict[str, Any]:
    by_operator: dict[str, list[dict[str, Any]]] = {operator: [] for operator in OPERATORS}
    identities: set[tuple[str, str]] = set()
    for row in rows:
        identity = (row["operator"], row["sample_id"])
        if identity in identities:
            raise ValueError(f"Duplicate APK gate case: {identity}")
        identities.add(identity)
        by_operator[row["operator"]].append(row)
    for operator, operator_rows in by_operator.items():
        if len(operator_rows) < min_cases_per_operator:
            raise ValueError(
                f"Operator {operator} has {len(operator_rows)} validated cases; "
                f"requires at least {min_cases_per_operator}"
            )

    summary = {}
    for operator, operator_rows in by_operator.items():
        total = len(operator_rows)
        counts = {
            gate: sum(int(row["cumulative_pass"][gate]) for row in operator_rows)
            for gate in GATES
        }
        valid_evasion_count = sum(int(row["valid_evasion"]) for row in operator_rows)
        detector_flip_count = sum(int(row["detector_flip"]) for row in operator_rows)
        per_seed_detector_flip_count = {
            str(seed): sum(int(row.get("detector_by_seed", {}).get(str(seed), False)) for row in operator_rows)
            for seed in (42, 43, 44)
        }
        per_seed_valid_evasion_count = {
            str(seed): sum(
                int(
                    row["cumulative_pass"]["behavior_ioc_pass"]
                    and row.get("detector_by_seed", {}).get(str(seed), False)
                )
                for row in operator_rows
            )
            for seed in (42, 43, 44)
        }
        summary[operator] = {
            "case_count": total,
            "edit_applied_count": counts["edit_applied_pass"],
            "repack_count": counts["repack_pass"],
            "art_install_count": counts["art_install_pass"],
            "smoke_count": counts["smoke_pass"],
            "behavior_ioc_count": counts["behavior_ioc_pass"],
            "valid_evasion_count": valid_evasion_count,
            "detector_flip_count": detector_flip_count,
            "per_seed_detector_flip_count": per_seed_detector_flip_count,
            "per_seed_valid_evasion_count": per_seed_valid_evasion_count,
            "edit_applied_rate": counts["edit_applied_pass"] / total,
            "repack_rate": counts["repack_pass"] / total,
            "art_install_rate": counts["art_install_pass"] / total,
            "smoke_rate": counts["smoke_pass"] / total,
            "behavior_ioc_rate": counts["behavior_ioc_pass"] / total,
            "valid_evasion_rate": valid_evasion_count / total,
        }
    return {
        "schema_version": 1,
        "gate_semantics": "cumulative_fail_closed",
        "operators": summary,
        "rows": rows,
    }


def attach_cohort(payload: dict[str, Any], cohort_path: Path) -> None:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort_rows = cohort.get("rows") if isinstance(cohort, dict) else None
    if cohort.get("schema_version") != 1 or not isinstance(cohort_rows, list):
        raise ValueError("Invalid APK cohort manifest")
    if cohort.get("detector_predictions_used_for_selection") is not False:
        raise ValueError("APK cohort selection must be detector-blind")
    sample_ids = [row.get("sample_id") for row in cohort_rows]
    if len(sample_ids) != len(set(sample_ids)) or any(not value for value in sample_ids):
        raise ValueError("APK cohort has duplicate/missing sample IDs")
    expected = {(operator, sample_id) for operator in OPERATORS for sample_id in sample_ids}
    observed = {(row["operator"], row["sample_id"]) for row in payload["rows"]}
    if observed != expected:
        raise ValueError("APK evidence rows do not exactly cover the preregistered matched cohort")
    payload["cohort"] = {
        "cohort_id": cohort.get("cohort_id"),
        "case_count": len(sample_ids),
        "selection_rule": cohort.get("selection_rule"),
        "sampling_seed": cohort.get("sampling_seed"),
        "carrier_repack_candidate_count": cohort.get("carrier_repack_candidate_count"),
        "detector_predictions_used_for_selection": False,
        "manifest_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-cases-per-operator", type=int, default=1)
    parser.add_argument("--cohort-manifest", type=Path)
    parser.add_argument("--require-score-timeout-seconds", type=int)
    args = parser.parse_args()
    if args.min_cases_per_operator < 1:
        raise ValueError("--min-cases-per-operator must be positive")

    validated = [
        validate_row(
            row, root=ROOT, required_score_timeout=args.require_score_timeout_seconds
        )
        for row in load_rows(args.evidence)
    ]
    payload = aggregate(validated, min_cases_per_operator=args.min_cases_per_operator)
    if args.require_score_timeout_seconds is not None:
        payload["per_seed_score_timeout_seconds"] = args.require_score_timeout_seconds
    if args.cohort_manifest is not None:
        attach_cohort(payload, args.cohort_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "operators": payload["operators"]}, indent=2))


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main()
