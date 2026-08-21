#!/usr/bin/env python3
"""Fail closed unless every requested component-loss row is fully auditable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LOSS_CONDITIONS = {"loss_only", "selection_plus_loss"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.report.read_text(encoding="utf-8"))
    samples = payload.get("detail", {}).get("per_sample", [])
    if not samples:
        raise SystemExit("report contains no per-sample rows")

    checked = 0
    for sample in samples:
        sample_id = sample.get("reference_summary", {}).get("sample_id")
        if not sample_id:
            raise SystemExit("sample row lacks reference sample_id")
        conditions = sample.get("conditions", {})
        missing = LOSS_CONDITIONS - set(conditions)
        if missing:
            raise SystemExit(f"{sample_id}: missing loss conditions {sorted(missing)}")
        for condition in sorted(LOSS_CONDITIONS):
            row = conditions[condition].get("summary", {})
            if row.get("circuit_loss_requested") is not True:
                raise SystemExit(f"{sample_id}/{condition}: loss not recorded as requested")
            if row.get("circuit_loss_required") is not True:
                raise SystemExit(f"{sample_id}/{condition}: fail-closed contract not recorded")
            if row.get("circuit_loss_active") is not True:
                raise SystemExit(f"{sample_id}/{condition}: loss inactive")
            clean_id = row.get("circuit_loss_clean_reference_id")
            attack_id = row.get("circuit_loss_attack_reference_id")
            if clean_id != sample_id or attack_id != sample_id:
                raise SystemExit(
                    f"{sample_id}/{condition}: source mismatch "
                    f"clean={clean_id} attack={attack_id}"
                )
            expected = set(row.get("circuit_loss_expected_keys", []))
            captured = set(row.get("circuit_loss_captured_keys", []))
            if not expected or captured != expected:
                raise SystemExit(
                    f"{sample_id}/{condition}: hook mismatch "
                    f"expected={sorted(expected)} captured={sorted(captured)}"
                )
            checked += 1

    print(
        json.dumps(
            {
                "report": str(args.report),
                "sample_count": len(samples),
                "validated_loss_rows": checked,
                "same_sample_alignment": True,
                "fail_closed_instrumentation": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
