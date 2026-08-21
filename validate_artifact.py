#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 43, 44)


def load(relative: str):
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


checksums = {}
for line in (ROOT / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
    expected, relative = line.split("  ", 1)
    checksums[relative] = expected
for relative, expected in checksums.items():
    require(digest(ROOT / relative) == expected, f"checksum mismatch: {relative}")

cohort_ids = {"gin": set(), "gcn": set()}
for architecture in ("gin", "gcn"):
    for seed in SEEDS:
        factorial = load(f"results/{architecture}/seed{seed}/factorial.json")
        restoration = load(f"results/{architecture}/seed{seed}/restoration.json")
        training = load(f"results/{architecture}/seed{seed}/training.json")
        require(len(factorial["rows"]) == 6, f"{architecture} seed {seed}: six conditions")
        require(len(factorial["per_sample"]) == 512, f"{architecture} seed {seed}: 512 samples")
        require(training["seed"] == seed, f"{architecture} seed {seed}: training seed")
        require(restoration["sample_count"] == 512, f"{architecture} seed {seed}: restoration cohort")
        require(
            restoration["shared_cohort_id"] == factorial["shared_cohort_id"],
            f"{architecture} seed {seed}: restoration/factorial cohort mismatch",
        )
        cohort_ids[architecture].add(factorial["shared_cohort_id"])
require(len(cohort_ids["gin"]) == 1, "GIN seeds do not share a cohort")
require(len(cohort_ids["gcn"]) == 1, "GCN seeds do not share a cohort")

apk = load("results/apk/apk_gate_summary.json")
evidence = load("results/apk/evidence_rows.json")
require(apk["cohort"]["case_count"] == 1462, "APK source cohort is not 1,462")
require(apk["per_seed_score_timeout_seconds"] == 300, "timeout provenance is not 300 seconds")
require(len(apk["rows"]) == 2924, "APK summary does not contain 2,924 operator rows")
require(len(evidence["rows"]) == 2924, "APK evidence does not contain 2,924 rows")
require({row["operator"] for row in evidence["rows"]} == {"carrier_rewrite", "wide_void"}, "operator set")
require(sum(bool(row.get("valid_evasion")) for row in apk["rows"]) == 0, "unexpected gate-valid evasion")

alignment = load("results/projection/alignment.json")
require(len(alignment["rows"]) == 442, "projection alignment must contain 442 rows")
require(
    not any("attacked_method_key" in row or "cohort_sample_id" in row for row in alignment["rows"]),
    "projection alignment rows retain program or source identifiers",
)
expected_alignment = {
    "42": {"carrier_rewrite": (74, 3, 13), "wide_void": (74, 36, 36)},
    "43": {"carrier_rewrite": (74, 11, 21), "wide_void": (74, 63, 63)},
    "44": {"carrier_rewrite": (73, 10, 20), "wide_void": (73, 36, 36)},
}
for seed, operators in expected_alignment.items():
    for operator, expected in operators.items():
        row = alignment["summaries"][seed][operator]
        actual = (
            row["pair_count"], row["exact_method_touch_count"],
            row["nonzero_same_method_change_count"],
        )
        require(actual == expected, f"alignment aggregate mismatch: {seed}/{operator}")

projection = load("results/projection/matched/summary.json")
projection_rows = load("results/projection/matched/evidence_rows.json")
require(len(projection_rows["rows"]) == 1105, "matched projection must contain 1,105 arms")
require(
    projection["overall_status_counts"] == {"complete": 1024, "failed": 21, "ineligible": 60},
    "matched-projection status counts",
)
require(
    projection["complete_case_paired_pool"]["seed_source_unit_count"] == 168,
    "matched-projection paired unit count",
)
require(
    not any(
        any(key in row for key in ("sample_id", "attacked_method", "target_method_used"))
        for row in projection_rows["rows"]
    ),
    "matched projection retains program or source identifiers",
)
status_counts = {}
for status in ("complete", "failed", "ineligible"):
    status_counts[status] = sum(row["status"] == status for row in projection_rows["rows"])
require(status_counts == projection["overall_status_counts"], "matched status/evidence mismatch")
for name in ("evidence_rows_json", "evidence_rows_csv"):
    relative = Path("results/projection/matched") / projection["provenance"][name]
    require(
        digest(ROOT / relative) == projection["provenance"][f"{name}_sha256"],
        f"matched projection provenance mismatch: {name}",
    )
for condition, aggregate in projection["conditions"].items():
    rows = [row for row in projection_rows["rows"] if row["condition"] == condition]
    require(len(rows) == 221, f"{condition}: frozen denominator")
    require(
        sum(row["status"] == "complete" for row in rows) == aggregate["status_counts"]["complete"],
        f"{condition}: completed-arm count",
    )
    require(aggregate["matching_detector_flip"]["count"] == 0, f"{condition}: matching flips")
    require(aggregate["all_checkpoint_detector_flip_count"] == 0, f"{condition}: all-checkpoint flips")

split = load("manifests/exact_identifier_split.json")
require(len(split["rows"]) == 16505, "split manifest unique-group count")
require(len({row["group_id"] for row in split["rows"]}) == 16505, "duplicate group assignment")
require({row["split"] for row in split["rows"]} == {"train", "test"}, "split labels")

print("PASS: checksums, six seeds, matched cohorts, restoration, projection, split, and 2,924 APK rows validate.")
