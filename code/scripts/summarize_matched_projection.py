#!/usr/bin/env python3
"""Summarize the frozen five-condition matched projection experiment.

The fixed denominator is every (attack checkpoint seed, source APK) target in
``progress.json`` for each condition.  Failures and predeclared ineligible
arms remain in that denominator.  Paired comparisons use only units for which
all five conditions reached the gated stage.  Confidence intervals resample
source APKs as clusters, retaining the three seed-level observations within a
source whenever present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


CONDITIONS = (
    "semantics_preserving",
    "latent_cosine",
    "destructive_upper",
    "random_direction",
    "random_method",
)
CONTRASTS = (
    ("latent_cosine", "semantics_preserving"),
    ("latent_cosine", "random_direction"),
    ("latent_cosine", "random_method"),
    ("destructive_upper", "semantics_preserving"),
)
GATES = (
    "edit_applied_pass",
    "repack_pass",
    "art_install_pass",
    "smoke_pass",
    "behavior_ioc_pass",
    "detector_flip",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float:
    seq = list(values)
    return sum(seq) / len(seq) if seq else math.nan


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def cluster_bootstrap_ci(
    rows: list[dict[str, Any]], value: Callable[[dict[str, Any]], float],
    *, iterations: int, seed: int,
) -> list[float]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[str(row["sample_id"])].append(row)
    keys = sorted(clusters)
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sampled = [rng.choice(keys) for _ in keys]
        values = [value(row) for key in sampled for row in clusters[key]]
        estimates.append(mean(values))
    estimates.sort()
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def cluster_sign_flip_p(
    rows: list[dict[str, Any]], value: Callable[[dict[str, Any]], float],
    *, iterations: int, seed: int,
) -> float:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        clusters[str(row["sample_id"])].append(value(row))
    cluster_means = [mean(values) for _, values in sorted(clusters.items())]
    observed = abs(mean(cluster_means))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        statistic = abs(mean(v * (-1.0 if rng.random() < 0.5 else 1.0) for v in cluster_means))
        extreme += statistic >= observed - 1e-15
    return (extreme + 1) / (iterations + 1)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def classify_failure(error: str) -> str:
    if "Invalid file" in error or "failed opening zip" in error:
        return "invalid_source_apk_zip"
    if "Invalid register: v16" in error:
        return "materialization_invalid_register"
    if "timed out" in error.lower() or "timeout" in error.lower():
        return "timeout"
    if "No space left" in error:
        return "disk_full"
    return "other"


def quantile_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "p99": percentile(ordered, 0.99),
        "maximum": max(ordered) if ordered else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--randomization-iterations", type=int, default=20_000)
    args = parser.parse_args()
    root = args.input_root.resolve()
    progress_path = root / "progress.json"
    progress = read_json(progress_path)
    arms = progress.get("arms", [])
    if len(arms) != 1105:
        raise ValueError(f"expected 1105 frozen arms, found {len(arms)}")

    evidence: list[dict[str, Any]] = []
    for arm in arms:
        seed = int(arm["seed"])
        sample_id = str(arm["sample_id"])
        condition = str(arm["condition"])
        row: dict[str, Any] = {
            "seed": seed,
            "sample_id": sample_id,
            "condition": condition,
            "attacked_method": arm.get("attacked_method"),
            "target_method_used": arm.get("target_method_used"),
            "status": arm.get("status"),
            "stage": arm.get("stage"),
            "ineligible_reason": arm.get("error") if arm.get("status") == "ineligible" else None,
            "failure_category": classify_failure(str(arm.get("error", "")))
            if arm.get("status") == "failed" else None,
        }
        if arm.get("status") == "complete":
            arm_dir = root / f"seed{seed}" / sample_id / condition
            material = read_json(arm_dir / "materialization" / "materialization_report.json")
            detector = read_json(arm_dir / "detector" / "detector_evidence.json")
            gates = read_json(arm_dir / "gates" / "evidence_row.json")
            alignment = material["measured_alignment"]
            matching = next(item for item in detector["checkpoints"] if int(item["seed"]) == seed)
            checkpoint_rows = []
            for checkpoint in detector["checkpoints"]:
                checkpoint_rows.append({
                    "seed": int(checkpoint["seed"]),
                    "original_malware": bool(checkpoint["original_malware"]),
                    "modified_malware": bool(checkpoint["modified_malware"]),
                    "original_malware_probability": float(checkpoint["original_malware_probability"]),
                    "modified_malware_probability": float(checkpoint["modified_malware_probability"]),
                    "malware_probability_delta": float(checkpoint["modified_malware_probability"])
                    - float(checkpoint["original_malware_probability"]),
                    "elapsed_seconds": float(checkpoint["elapsed_seconds"]),
                    "timeout_seconds": int(checkpoint["timeout_seconds"]),
                })
            row.update({
                "selected_candidate": (material.get("selected_candidate") or {}).get("name"),
                "latent_cosine": float(alignment["latent_cosine"]),
                "latent_delta_l2": float(alignment["latent_delta_l2"]),
                "realized_delta_l2": float(alignment["realized_delta_l2"]),
                "relative_target_distance": float(alignment["relative_target_distance"]),
                "matching_original_malware_probability": float(matching["original_malware_probability"]),
                "matching_modified_malware_probability": float(matching["modified_malware_probability"]),
                "matching_malware_probability_delta": float(matching["modified_malware_probability"])
                - float(matching["original_malware_probability"]),
                "matching_detector_flip": bool(matching["original_malware"])
                and not bool(matching["modified_malware"]),
                "all_checkpoint_detector_flip": bool(gates["detector_flip"]),
                "checkpoints": checkpoint_rows,
            })
            cumulative = True
            for gate in GATES:
                cumulative = cumulative and bool(gates[gate])
                row[gate] = bool(gates[gate])
                row[f"cumulative_{gate}"] = cumulative
        evidence.append(row)

    evidence_json = root / "evidence_rows.json"
    write_json(evidence_json, {"schema_version": 1, "rows": evidence})
    evidence_csv = root / "evidence_rows.csv"
    csv_fields = [
        "seed", "sample_id", "condition", "status", "stage", "ineligible_reason",
        "failure_category", "attacked_method", "target_method_used", "selected_candidate",
        "latent_cosine", "latent_delta_l2", "realized_delta_l2", "relative_target_distance",
        "matching_original_malware_probability", "matching_modified_malware_probability",
        "matching_malware_probability_delta", "matching_detector_flip",
        "all_checkpoint_detector_flip",
    ] + [item for gate in GATES for item in (gate, f"cumulative_{gate}")]
    with evidence_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(evidence)

    complete = [row for row in evidence if row["status"] == "complete"]
    fixed_per_condition = len(arms) // len(CONDITIONS)
    condition_summary: dict[str, Any] = {}
    all_elapsed = []
    for condition_index, condition in enumerate(CONDITIONS):
        condition_arms = [row for row in evidence if row["condition"] == condition]
        rows = [row for row in condition_arms if row["status"] == "complete"]
        status_counts = Counter(row["status"] for row in condition_arms)
        cumulative_counts = {
            gate: sum(bool(row.get(f"cumulative_{gate}")) for row in rows) for gate in GATES
        }
        elapsed = [item["elapsed_seconds"] for row in rows for item in row["checkpoints"]]
        all_elapsed.extend(elapsed)
        matching_flips = sum(bool(row["matching_detector_flip"]) for row in rows)
        condition_summary[condition] = {
            "frozen_denominator": fixed_per_condition,
            "status_counts": dict(status_counts),
            "complete_source_count": len({row["sample_id"] for row in rows}),
            "cumulative_gate_counts": cumulative_counts,
            "latent_cosine": {
                "mean": mean(row["latent_cosine"] for row in rows),
                "cluster_bootstrap_95_ci": cluster_bootstrap_ci(
                    rows, lambda row: row["latent_cosine"], iterations=args.bootstrap_iterations,
                    seed=41_000 + condition_index,
                ),
            },
            "realized_delta_l2": {
                "mean": mean(row["realized_delta_l2"] for row in rows),
                "cluster_bootstrap_95_ci": cluster_bootstrap_ci(
                    rows, lambda row: row["realized_delta_l2"], iterations=args.bootstrap_iterations,
                    seed=42_000 + condition_index,
                ),
            },
            "relative_target_distance": {
                "mean": mean(row["relative_target_distance"] for row in rows),
                "cluster_bootstrap_95_ci": cluster_bootstrap_ci(
                    rows, lambda row: row["relative_target_distance"], iterations=args.bootstrap_iterations,
                    seed=43_000 + condition_index,
                ),
            },
            "matching_malware_probability_delta": {
                "mean": mean(row["matching_malware_probability_delta"] for row in rows),
                "cluster_bootstrap_95_ci": cluster_bootstrap_ci(
                    rows, lambda row: row["matching_malware_probability_delta"],
                    iterations=args.bootstrap_iterations, seed=44_000 + condition_index,
                ),
            },
            "matching_detector_flip": {
                "count": matching_flips,
                "rate_among_complete": matching_flips / len(rows),
                "wilson_95_ci": wilson_interval(matching_flips, len(rows)),
            },
            "all_checkpoint_detector_flip_count": sum(
                bool(row["all_checkpoint_detector_flip"]) for row in rows
            ),
            "checkpoint_runtime_seconds": quantile_summary(elapsed),
            "checkpoint_calls_exceeding_300_seconds": sum(value > 300.0 for value in elapsed),
        }

    unit_conditions: dict[tuple[int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in complete:
        unit_conditions[(int(row["seed"]), str(row["sample_id"]))][str(row["condition"])] = row
    common_units = {
        key: rows for key, rows in unit_conditions.items() if set(rows) == set(CONDITIONS)
    }
    paired_rows = [
        {"seed": seed, "sample_id": sample_id, "conditions": rows}
        for (seed, sample_id), rows in sorted(common_units.items())
    ]
    contrasts: dict[str, Any] = {}
    for contrast_index, (first, second) in enumerate(CONTRASTS):
        def difference(metric: str) -> Callable[[dict[str, Any]], float]:
            return lambda row: float(row["conditions"][first][metric]) - float(
                row["conditions"][second][metric]
            )

        cosine_value = difference("latent_cosine")
        score_value = difference("matching_malware_probability_delta")
        cosine_values = [cosine_value(row) for row in paired_rows]
        score_values = [score_value(row) for row in paired_rows]
        first_flip = sum(row["conditions"][first]["matching_detector_flip"] for row in paired_rows)
        second_flip = sum(row["conditions"][second]["matching_detector_flip"] for row in paired_rows)
        name = f"{first}_minus_{second}"
        contrasts[name] = {
            "paired_unit_count": len(paired_rows),
            "source_cluster_count": len({row["sample_id"] for row in paired_rows}),
            "latent_cosine_difference": {
                "mean": mean(cosine_values),
                "cluster_bootstrap_95_ci": cluster_bootstrap_ci(
                    paired_rows, cosine_value, iterations=args.bootstrap_iterations,
                    seed=51_000 + contrast_index,
                ),
                "cluster_sign_flip_p_value": cluster_sign_flip_p(
                    paired_rows, cosine_value, iterations=args.randomization_iterations,
                    seed=52_000 + contrast_index,
                ),
            },
            "matching_malware_probability_delta_difference": {
                "mean": mean(score_values),
                "cluster_bootstrap_95_ci": cluster_bootstrap_ci(
                    paired_rows, score_value, iterations=args.bootstrap_iterations,
                    seed=53_000 + contrast_index,
                ),
                "cluster_sign_flip_p_value": cluster_sign_flip_p(
                    paired_rows, score_value, iterations=args.randomization_iterations,
                    seed=54_000 + contrast_index,
                ),
            },
            "matching_detector_flip_counts": {first: first_flip, second: second_flip},
            "discordant_pair_count": sum(
                row["conditions"][first]["matching_detector_flip"]
                != row["conditions"][second]["matching_detector_flip"] for row in paired_rows
            ),
        }

    failures = [row for row in evidence if row["status"] == "failed"]
    summary = {
        "schema_version": 1,
        "design": {
            "conditions": list(CONDITIONS),
            "frozen_arm_count": len(arms),
            "frozen_seed_source_target_count": fixed_per_condition,
            "frozen_source_count": int(progress["cohort_sample_count"]),
            "seeds": sorted({int(row["seed"]) for row in evidence}),
            "failure_policy": "fail_closed_in_frozen_condition_denominator",
            "paired_pool_policy": "all_five_conditions_complete_at_gated_stage",
            "confidence_interval_policy": "source_APK_cluster_bootstrap_10000_resamples",
        },
        "overall_status_counts": dict(Counter(row["status"] for row in evidence)),
        "failure_categories": dict(Counter(row["failure_category"] for row in failures)),
        "failed_source_count": len({row["sample_id"] for row in failures}),
        "conditions": condition_summary,
        "complete_case_paired_pool": {
            "seed_source_unit_count": len(paired_rows),
            "source_count": len({row["sample_id"] for row in paired_rows}),
            "unit_keys": [f"seed{row['seed']}:{row['sample_id']}" for row in paired_rows],
        },
        "paired_contrasts": contrasts,
        "runtime": {
            "all_completed_checkpoint_calls": len(all_elapsed),
            "seconds": quantile_summary(all_elapsed),
            "calls_exceeding_300_seconds": sum(value > 300.0 for value in all_elapsed),
        },
        "provenance": {
            "progress_json": str(progress_path),
            "progress_sha256": sha256(progress_path),
            "evidence_rows_json": str(evidence_json),
            "evidence_rows_json_sha256": sha256(evidence_json),
            "evidence_rows_csv": str(evidence_csv),
            "evidence_rows_csv_sha256": sha256(evidence_csv),
        },
    }
    summary_path = root / "summary.json"
    write_json(summary_path, summary)
    print(json.dumps({
        "summary": str(summary_path),
        "evidence_json": str(evidence_json),
        "evidence_csv": str(evidence_csv),
        "overall_status_counts": summary["overall_status_counts"],
        "paired_pool": summary["complete_case_paired_pool"],
    }, indent=2))


if __name__ == "__main__":
    main()
