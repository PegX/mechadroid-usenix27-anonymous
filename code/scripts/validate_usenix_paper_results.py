#!/usr/bin/env python3
"""Fail-closed consistency checks for the USENIX manuscript."""

from __future__ import annotations

import json
import hashlib
import math
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "paper" / "usenix" / "main_usenix.tex"
GIN_ROOT = ROOT / "outputs" / "leakage_safe_shared_three_seed_same_sample_v2"
GCN_ROOT = ROOT / "outputs" / "leakage_safe_gcn_shared_three_seed"
PROJECTION_ROOT = (
    GIN_ROOT / "projection_alignment" / "matched_conditions" / "full_74"
)
ALIGNMENT = GIN_ROOT / "projection_alignment" / "embedding_alignment.json"
APK_SUMMARY = (
    ROOT / "outputs" / "leakage_safe_shared_three_seed" / "apk_gates"
    / "matched_all" / "run" / "apk_gate_summary.json"
)
DATASET_STATS = ROOT / "pyg_datasets_cicmaldroid_leakage_safe_seed42" / "stats.json"
SEEDS = (42, 43, 44)
CONDITIONS = (
    "random_noise", "random_node", "plain", "selection_only",
    "loss_only", "selection_plus_loss",
)
CONTRASTS = (
    "selection_only", "loss_only", "selection_plus_loss", "random_node", "random_noise",
)
T_CRITICAL_95_DF2 = 4.3026527299


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(text: str, fragment: str, label: str) -> None:
    if fragment not in text:
        raise ValueError(f"missing or stale {label}: {fragment}")


def reject(text: str, fragment: str, label: str) -> None:
    if fragment in text:
        raise ValueError(f"forbidden {label}: {fragment}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def condition_rates(
    run_root: Path, seed: int, *, require_loss_provenance: bool
) -> dict[str, float]:
    report = load_json(
        run_root / f"seed{seed}" / "factorial" / "node_latent_small_benchmark.json"
    )
    if report.get("detail", {}).get("schema_version") != 2:
        raise ValueError(f"seed {seed}: factorial report is not schema v2")
    rows = {row["condition"]: float(row["success_rate"]) for row in report["rows"]}
    if require_loss_provenance:
        for sample in report["detail"]["per_sample"]:
            for condition in ("loss_only", "selection_plus_loss"):
                record = sample["conditions"][condition]["summary"]
                for key in (
                    "circuit_loss_requested",
                    "circuit_loss_required",
                    "circuit_loss_active",
                ):
                    if record.get(key) is not True:
                        raise ValueError(
                            f"seed {seed}/{condition}: {key} is not true"
                        )
    return rows


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    center = statistics.fmean(values)
    half = T_CRITICAL_95_DF2 * statistics.stdev(values) / math.sqrt(3)
    return center, max(-1.0, center - half), min(1.0, center + half)


def probability_drop(row: dict) -> float:
    return float(row["reference_clean_malware_prob"]) - float(row["reference_adv_malware_prob"])


def per_reference(report: dict, condition: str) -> dict[str, dict]:
    rows = report["detail"]["per_condition_rows"][condition]
    return {row["reference_sample_id"]: row for row in rows}


def audit_all_figure_table_data(text: str) -> None:
    """Recompute every empirical figure/table value from authoritative artifacts."""
    flat_text = " ".join(text.split())

    def require_table(fragment: str, label: str) -> None:
        require(flat_text, " ".join(fragment.split()), label)

    stats = load_json(DATASET_STATS)
    for fragment, label in (
        (f"Extracted artifacts / failed paths & {stats['input_artifact_count']:,} / 734", "Table 1 artifacts"),
        (f"Unique identifier groups & {stats['unique_group_count']:,}", "Table 1 groups"),
        (f"Duplicate artifact copies removed & {stats['duplicate_artifacts_removed']:,}", "Table 1 duplicates"),
        (f"Valid train / test graphs & {stats['splits']['train']['count']:,} / {stats['splits']['test']['count']:,}", "Table 1 split"),
        (f"Invalid graphs rejected & {stats['total_rejected']}", "Table 1 rejected"),
    ):
        require_table(fragment, label)

    reports: dict[str, dict[int, dict]] = {"GIN": {}, "GCN": {}}
    attack_reports: dict[str, dict[int, dict]] = {"GIN": {}, "GCN": {}}
    roots = {"GIN": GIN_ROOT, "GCN": GCN_ROOT}
    for backbone, run_prefix in (("GIN", "leakage_safe_seed"), ("GCN", "leakage_safe_gcn_seed")):
        for seed in SEEDS:
            metrics = load_json(ROOT / "training_runs" / f"{run_prefix}{seed}" / "metrics.json")
            reports[backbone][seed] = metrics
            values = metrics["test_metrics"]
            require_table(
                f"{backbone} & {seed} & {metrics['best_epoch']} & "
                + " & ".join(f"{float(values[key]):.4f}" for key in ("accuracy", "precision", "recall", "f1", "roc_auc")),
                f"Table 2 {backbone}/{seed}",
            )
            attack_reports[backbone][seed] = load_json(
                roots[backbone] / f"seed{seed}" / "factorial" / "node_latent_small_benchmark.json"
            )

    gin_rows = {
        seed: {row["condition"]: row for row in attack_reports["GIN"][seed]["rows"]}
        for seed in SEEDS
    }
    seed_names = {42: "FortyTwo", 43: "FortyThree", 44: "FortyFour"}
    for seed in SEEDS:
        asr = " ".join(
            f"({index},{float(gin_rows[seed][condition]['success_rate']):.4f})"
            for index, condition in enumerate(CONDITIONS)
        )
        drop = " ".join(
            f"({index},{float(gin_rows[seed][condition]['mean_malware_prob_drop']):.4f})"
            for index, condition in enumerate(CONDITIONS)
        )
        require(text, rf"\def\FactorialASRSeed{seed_names[seed]}{{{asr}}}", f"Figure 2 ASR/{seed}")
        require(text, rf"\def\FactorialDropSeed{seed_names[seed]}{{{drop}}}", f"Figure 2 drop/{seed}")
    for metric, macro in (("success_rate", "FactorialASRMean"), ("mean_malware_prob_drop", "FactorialDropMean")):
        coords = " ".join(
            f"({index},{statistics.fmean(float(gin_rows[seed][condition][metric]) for seed in SEEDS):.4f})"
            for index, condition in enumerate(CONDITIONS)
        )
        require(text, rf"\def\{macro}{{{coords}}}", f"Figure 2 {macro}")

    ids = [row["sample_id"] for row in attack_reports["GIN"][42]["detail"]["selected_references"]]
    expected_asr: list[str] = []
    expected_drop: list[str] = []
    for position, condition in enumerate(CONTRASTS):
        asr_values, drop_values = [], []
        for seed in SEEDS:
            plain = per_reference(attack_reports["GIN"][seed], "plain")
            treated = per_reference(attack_reports["GIN"][seed], condition)
            asr_values.append(statistics.fmean(
                float(treated[item]["reference_adv_success"]) - float(plain[item]["reference_adv_success"])
                for item in ids
            ))
            drop_values.append(statistics.fmean(
                probability_drop(treated[item]) - probability_drop(plain[item]) for item in ids
            ))
        y = 5 - position
        for values, target in ((asr_values, expected_asr), (drop_values, expected_drop)):
            center, low, high = mean_ci(values)
            target.extend([
                f"coordinates {{({low:.4f},{y}) ({high:.4f},{y})}};",
                f"coordinates {{({center:.4f},{y})}};",
            ])
    for fragment in expected_asr + expected_drop:
        require(text, fragment, "Figure 3 paired coordinate")

    for backbone in ("GIN", "GCN"):
        selection_rows = []
        for seed in SEEDS:
            rows = {row["condition"]: row for row in attack_reports[backbone][seed]["rows"]}
            plain = 100.0 * float(rows["plain"]["success_rate"])
            selection = 100.0 * float(rows["selection_only"]["success_rate"])
            selection_rows.append((plain, selection))
            require_table(f"{backbone} & {seed} & {plain:.2f} & {selection:.2f} & {selection-plain:+.2f}", f"Table 4 {backbone}/{seed}")
        plain = statistics.fmean(row[0] for row in selection_rows)
        selection = statistics.fmean(row[1] for row in selection_rows)
        require_table(f"{backbone} & Mean & {plain:.2f} & {selection:.2f} & {selection-plain:+.2f}", f"Table 4 {backbone}/mean")

    selected_count = len(ids)
    layer_counts = {layer: [] for layer in range(3)}
    for seed in SEEDS:
        counts = Counter()
        for sample in attack_reports["GIN"][seed]["detail"]["per_sample"]:
            scores = [score for score in sample["top_scores"] if score["circuit"]["module"] == "gnn"]
            counts[int(scores[0]["circuit"]["layer"])] += 1
        for layer in range(3):
            layer_counts[layer].append(100.0 * counts[layer] / selected_count)
    for layer, values in layer_counts.items():
        coords = " ".join(f"({index},{value:.1f})" for index, value in enumerate(values))
        require(text, rf"\csname Hotspotgnn{layer}\endcsname{{{coords}}}", f"Figure 4 hotspot/{layer}")

    restorations = {
        seed: load_json(GIN_ROOT / f"seed{seed}" / "restoration" / "restoration_local_node.json")
        for seed in SEEDS
    }
    restoration_macros = {
        "RestorationHotspot": "mean_hotspot_target_node_recovery_ratio",
        "RestorationRandomLayer": "mean_random_layer_target_node_recovery_ratio",
        "RestorationRandomNode": "mean_hotspot_random_node_recovery_ratio",
    }
    for macro, key in restoration_macros.items():
        coords = " ".join(f"({index},{float(restorations[seed][key]):.4f})" for index, seed in enumerate(SEEDS))
        require(text, rf"\def\{macro}{{{coords}}}", f"Figure 4 {macro}")
    require(text, rf"\def\RestorationEligible{{{'/'.join(str(restorations[s]['eligible_sample_count']) for s in SEEDS)}}}", "Figure 4 eligible")
    require(text, rf"\def\RestorationLabelRecovery{{{'/'.join(str(restorations[s]['hotspot_label_recovery_count']) for s in SEEDS)}}}", "Figure 4 label recovery")
    require(text, rf"\def\RestorationSuccessfulAttacks{{{'/'.join(str(restorations[s]['successful_attack_count']) for s in SEEDS)}}}", "Figure 4 successful attacks")

    alignment_rows = load_json(ALIGNMENT)["rows"]
    for seed in SEEDS:
        cells = {}
        for operator in ("carrier_rewrite", "wide_void"):
            rows = [row for row in alignment_rows if int(row["seed"]) == seed and row["operator"] == operator]
            changed = [row for row in rows if float(row["apk_delta_l2"]) > 0.0]
            cells[operator] = {
                "pairs": len(rows),
                "touch": sum(bool(row["materializer_exact_method_touch"]) for row in rows),
                "change": len(changed),
                "cosine": statistics.fmean(float(row["cosine"]) for row in changed),
                "distance": statistics.median(float(row["relative_target_distance"]) for row in changed),
            }
        carrier, wide = cells["carrier_rewrite"], cells["wide_void"]
        require_table(
            f"{seed} & {carrier['pairs']} & {carrier['touch']}/{carrier['pairs']}; {carrier['change']}; "
            f"${carrier['cosine']:.3f}$ & {wide['touch']}/{wide['pairs']}; {wide['change']}; "
            f"${wide['cosine']:.3f}$; {wide['distance']:.2f}",
            f"Table 5 seed {seed}",
        )

    projection = load_json(PROJECTION_ROOT / "summary.json")
    labels = {
        "semantics_preserving": "Semantic one-op",
        "latent_cosine": "Cosine-ranked one-op",
        "destructive_upper": "Destructive upper bound",
        "random_direction": "Random direction",
        "random_method": "Random method",
    }
    for condition, label in labels.items():
        row = projection["conditions"][condition]
        cosine = row["latent_cosine"]
        delta = row["matching_malware_probability_delta"]
        gates = row["cumulative_gate_counts"]
        require_table(
            f"{label} & {row['status_counts']['complete']} & "
            f"${cosine['mean']:.3f}$ [${cosine['cluster_bootstrap_95_ci'][0]:.3f},{cosine['cluster_bootstrap_95_ci'][1]:.3f}$] & "
            f"${10000.0*delta['mean']:.3f}$ [${10000.0*delta['cluster_bootstrap_95_ci'][0]:.3f},{10000.0*delta['cluster_bootstrap_95_ci'][1]:.3f}$] & "
            f"{gates['art_install_pass']} & {gates['smoke_pass']} & {gates['behavior_ioc_pass']} / 0",
            f"Table 6 {condition}",
        )

    apk = load_json(APK_SUMMARY)
    carrier = apk["operators"]["carrier_rewrite"]
    wide = apk["operators"]["wide_void"]
    total = carrier["case_count"]
    for label, key in (
        ("DEX edit", "edit_applied_count"),
        ("Repack", "repack_count"),
        ("ART/install", "art_install_count"),
        ("Smoke", "smoke_count"),
        ("IOC proxy", "behavior_ioc_count"),
        ("Gate-valid evasion", "valid_evasion_count"),
    ):
        require_table(
            f"{label} & {carrier[key]}/{total} & {wide[key]}/{total}",
            f"Table 7 {label}",
        )


def main() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "update_sp_paper_results.py"), "--check-only"],
        cwd=ROOT,
        check=True,
    )
    text = PAPER.read_text(encoding="utf-8")

    audit_all_figure_table_data(text)

    require(text, r"\documentclass[letterpaper,twocolumn,10pt]{article}", "USENIX class")
    require(text, r"\usepackage{usenix-2020-09}", "USENIX style")
    require(
        text,
        "MechaDroid: Auditing the Latent-to-APK Boundary in",
        "frozen title",
    )
    require(text, r"\section{Open Science}", "Open Science appendix")
    for forbidden, label in (
        ("IEEEtran", "IEEE template"),
        ("leakage-safe", "overbroad leakage claim"),
        ("A stronger adversary", "threat-model ordering"),
        (r"\balance", "manual column balancing"),
    ):
        reject(text, forbidden, label)

    condition_order = ("plain", "selection_only")
    for backbone, root in (("GIN", GIN_ROOT), ("GCN", GCN_ROOT)):
        seed_rows: list[dict[str, float]] = []
        for seed in (42, 43, 44):
            rates = condition_rates(
                root, seed, require_loss_provenance=(backbone == "GIN")
            )
            seed_rows.append(rates)
            values = " & ".join(f"{100.0 * rates[name]:.2f}" for name in condition_order)
            delta = 100.0 * (rates["selection_only"] - rates["plain"])
            require(
                text,
                f"{backbone} & {seed} & {values} & {delta:+.2f}",
                f"{backbone} seed {seed} row",
            )
        means = {
            name: sum(row[name] for row in seed_rows) / len(seed_rows)
            for name in condition_order
        }
        values = " & ".join(f"{100.0 * means[name]:.2f}" for name in condition_order)
        delta = 100.0 * (means["selection_only"] - means["plain"])
        require(
            text,
            f"{backbone} & Mean & {values} & {delta:+.2f}",
            f"{backbone} mean row",
        )

    gcn_summary = load_json(GCN_ROOT / "second_architecture_summary.json")
    if gcn_summary.get("architecture") != "gcn" or gcn_summary.get("shared_cohort") != 512:
        raise ValueError("GCN summary does not describe the frozen 512-sample cohort")

    projection = load_json(PROJECTION_ROOT / "summary.json")
    if projection.get("overall_status_counts") != {
        "complete": 1024, "failed": 21, "ineligible": 60
    }:
        raise ValueError("matched-projection status counts are stale")
    paired = projection.get("complete_case_paired_pool", {})
    if paired.get("seed_source_unit_count") != 168 or paired.get("source_count") != 71:
        raise ValueError("matched-projection complete-case pool is stale")
    for condition, row in projection.get("conditions", {}).items():
        if row.get("matching_detector_flip", {}).get("count") != 0:
            raise ValueError(f"{condition}: matching-checkpoint flip count is not zero")
        if row.get("all_checkpoint_detector_flip_count") != 0:
            raise ValueError(f"{condition}: all-checkpoint flip count is not zero")
    provenance = projection.get("provenance", {})
    for name in ("evidence_rows_json", "evidence_rows_csv"):
        path = Path(provenance[name])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256(path) != provenance[f"{name}_sha256"]:
            raise ValueError(f"matched-projection {name} hash mismatch")
    require(text, "The complete-case paired pool contains 168", "projection paired pool")
    require(text, "Semantic one-op & 214", "projection semantic row")
    require(text, "Cosine-ranked one-op & 214", "projection targeted row")
    require(text, "Random direction & 208", "projection random-direction row")

    print(
        "USENIX template, claims, GIN/GCN tables, matched projection, and "
        "artifact provenance validate."
    )


if __name__ == "__main__":
    main()
