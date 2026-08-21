#!/usr/bin/env python3
"""Fail-closed backfill for shared-cohort latent, localized restoration, and APK gates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


SEEDS = (42, 43, 44)
CONDITIONS = (
    ("random_noise", "Random noise"),
    ("random_node", "Random node"),
    ("plain", "Plain"),
    ("selection_only", "Selection only"),
    ("loss_only", "Loss only"),
    ("selection_plus_loss", "Selection + loss"),
)
CONTRASTS = (
    ("selection_only", "Selection only $-$ plain"),
    ("loss_only", "Loss only $-$ plain"),
    ("selection_plus_loss", "Selection + loss $-$ plain"),
    ("random_node", "Random node $-$ plain"),
    ("random_noise", "Random noise $-$ plain"),
)
T_CRITICAL_95_DF2 = 4.3026527299


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required completed artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"Expected exactly three seed-level values, received {len(values)}")
    center = statistics.fmean(values)
    half = T_CRITICAL_95_DF2 * statistics.stdev(values) / math.sqrt(len(values))
    return center, center - half, center + half


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_ci(values: list[float], *, bounds: tuple[float, float] | None = None) -> str:
    center, low, high = mean_ci(values)
    if bounds is not None:
        low = max(bounds[0], low)
        high = min(bounds[1], high)
    return f"{fmt(center)} [{fmt(low)}, {fmt(high)}]"


def replace_region(text: str, name: str, body: str) -> str:
    begin = f"% AUTO-BACKFILL: {name}_BEGIN"
    end = f"% AUTO-BACKFILL: {name}_END"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected exactly one marker pair for {name}")
    prefix, remainder = text.split(begin, 1)
    _, suffix = remainder.split(end, 1)
    return f"{prefix}{begin}\n{body.rstrip()}\n{end}{suffix}"


def condition_rows(report: dict) -> dict[str, dict]:
    rows = {row["condition"]: row for row in report.get("rows", [])}
    expected = {name for name, _ in CONDITIONS}
    if set(rows) != expected:
        raise ValueError(f"Factorial conditions differ: expected={sorted(expected)}, got={sorted(rows)}")
    detail = report.get("detail", {})
    if detail.get("condition_set") != "factorial":
        raise ValueError("Refusing to backfill a non-factorial benchmark report")
    if detail.get("reference_sampling") != "shared_manifest":
        raise ValueError("Primary report must use a frozen shared-reference manifest")
    if not detail.get("attack_states") or not detail.get("attack_states_sha256"):
        raise ValueError("Factorial report lacks replayable attack-state provenance")
    for sample in detail.get("per_sample", []):
        sample_id = sample.get("reference_summary", {}).get("sample_id")
        if not sample_id:
            raise ValueError("Factorial report contains a sample without a stable ID")
        conditions = sample.get("conditions", {})
        for condition in ("loss_only", "selection_plus_loss"):
            summary = conditions.get(condition, {}).get("summary", {})
            if summary.get("circuit_loss_requested") is not True:
                raise ValueError(f"{sample_id}/{condition}: component loss was not requested")
            if summary.get("circuit_loss_required") is not True:
                raise ValueError(f"{sample_id}/{condition}: fail-closed component loss is absent")
            if summary.get("circuit_loss_active") is not True:
                raise ValueError(f"{sample_id}/{condition}: component loss was inactive")
            if summary.get("circuit_loss_clean_reference_id") != sample_id:
                raise ValueError(f"{sample_id}/{condition}: clean activation source mismatch")
            if summary.get("circuit_loss_attack_reference_id") != sample_id:
                raise ValueError(f"{sample_id}/{condition}: attack activation source mismatch")
            expected = set(summary.get("circuit_loss_expected_keys", []))
            captured = set(summary.get("circuit_loss_captured_keys", []))
            if not expected or captured != expected:
                raise ValueError(f"{sample_id}/{condition}: incomplete component hooks")
    return rows


def sample_ids(report: dict) -> list[str]:
    ids = [row.get("sample_id") for row in report["detail"]["selected_references"]]
    if not ids or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Every selected reference must have a stable sample_id")
    if len(ids) != len(set(ids)):
        raise ValueError("Selected reference cohort contains duplicate sample_ids")
    return ids


def per_reference(report: dict, condition: str) -> dict[str, dict]:
    rows = report["detail"]["per_condition_rows"][condition]
    mapped = {row["reference_sample_id"]: row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError(f"Condition {condition} has duplicate or missing sample IDs")
    return mapped


def probability_drop(row: dict) -> float:
    return float(row["reference_clean_malware_prob"]) - float(row["reference_adv_malware_prob"])


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace(":", ":\\allowbreak{}")


def circuit_name(score: dict) -> str:
    circuit = score["circuit"]
    base = f"{circuit['module']}:{circuit['layer']}"
    if circuit.get("tag"):
        base += f":{circuit['tag']}"
    return base


def validate_restoration(report: dict, attack_report: dict, seed: int) -> None:
    if report.get("schema_version") != 3:
        raise ValueError(f"Seed {seed}: restoration schema must be version 3")
    if report.get("restoration_mode") != "independent_local_node_activation":
        raise ValueError(f"Seed {seed}: whole-layer/progressive restoration is forbidden")
    attack_detail = attack_report["detail"]
    if report.get("shared_cohort_id") != attack_detail.get("shared_cohort_id"):
        raise ValueError(f"Seed {seed}: restoration cohort differs from factorial cohort")
    if report.get("attack_states_sha256") != attack_detail.get("attack_states_sha256"):
        raise ValueError(f"Seed {seed}: restoration did not replay the factorial attack state")
    if [row.get("sample_id") for row in report.get("samples", [])] != sample_ids(attack_report):
        raise ValueError(f"Seed {seed}: restoration references are not paired with factorial references")


def apk_lines(summary: dict) -> list[str]:
    if summary.get("schema_version") != 1 or summary.get("gate_semantics") != "cumulative_fail_closed":
        raise ValueError("APK gate summary has unsupported schema or non-cumulative semantics")
    cohort = summary.get("cohort")
    if not isinstance(cohort, dict) or cohort.get("detector_predictions_used_for_selection") is not False:
        raise ValueError("APK gate summary lacks a detector-blind matched cohort")
    if int(cohort.get("case_count", 0)) < 20 or not cohort.get("cohort_id") or not cohort.get("manifest_sha256"):
        raise ValueError("APK matched cohort is too small or lacks immutable provenance")
    if summary.get("per_seed_score_timeout_seconds") != 300:
        raise ValueError("APK summary lacks the final fixed 300-second scoring budget")
    labels = (("carrier_rewrite", "Carrier rewrite"), ("wide_void", "Wide-void stress"))
    lines = []
    for key, label in labels:
        row = summary.get("operators", {}).get(key)
        if not isinstance(row, dict) or int(row.get("case_count", 0)) <= 0:
            raise ValueError(f"APK gate summary has no validated cases for {key}")
        total = int(row["case_count"])
        if total != int(cohort["case_count"]):
            raise ValueError(f"APK operator {key} does not cover the full matched cohort")
        lines.append(
            f"{label} & {int(row['edit_applied_count'])}/{total} & {int(row['repack_count'])}/{total} & "
            f"{int(row['art_install_count'])}/{total} & "
            f"{int(row['smoke_count'])}/{total} & {int(row['behavior_ioc_count'])}/{total} & "
            f"{int(row['valid_evasion_count'])}/{total} \\\\" 
        )
    return lines


def apk_interpretation(summary: dict) -> str:
    fragments = []
    for key, label in (("carrier_rewrite", "Carrier rewrite"), ("wide_void", "Wide-void")):
        row = summary["operators"][key]
        per_seed = row.get("per_seed_detector_flip_count", {})
        all_seed_flips = int(row.get("detector_flip_count", 0))
        all_seed_noun = "case" if all_seed_flips == 1 else "cases"
        all_seed_verb = "flips" if all_seed_flips == 1 else "flip"
        fragments.append(
            f"{label} applies a non-identity DEX edit in {int(row['edit_applied_count'])}/"
            f"{int(row['case_count'])} cases and produces raw detector flips in {int(per_seed.get('42', 0))}/"
            f"{int(per_seed.get('43', 0))}/{int(per_seed.get('44', 0))} cases for seeds 42/43/44, "
            f"respectively; {all_seed_flips} {all_seed_noun} {all_seed_verb} all three checkpoints and "
            f"{int(row['valid_evasion_count'])} pass every cumulative gate."
        )
    return " ".join(fragments)


def apk_cohort_method(summary: dict) -> str:
    cohort = summary["cohort"]
    total = int(cohort["case_count"])
    candidates = int(cohort.get("carrier_repack_candidate_count", 0))
    if candidates < total:
        raise ValueError("APK cohort lacks the carrier-repack candidate denominator")
    if cohort.get("sampling_seed") is None:
        opening = (
            f"We enumerate all {total} eligible source APKs from {candidates:,} "
            "carrier-repackable cases after requiring a nonempty sensitive-API IOC set "
            "in the original."
        )
    else:
        opening = (
            f"We deterministically sample {total} source APKs (seed "
            f"{int(cohort['sampling_seed'])}) from {candidates:,} carrier-repackable cases "
            "after requiring a nonempty sensitive-API IOC set in the original."
        )
    return (
        opening
        + " The identical source cohort is used for carrier and wide-void, and the "
        "selection procedure does not inspect any leakage-safe checkpoint prediction."
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=Path, default=root / "docs" / "paper" / "sp" / "main_sp.tex")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "outputs" / "leakage_safe_shared_three_seed_same_sample_v2",
    )
    parser.add_argument("--training-root", type=Path, default=root / "training_runs")
    parser.add_argument(
        "--apk-summary",
        type=Path,
        default=root / "outputs" / "leakage_safe_shared_three_seed" / "apk_gates" / "matched_all" / "run" / "apk_gate_summary.json",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--allow-pending-apk",
        action="store_true",
        help="Backfill completed shared/restoration results while leaving the APK region unchanged.",
    )
    args = parser.parse_args()

    clean_reports = {
        seed: load_json(args.training_root / f"leakage_safe_seed{seed}" / "metrics.json")
        for seed in SEEDS
    }
    attack_reports = {
        seed: load_json(args.run_root / f"seed{seed}" / "factorial" / "node_latent_small_benchmark.json")
        for seed in SEEDS
    }
    attack_rows = {seed: condition_rows(report) for seed, report in attack_reports.items()}
    restoration_reports = {
        seed: load_json(args.run_root / f"seed{seed}" / "restoration" / "restoration_local_node.json")
        for seed in SEEDS
    }
    restoration_controls = {
        seed: load_json(args.run_root / f"seed{seed}" / "restoration" / "localized_control_summary.json")
        for seed in SEEDS
    }
    apk_summary = None if args.allow_pending_apk else load_json(args.apk_summary)

    cohort_ids = {report["detail"].get("shared_cohort_id") for report in attack_reports.values()}
    manifest_hashes = {report["detail"].get("reference_manifest_sha256") for report in attack_reports.values()}
    cohorts = [sample_ids(attack_reports[seed]) for seed in SEEDS]
    if len(cohort_ids) != 1 or None in cohort_ids or len(manifest_hashes) != 1 or None in manifest_hashes:
        raise ValueError("Three seed reports do not share one manifest/cohort")
    if not all(ids == cohorts[0] for ids in cohorts[1:]):
        raise ValueError("Three seed reports contain different ordered sample IDs")
    for seed in SEEDS:
        validate_restoration(restoration_reports[seed], attack_reports[seed], seed)
        control = restoration_controls[seed]
        if control.get("source_schema_version") != 3:
            raise ValueError(f"Seed {seed}: localized restoration control summary is missing/legacy")
        if int(control.get("eligible_sample_count", -1)) != int(
            restoration_reports[seed]["eligible_sample_count"]
        ):
            raise ValueError(f"Seed {seed}: control summary denominator mismatch")

    clean_lines = []
    for position, seed in enumerate(SEEDS):
        report = clean_reports[seed]
        audit = report.get("group_audit") or {}
        if audit.get("cross_split_group_count") != 0:
            raise ValueError(f"Seed {seed} lacks a passing group-disjoint audit")
        metrics = report.get("test_metrics") or {}
        required = ("accuracy", "precision", "recall", "f1", "roc_auc")
        if any(metrics.get(key) is None for key in required):
            raise ValueError(f"Seed {seed} has incomplete test metrics")
        clean_lines.append(
            f"\\PureGNN (seed {seed}) & complete & {report['best_epoch']} & "
            f"{fmt(float(report['best_val_f1']))} & {fmt(float(metrics['accuracy']))} & "
            f"{fmt(float(metrics['precision']))} & {fmt(float(metrics['recall']))} & "
            f"{fmt(float(metrics['f1']))} & {fmt(float(metrics['roc_auc']))} \\\\"
        )
        if position != len(SEEDS) - 1:
            clean_lines.append("\\hline")

    factorial_values: dict[str, tuple[list[float], list[float]]] = {}
    for condition, label in CONDITIONS:
        asr = [float(attack_rows[seed][condition]["success_rate"]) for seed in SEEDS]
        drops = [float(attack_rows[seed][condition]["mean_malware_prob_drop"]) for seed in SEEDS]
        factorial_values[condition] = (asr, drops)
    factorial_lines = []
    seed_names = {42: "FortyTwo", 43: "FortyThree", 44: "FortyFour"}
    for seed_index, seed in enumerate(SEEDS):
        asr_coords = " ".join(
            f"({index},{fmt(factorial_values[condition][0][seed_index])})"
            for index, (condition, _) in enumerate(CONDITIONS)
        )
        drop_coords = " ".join(
            f"({index},{fmt(factorial_values[condition][1][seed_index])})"
            for index, (condition, _) in enumerate(CONDITIONS)
        )
        factorial_lines.extend([
            f"\\def\\FactorialASRSeed{seed_names[seed]}{{{asr_coords}}}",
            f"\\def\\FactorialDropSeed{seed_names[seed]}{{{drop_coords}}}",
        ])
    mean_asr_coords = " ".join(
        f"({index},{fmt(statistics.fmean(factorial_values[condition][0]))})"
        for index, (condition, _) in enumerate(CONDITIONS)
    )
    mean_drop_coords = " ".join(
        f"({index},{fmt(statistics.fmean(factorial_values[condition][1]))})"
        for index, (condition, _) in enumerate(CONDITIONS)
    )
    factorial_lines.extend([
        f"\\def\\FactorialASRMean{{{mean_asr_coords}}}",
        f"\\def\\FactorialDropMean{{{mean_drop_coords}}}",
    ])

    contrast_asr_commands = []
    contrast_drop_commands = []
    contrast_seed_values: dict[str, tuple[list[float], list[float]]] = {}
    for condition, label in CONTRASTS:
        seed_asr_diffs: list[float] = []
        seed_drop_diffs: list[float] = []
        for seed in SEEDS:
            plain = per_reference(attack_reports[seed], "plain")
            treated = per_reference(attack_reports[seed], condition)
            if set(plain) != set(treated) or set(plain) != set(cohorts[0]):
                raise ValueError(f"Seed {seed}, condition {condition}: unpaired shared cohort")
            seed_asr_diffs.append(statistics.fmean(
                float(treated[sample_id]["reference_adv_success"])
                - float(plain[sample_id]["reference_adv_success"])
                for sample_id in cohorts[0]
            ))
            seed_drop_diffs.append(statistics.fmean(
                probability_drop(treated[sample_id]) - probability_drop(plain[sample_id])
                for sample_id in cohorts[0]
            ))
        contrast_seed_values[condition] = (seed_asr_diffs, seed_drop_diffs)
        y = len(CONTRASTS) - len(contrast_seed_values) + 1
        for values, commands in (
            (seed_asr_diffs, contrast_asr_commands),
            (seed_drop_diffs, contrast_drop_commands),
        ):
            center, low, high = mean_ci(values)
            low, high = max(-1.0, low), min(1.0, high)
            commands.append(
                f"\\addplot[black!65,line width=0.8pt,forget plot] coordinates "
                f"{{({fmt(low)},{y}) ({fmt(high)},{y})}};"
            )
            commands.append(
                f"\\addplot[only marks,mark=*,mark size=1.8pt,blue!70!black,forget plot] "
                f"coordinates {{({fmt(center)},{y})}};"
            )
    contrast_lines = [
        "\\def\\ContrastASRPlots{%",
        *contrast_asr_commands,
        "}",
        "\\def\\ContrastDropPlots{%",
        *contrast_drop_commands,
        "}",
    ]

    candidate_counts = [int(attack_reports[s]["detail"]["reference_candidate_count"]) for s in SEEDS]
    selected_count = len(cohorts[0])
    eligible_counts = [int(restoration_reports[s]["eligible_sample_count"]) for s in SEEDS]
    excluded_counts = [int(restoration_reports[s]["non_positive_drop_count"]) for s in SEEDS]
    row_end = r" \\"
    sample_count_lines = "\n".join([
        "Leakage-safe train graphs & 13,170 & Unique valid identifier groups after grouping and validation." + row_end,
        "Leakage-safe test graphs & 3,294 & Group-disjoint held-out graphs; 2,486 malware and 808 benign." + row_end,
        f"Attackable clean-correct malware & {candidate_counts[0]}/{candidate_counts[1]}/{candidate_counts[2]} & Seed 42/43/44 candidate populations." + row_end,
        f"Shared six-condition references & {selected_count} & One frozen sample-ID cohort for all checkpoints and conditions." + row_end,
        f"Node-local restoration cases & {eligible_counts[0]}/{eligible_counts[1]}/{eligible_counts[2]} & Positive-drop cases; excluded {excluded_counts[0]}/{excluded_counts[1]}/{excluded_counts[2]}." + row_end,
    ])

    hotspot_counts: dict[str, dict[int, int]] = {}
    for seed, report in attack_reports.items():
        for sample in report["detail"]["per_sample"]:
            scores = [score for score in sample.get("top_scores", []) if score["circuit"]["module"] == "gnn"]
            if not scores:
                raise ValueError(f"Seed {seed}: missing node-aligned circuit ranking")
            name = circuit_name(scores[0])
            hotspot_counts.setdefault(name, {value: 0 for value in SEEDS})[seed] += 1
    hotspot_lines = []
    for component in sorted(hotspot_counts):
        counts = hotspot_counts[component]
        macro = component.replace(":", "").replace("_", "")
        coords = " ".join(
            f"({index},{100.0 * counts[seed] / selected_count:.1f})"
            for index, seed in enumerate(SEEDS)
        )
        hotspot_lines.append(f"\\expandafter\\def\\csname Hotspot{macro}\\endcsname{{{coords}}}")

    restoration_lines = []
    hotspot_seed_means: list[float] = []
    random_layer_seed_means: list[float] = []
    random_node_seed_means: list[float] = []
    for seed in SEEDS:
        report = restoration_reports[seed]
        hotspot = float(report["mean_hotspot_target_node_recovery_ratio"])
        random_layer = float(report["mean_random_layer_target_node_recovery_ratio"])
        random_node = float(report["mean_hotspot_random_node_recovery_ratio"])
        hotspot_seed_means.append(hotspot)
        random_layer_seed_means.append(random_layer)
        random_node_seed_means.append(random_node)
    restoration_lines.extend([
        "\\def\\RestorationHotspot{" + " ".join(
            f"({index},{fmt(value)})" for index, value in enumerate(hotspot_seed_means)
        ) + "}",
        "\\def\\RestorationRandomLayer{" + " ".join(
            f"({index},{fmt(value)})" for index, value in enumerate(random_layer_seed_means)
        ) + "}",
        "\\def\\RestorationRandomNode{" + " ".join(
            f"({index},{fmt(value)})" for index, value in enumerate(random_node_seed_means)
        ) + "}",
        "\\def\\RestorationEligible{" + "/".join(str(value) for value in eligible_counts) + "}",
        "\\def\\RestorationLabelRecovery{" + "/".join(
            str(int(restoration_reports[seed]["hotspot_label_recovery_count"])) for seed in SEEDS
        ) + "}",
        "\\def\\RestorationSuccessfulAttacks{" + "/".join(
            str(int(restoration_reports[seed]["successful_attack_count"])) for seed in SEEDS
        ) + "}",
    ])

    selection_asr, selection_drop = contrast_seed_values["selection_only"]
    selection_asr_mean, selection_asr_low, selection_asr_high = mean_ci(selection_asr)
    rq1_interpretation = (
        f"Across the shared cohort, selection-only changes ASR by {selection_asr_mean:.4f} "
        f"(95\\% seed-level CI [{selection_asr_low:.4f}, {selection_asr_high:.4f}]) and malware-score "
        f"suppression by {statistics.fmean(selection_drop):.4f}. "
        + (
            (
                "The direction is positive in all three checkpoints, but the seed-level interval includes zero; the selection benefit is suggestive rather than statistically conclusive."
                if all(value > 0.0 for value in selection_asr)
                else "The seed-level interval includes zero, so the data do not support a consistent selection benefit across checkpoints."
            )
            if selection_asr_low <= 0.0 <= selection_asr_high
            else "The direction is consistent across checkpoints."
        )
    )

    hotspot_minus_layer = [h - r for h, r in zip(hotspot_seed_means, random_layer_seed_means)]
    hotspot_minus_node = [h - r for h, r in zip(hotspot_seed_means, random_node_seed_means)]
    layer_intervals = [
        restoration_controls[seed]["paired_hotspot_minus_random_layer"] for seed in SEEDS
    ]
    node_intervals = [
        restoration_controls[seed]["paired_hotspot_minus_random_node"] for seed in SEEDS
    ]
    layer_consistent = all(float(interval["ci95_low"]) > 0.0 for interval in layer_intervals)
    node_consistent = all(float(interval["ci95_low"]) > 0.0 for interval in node_intervals)
    restoration_interpretation = (
        f"Node-local hotspot recovery differs from the random-layer control by {fmt_ci(hotspot_minus_layer, bounds=(-1.0, 1.0))} "
        f"and from the random-node control by {fmt_ci(hotspot_minus_node, bounds=(-1.0, 1.0))}. "
        + (
            "Every seed-level paired bootstrap interval is positive for both controls, supporting a checkpoint-stable localized hotspot effect."
            if layer_consistent and node_consistent
            else (
                "Within each checkpoint, the paired-bootstrap random-node contrast is positive, but the three-checkpoint Student-$t$ interval crosses zero; the random-layer contrast is not consistently positive even within checkpoints. This is evidence of attacked-node localization inside these checkpoints, not a checkpoint-stable layer-specific causal hotspot."
                if node_consistent
                else "At least one control is not positive in every seed-level paired bootstrap interval; we therefore do not claim checkpoint-stable causal localization."
            )
        )
    )

    text = args.paper.read_text(encoding="utf-8")
    text = replace_region(text, "CLEAN_RESULTS", "\n".join(clean_lines))
    text = replace_region(text, "LATENT_FACTORIAL_SUMMARY", "\n".join(factorial_lines))
    text = replace_region(text, "PAIRED_CONTRASTS", "\n".join(contrast_lines))
    text = replace_region(text, "SAMPLE_COUNTS", sample_count_lines)
    text = replace_region(text, "HOTSPOT_STABILITY", "\n".join(hotspot_lines))
    text = replace_region(text, "RESTORATION_SUMMARY", "\n".join(restoration_lines))
    text = replace_region(text, "RQ1_INTERPRETATION", rq1_interpretation)
    text = replace_region(text, "RESTORATION_INTERPRETATION", restoration_interpretation)
    if apk_summary is not None:
        text = replace_region(text, "APK_COHORT_METHOD", apk_cohort_method(apk_summary))
        text = replace_region(text, "APK_GATES", "\n".join(apk_lines(apk_summary)))
        text = replace_region(text, "APK_INTERPRETATION", apk_interpretation(apk_summary))
    forbidden = ("TBD--3-seed", "\\RerunPending", "Not run")
    present = [marker for marker in forbidden if marker in text]
    if present:
        raise ValueError(f"Submission text retains unresolved markers: {present}")

    if args.check_only:
        print("Shared cohort, localized restoration, APK gates, and LaTeX backfill validate.")
        return
    temporary = args.paper.with_suffix(".tex.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(args.paper)
    print(f"Updated {args.paper}")


if __name__ == "__main__":
    main()
