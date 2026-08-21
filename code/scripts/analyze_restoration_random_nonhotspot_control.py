#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_mean_ci(values: list[float], *, seed: int, draws: int = 10000) -> dict:
    if not values:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "count": 0}
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(values) for _ in values)
        for _ in range(draws)
    )
    low = estimates[int(0.025 * (draws - 1))]
    high = estimates[int(0.975 * (draws - 1))]
    return {"mean": mean(values), "ci95_low": low, "ci95_high": high, "count": len(values)}


def summarize(report: dict, seed: int) -> dict:
    if report.get("schema_version") != 3 or report.get("restoration_mode") != "independent_local_node_activation":
        raise ValueError(
            "This analysis requires schema-v3 independent local-node restoration; "
            "whole-layer or progressive reports are not valid controls."
        )

    hotspot: list[float] = []
    random_layer: list[float] = []
    random_node: list[float] = []
    hotspot_minus_random_layer: list[float] = []
    hotspot_minus_random_node: list[float] = []
    hotspot_stage_counts: dict[str, int] = {}

    for sample in report.get("samples", []):
        h = sample.get("hotspot_target_node_recovery_ratio")
        rl = sample.get("random_layer_target_node_recovery_ratio")
        rn = sample.get("hotspot_random_node_recovery_ratio")
        if h is None:
            continue
        h = float(h)
        hotspot.append(h)
        stage = sample.get("preselected_hotspot_stage")
        if stage:
            hotspot_stage_counts[stage] = hotspot_stage_counts.get(stage, 0) + 1
        if rl is not None:
            rl = float(rl)
            random_layer.append(rl)
            hotspot_minus_random_layer.append(h - rl)
        if rn is not None:
            rn = float(rn)
            random_node.append(rn)
            hotspot_minus_random_node.append(h - rn)

    return {
        "source_schema_version": 3,
        "source_restoration_mode": report["restoration_mode"],
        "source_sample_count": int(report.get("sample_count", 0)),
        "eligible_sample_count": len(hotspot),
        "seed": seed,
        "hotspot_stage_counts": hotspot_stage_counts,
        "hotspot_target_node": bootstrap_mean_ci(hotspot, seed=seed + 11),
        "random_layer_same_target_node": bootstrap_mean_ci(random_layer, seed=seed + 17),
        "hotspot_layer_random_node": bootstrap_mean_ci(random_node, seed=seed + 23),
        "paired_hotspot_minus_random_layer": bootstrap_mean_ci(
            hotspot_minus_random_layer, seed=seed + 29
        ),
        "paired_hotspot_minus_random_node": bootstrap_mean_ci(
            hotspot_minus_random_node, seed=seed + 31
        ),
    }


def build_markdown(summary: dict, report_path: Path) -> str:
    def line(label: str, key: str) -> str:
        value = summary[key]
        if value["mean"] is None:
            return f"- {label}: unavailable"
        return (
            f"- {label}: `{value['mean']:.4f}` "
            f"[95% bootstrap CI `{value['ci95_low']:.4f}`, `{value['ci95_high']:.4f}`], "
            f"n=`{value['count']}`"
        )

    return "\n".join([
        "# Localized Restoration Controls",
        "",
        f"- Source report: `{report_path}`",
        f"- Eligible samples: `{summary['eligible_sample_count']}` / `{summary['source_sample_count']}`",
        line("Hotspot layer, attacked node", "hotspot_target_node"),
        line("Random non-hotspot layer, same attacked node", "random_layer_same_target_node"),
        line("Hotspot layer, matched random node", "hotspot_layer_random_node"),
        line("Paired hotspot minus random layer", "paired_hotspot_minus_random_layer"),
        line("Paired hotspot minus random node", "paired_hotspot_minus_random_node"),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    report = load_json(args.report)
    summary = summarize(report, seed=args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.output_md.write_text(build_markdown(summary, args.report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
