#!/usr/bin/env python3
"""Audit whether latent attack targets are the methods edited in APK space.

The script deliberately performs no fuzzy matching.  It joins the frozen
latent attack states to the frozen APK cohort through the serialized PyG
dataset, resolves the attacked node through ``node_metadata``, and compares
that exact method key with the methods recorded by the carrier-rewrite and
wide-void materializers.

This is a mapping audit, not an embedding-direction analysis.  A later stage
must re-extract original and modified APK graphs and compare their embedding
displacements with the serialized latent deltas.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalise_method_key(value: object) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _class_key(method_key: str) -> str:
    return method_key.split("->", 1)[0] if "->" in method_key else ""


def _parse_seed_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SEED=PATH")
    seed, raw_path = value.split("=", 1)
    seed = seed.strip()
    if not seed:
        raise argparse.ArgumentTypeError("seed must not be empty")
    return seed, Path(raw_path).expanduser()


def _dataset_index(dataset_path: Path) -> dict[str, Any]:
    rows = torch.load(dataset_path, map_location="cpu", weights_only=False)
    index: dict[str, Any] = {}
    for row in rows:
        sample_id = str(getattr(row, "sample_id", ""))
        if not sample_id or sample_id in index:
            raise ValueError(f"missing or duplicate dataset sample_id: {sample_id!r}")
        index[sample_id] = row
    return index


def _cohort_index(manifest_path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(manifest_path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{manifest_path} does not contain a rows list")
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        file_name = Path(str(row.get("file_name", ""))).name
        key = Path(file_name).stem
        if not key or key in index:
            raise ValueError(f"missing or duplicate cohort file key: {key!r}")
        index[key] = row
    return index


def _carrier_report_index(root: Path) -> dict[tuple[str, str], Path]:
    index: dict[tuple[str, str], Path] = {}
    for path in root.rglob("carrier_rewrite_execution_report.json"):
        relative = path.relative_to(root)
        if len(relative.parts) < 3:
            continue
        key = (relative.parts[0], relative.parts[1])
        if key in index:
            raise ValueError(f"duplicate carrier report key {key}: {path}")
        index[key] = path
    return index


def _wide_report_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("rewrite_report.json"):
        key = path.parent.name
        if key in index:
            raise ValueError(f"duplicate wide-void report key {key}: {path}")
        index[key] = path
    return index


def _carrier_lookup_key(cohort_row: dict[str, Any]) -> tuple[str, str]:
    modified = Path(str(cohort_row.get("carrier_modified_apk", "")))
    parts = modified.parts
    try:
        marker = parts.index("carrier_rewrite_stage3_joint")
        return parts[marker + 1], parts[marker + 2]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"cannot derive carrier scenario from {modified}") from exc


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "joined_cases": len(rows),
        "one_attacked_node_cases": sum(row.get("attack_node_count") == 1 for row in rows),
    }
    resolved_rows = [row for row in rows if row.get("node_metadata_resolved")]
    summary["node_metadata_resolved"] = {
        "count": len(resolved_rows),
        "denominator": len(rows),
        "rate": (len(resolved_rows) / len(rows)) if rows else None,
    }
    report_fields = ("carrier_report_found", "wide_report_found")
    for field in report_fields:
        count = sum(bool(row.get(field)) for row in resolved_rows)
        summary[field] = {
            "count": count,
            "denominator": len(resolved_rows),
            "rate": (count / len(resolved_rows)) if resolved_rows else None,
        }
    boolean_fields = (
        "carrier_attacked_is_anchor",
        "carrier_attacked_is_source_target",
        "carrier_attacked_is_carrier",
        "carrier_attacked_exactly_touched",
        "carrier_attacked_class_touched",
        "wide_attacked_exactly_touched",
        "wide_attacked_class_touched",
    )
    for field in boolean_fields:
        report_field = "wide_report_found" if field.startswith("wide_") else "carrier_report_found"
        eligible = [row for row in resolved_rows if row.get(report_field)]
        count = sum(bool(row.get(field)) for row in eligible)
        summary[field] = {
            "count": count,
            "denominator": len(eligible),
            "rate": (count / len(eligible)) if eligible else None,
        }
    summary["carrier_transplant_modes"] = dict(
        Counter(str(row.get("carrier_transplant_mode", "missing")) for row in rows)
    )
    for operator in ("carrier", "wide"):
        eligible = [
            row
            for row in resolved_rows
            if row.get(f"{operator}_report_found")
            and row.get(f"{operator}_random_node_touch_probability") is not None
        ]
        if eligible:
            observed = sum(
                bool(row.get(f"{operator}_attacked_exactly_touched")) for row in eligible
            ) / len(eligible)
            random_expectation = sum(
                float(row[f"{operator}_random_node_touch_probability"]) for row in eligible
            ) / len(eligible)
            summary[f"{operator}_attacked_vs_random_touch"] = {
                "observed_attacked_touch_rate": observed,
                "mean_random_node_touch_probability": random_expectation,
                "difference": observed - random_expectation,
                "denominator": len(eligible),
            }
    return summary


def audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _dataset_index(args.dataset)
    cohort = _cohort_index(args.cohort_manifest)
    carrier_reports = _carrier_report_index(args.carrier_report_root)
    wide_reports = _wide_report_index(args.wide_report_root)

    seed_results: dict[str, Any] = {}
    for seed, state_path in args.attack_state:
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        states = payload.get("states")
        if not isinstance(states, dict):
            raise ValueError(f"{state_path} does not contain a states mapping")
        rows: list[dict[str, Any]] = []
        missing_dataset = 0
        outside_cohort = 0
        for state_sample_id, state in states.items():
            graph = dataset.get(str(state_sample_id))
            if graph is None:
                missing_dataset += 1
                continue
            file_key = Path(str(getattr(graph, "file_name", ""))).stem
            cohort_row = cohort.get(file_key)
            if cohort_row is None:
                outside_cohort += 1
                continue

            delta_overrides = state.get("delta_overrides") or {}
            attacked_indices = sorted(int(index) for index in delta_overrides)
            if len(attacked_indices) > 1:
                raise ValueError(f"attack budget exceeded for {state_sample_id}: {attacked_indices}")
            attacked_index = attacked_indices[0] if attacked_indices else None
            node_metadata = getattr(graph, "node_metadata", None)
            node_resolved = (
                attacked_index is not None
                and
                isinstance(node_metadata, list)
                and 0 <= attacked_index < len(node_metadata)
                and isinstance(node_metadata[attacked_index], dict)
            )
            attacked_method = ""
            if node_resolved:
                attacked_method = _normalise_method_key(node_metadata[attacked_index].get("key"))
            attacked_class = _class_key(attacked_method)
            graph_method_keys = {
                _normalise_method_key(item.get("key"))
                for item in (node_metadata or [])
                if isinstance(item, dict) and item.get("key")
            }

            result: dict[str, Any] = {
                "seed": seed,
                "state_sample_id": str(state_sample_id),
                "cohort_sample_id": str(cohort_row.get("sample_id", "")),
                "file_key": file_key,
                "attack_node_count": len(attacked_indices),
                "attacked_node_index": attacked_index,
                "attacked_method_key": attacked_method,
                "attacked_class_key": attacked_class,
                "node_metadata_resolved": node_resolved,
                "graph_method_count": len(graph_method_keys),
            }

            carrier_key = _carrier_lookup_key(cohort_row)
            carrier_path = carrier_reports.get(carrier_key)
            result["carrier_report_found"] = carrier_path is not None
            if carrier_path is not None:
                report = _read_json(carrier_path)
                anchor = _normalise_method_key(report.get("anchor_method_key"))
                source = _normalise_method_key(report.get("source_method_key"))
                carrier = _normalise_method_key(report.get("carrier_method_key"))
                transplant = report.get("carrier_body_transplant") or {}
                transplant_applied = bool(transplant.get("transplant_applied"))
                touched = {anchor}
                if transplant_applied:
                    touched.add(carrier)
                touched.discard("")
                touched_classes = {_class_key(key) for key in touched}
                touched_graph_nodes = touched & graph_method_keys
                result.update(
                    {
                        "carrier_anchor_method_key": anchor,
                        "carrier_source_target_method_key": source,
                        "carrier_method_key": carrier,
                        "carrier_transplant_mode": str(transplant.get("transplant_mode", "missing")),
                        "carrier_transplant_applied": transplant_applied,
                        "carrier_attacked_is_anchor": attacked_method == anchor and bool(anchor),
                        "carrier_attacked_is_source_target": attacked_method == source and bool(source),
                        "carrier_attacked_is_carrier": attacked_method == carrier and bool(carrier),
                        "carrier_attacked_exactly_touched": attacked_method in touched,
                        "carrier_attacked_class_touched": attacked_class in touched_classes
                        and bool(attacked_class),
                        "carrier_touched_graph_node_count": len(touched_graph_nodes),
                        "carrier_random_node_touch_probability": (
                            len(touched_graph_nodes) / len(graph_method_keys)
                            if graph_method_keys
                            else None
                        ),
                    }
                )

            original_name = Path(str(cohort_row.get("original_apk", ""))).name
            wide_path = wide_reports.get(original_name)
            result["wide_report_found"] = wide_path is not None
            if wide_path is not None:
                report = _read_json(wide_path)
                applied = report.get("applied") or []
                touched = {
                    _normalise_method_key(item.get("method_key"))
                    for item in applied
                    if isinstance(item, dict)
                }
                touched.discard("")
                touched_classes = {_class_key(key) for key in touched}
                touched_graph_nodes = touched & graph_method_keys
                result.update(
                    {
                        "wide_touched_method_count": len(touched),
                        "wide_attacked_exactly_touched": attacked_method in touched,
                        "wide_attacked_class_touched": attacked_class in touched_classes
                        and bool(attacked_class),
                        "wide_touched_graph_node_count": len(touched_graph_nodes),
                        "wide_random_node_touch_probability": (
                            len(touched_graph_nodes) / len(graph_method_keys)
                            if graph_method_keys
                            else None
                        ),
                    }
                )
            rows.append(result)

        seed_results[seed] = {
            "attack_state_path": str(state_path),
            "attack_state_count": len(states),
            "missing_dataset_count": missing_dataset,
            "outside_apk_cohort_count": outside_cohort,
            "summary": _summarise(rows),
            "rows": rows,
        }

    return {
        "schema_version": 1,
        "analysis_kind": "latent_to_apk_exact_method_mapping_audit",
        "scope_note": (
            "Exact method/class mapping only. This report does not establish embedding "
            "direction alignment, instruction-space reachability, or functionality preservation."
        ),
        "inputs": {
            "dataset": str(args.dataset),
            "cohort_manifest": str(args.cohort_manifest),
            "carrier_report_root": str(args.carrier_report_root),
            "wide_report_root": str(args.wide_report_root),
        },
        "seeds": seed_results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--attack-state",
        action="append",
        type=_parse_seed_path,
        required=True,
        metavar="SEED=PATH",
    )
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--carrier-report-root", type=Path, required=True)
    parser.add_argument("--wide-report-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    for seed, payload in result["seeds"].items():
        summary = payload["summary"]
        carrier = summary["carrier_attacked_exactly_touched"]
        wide = summary["wide_attacked_exactly_touched"]
        print(
            f"seed={seed} joined={summary['joined_cases']} "
            f"carrier_exact={carrier['count']}/{carrier['denominator']} "
            f"wide_exact={wide['count']}/{wide['denominator']}"
        )


if __name__ == "__main__":
    main()
