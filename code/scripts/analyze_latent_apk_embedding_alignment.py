#!/usr/bin/env python3
"""Measure direction agreement between latent deltas and realized APK edits."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import torch


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _normalise_method_key(value: object) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _parse_seed_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected SEED=PATH")
    seed, raw_path = value.split("=", 1)
    return seed.strip(), Path(raw_path).expanduser()


def _artifact_methods(path: Path) -> dict[str, torch.Tensor]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    graph = artifact.get("graph") or {}
    features = torch.as_tensor(graph.get("node_features"), dtype=torch.float32)
    metadata = graph.get("node_metadata") or []
    if features.ndim != 2 or features.shape[0] != len(metadata):
        raise ValueError(f"feature/metadata mismatch: {path}")
    result: dict[str, torch.Tensor] = {}
    for index, item in enumerate(metadata):
        if not isinstance(item, dict):
            continue
        key = _normalise_method_key(item.get("key"))
        if not key:
            continue
        if key in result:
            raise ValueError(f"duplicate method key {key} in {path}")
        result[key] = features[index]
    return result


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    if denominator <= 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _summarise(rows: list[dict[str, Any]], operator: str) -> dict[str, Any]:
    operator_rows = [row for row in rows if row.get("operator") == operator]
    eligible = [
        row for row in operator_rows if row.get("cosine") is not None
    ]
    cosines = [float(row["cosine"]) for row in eligible]
    apk_norms = [float(row["apk_delta_l2"]) for row in eligible]
    relative_distances = [float(row["relative_target_distance"]) for row in eligible]
    percentiles = [
        float(row["cosine_percentile_among_changed_nodes"])
        for row in eligible
        if row.get("cosine_percentile_among_changed_nodes") is not None
    ]
    touched = [row for row in eligible if row.get("materializer_exact_method_touch")]
    untouched = [row for row in eligible if not row.get("materializer_exact_method_touch")]
    return {
        "row_count": len(operator_rows),
        "eligible_count": len(eligible),
        "zero_or_undefined_direction_count": len(operator_rows) - len(eligible),
        "attacked_embedding_changed_rate": (
            len(eligible) / len(operator_rows) if operator_rows else None
        ),
        "mean_cosine": _mean(cosines),
        "median_cosine": _median(cosines),
        "positive_cosine_rate": (
            sum(value > 0.0 for value in cosines) / len(cosines) if cosines else None
        ),
        "mean_apk_delta_l2": _mean(apk_norms),
        "median_relative_target_distance": _median(relative_distances),
        "mean_cosine_percentile_among_changed_nodes": _mean(percentiles),
        "exactly_touched_count": len(touched),
        "exactly_touched_mean_cosine": _mean([float(row["cosine"]) for row in touched]),
        "not_exactly_touched_count": len(untouched),
        "not_exactly_touched_mean_cosine": _mean(
            [float(row["cosine"]) for row in untouched]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-audit", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--attack-state",
        action="append",
        type=_parse_seed_path,
        required=True,
        metavar="SEED=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mapping = _read_json(args.mapping_audit)
    attack_states: dict[str, dict[str, Any]] = {}
    for seed, path in args.attack_state:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        states = payload.get("states")
        if not isinstance(states, dict):
            raise ValueError(f"{path} does not contain states")
        attack_states[seed] = states

    all_rows: list[dict[str, Any]] = []
    seed_summaries: dict[str, Any] = {}
    for seed, seed_payload in (mapping.get("seeds") or {}).items():
        if seed not in attack_states:
            continue
        seed_rows: list[dict[str, Any]] = []
        for mapping_row in seed_payload.get("rows", []):
            if not mapping_row.get("node_metadata_resolved"):
                continue
            sample_id = str(mapping_row["cohort_sample_id"])
            state_sample_id = str(mapping_row["state_sample_id"])
            state = attack_states[seed].get(state_sample_id)
            if not isinstance(state, dict):
                continue
            overrides = state.get("delta_overrides") or {}
            if len(overrides) != 1:
                continue
            latent_delta = torch.as_tensor(next(iter(overrides.values())), dtype=torch.float32)
            sample_dir = args.artifact_dir / sample_id
            original_path = sample_dir / "original.pt"
            if not original_path.is_file():
                continue
            original = _artifact_methods(original_path)
            attacked_method = _normalise_method_key(mapping_row.get("attacked_method_key"))
            if attacked_method not in original:
                continue
            original_vector = original[attacked_method]
            for operator in ("carrier_rewrite", "wide_void"):
                modified_path = sample_dir / f"{operator}.pt"
                if not modified_path.is_file():
                    continue
                modified = _artifact_methods(modified_path)
                if attacked_method not in modified:
                    continue
                if original_vector.shape != latent_delta.shape:
                    raise ValueError(
                        f"latent dimension mismatch for seed {seed}, sample {sample_id}"
                    )
                common = sorted(set(original) & set(modified))
                deltas = {
                    key: modified[key] - original[key]
                    for key in common
                    if original[key].shape == modified[key].shape
                }
                apk_delta = deltas[attacked_method]
                apk_norm = float(torch.linalg.vector_norm(apk_delta))
                latent_norm = float(torch.linalg.vector_norm(latent_delta))
                cosine = _cosine(latent_delta, apk_delta)
                target_distance = float(torch.linalg.vector_norm(apk_delta - latent_delta))
                changed_cosines = sorted(
                    value
                    for value in (_cosine(latent_delta, delta) for delta in deltas.values())
                    if value is not None
                )
                percentile = None
                if cosine is not None and changed_cosines:
                    percentile = sum(value <= cosine for value in changed_cosines) / len(
                        changed_cosines
                    )
                norm_rank = None
                changed_norms = sorted(
                    (float(torch.linalg.vector_norm(delta)) for delta in deltas.values()),
                    reverse=True,
                )
                if changed_norms:
                    norm_rank = 1 + sum(value > apk_norm for value in changed_norms)
                prefix = "carrier" if operator == "carrier_rewrite" else "wide"
                row = {
                    "seed": seed,
                    "cohort_sample_id": sample_id,
                    "state_sample_id": state_sample_id,
                    "operator": operator,
                    "attacked_method_key": attacked_method,
                    "materializer_exact_method_touch": bool(
                        mapping_row.get(f"{prefix}_attacked_exactly_touched")
                    ),
                    "latent_delta_l2": latent_norm,
                    "latent_delta_linf": float(torch.linalg.vector_norm(latent_delta, ord=math.inf)),
                    "apk_delta_l2": apk_norm,
                    "apk_delta_linf": float(torch.linalg.vector_norm(apk_delta, ord=math.inf)),
                    "cosine": cosine,
                    "target_distance_l2": target_distance,
                    "relative_target_distance": (
                        target_distance / latent_norm if latent_norm > 0.0 else None
                    ),
                    "common_method_count": len(deltas),
                    "attacked_method_apk_delta_norm_rank": norm_rank,
                    "cosine_percentile_among_changed_nodes": percentile,
                }
                seed_rows.append(row)
                all_rows.append(row)
        seed_summaries[seed] = {
            operator: _summarise(seed_rows, operator)
            for operator in ("carrier_rewrite", "wide_void")
        }

    output = {
        "schema_version": 1,
        "analysis_kind": "latent_to_apk_embedding_direction_alignment",
        "scope_note": (
            "Distances are to the two observed operator outcomes, not to the full set of "
            "semantics-preserving program transformations. They therefore do not estimate "
            "the globally nearest realizable instruction sequence."
        ),
        "seeds": seed_summaries,
        "rows": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(seed_summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
