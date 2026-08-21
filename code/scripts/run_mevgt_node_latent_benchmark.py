#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

HERE = Path(__file__).resolve().parent
BLUEPRINT_ROOT = HERE.parent
SRC_ROOT = BLUEPRINT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from android_llm_gnn.mevgt import (  # noqa: E402
    build_android_mevgt_adapter,
)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(BLUEPRINT_ROOT.resolve()))
    except ValueError:
        return resolved.name


def ensure_external_imports(cfg: dict) -> None:
    task_cfg = cfg.get("task", {})
    mevgt_root = task_cfg.get("mevgt_root")
    if mevgt_root:
        resolved = str(Path(mevgt_root).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def select_reference_summaries(
    summaries: List[Dict[str, Any]],
    subset_size: int | None,
    *,
    seed: int,
    mode: str = "deterministic_random",
) -> List[Dict[str, Any]]:
    """Select a reproducible evaluation subset without dataset-order bias."""
    limit = len(summaries) if subset_size is None else min(len(summaries), int(subset_size))
    if limit <= 0:
        return []
    if mode == "head":
        return list(summaries[:limit])
    if mode != "deterministic_random":
        raise ValueError(f"Unsupported reference_sampling={mode!r}")
    indices = random.Random(int(seed)).sample(range(len(summaries)), limit)
    return [summaries[index] for index in indices]


def select_reference_summaries_from_manifest(
    summaries: List[Dict[str, Any]], manifest: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Resolve a frozen cross-checkpoint cohort by stable sample identifier."""
    sample_ids = manifest.get("sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids:
        raise ValueError("Shared reference manifest must contain a non-empty sample_ids list")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Shared reference manifest contains duplicate sample_ids")

    by_id: Dict[str, Dict[str, Any]] = {}
    for summary in summaries:
        sample_id = summary.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            continue
        if sample_id in by_id:
            raise ValueError(f"Duplicate clean-correct sample_id in checkpoint candidate pool: {sample_id}")
        by_id[sample_id] = summary

    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Checkpoint is not clean-correct on {len(missing)} frozen shared references: {preview}"
        )
    return [by_id[sample_id] for sample_id in sample_ids]


def summarize_condition(
    name: str,
    reference_index: int,
    reference_summary: Dict[str, Any],
    clean_metrics,
    ref_clean,
    adv_result,
    adv_eval,
    top_circuits: List[Any],
    candidate_count: int,
    candidate_selection: str,
) -> Dict[str, Any]:
    info = getattr(adv_result, "info", {}) if adv_result is not None else {}
    adv_metrics = getattr(adv_eval, "metrics", {}) if adv_eval is not None else {}
    return {
        "condition": name,
        "reference_index": int(reference_index),
        "reference_sample_id": reference_summary.get("sample_id"),
        "reference_file_name": reference_summary.get("file_name"),
        "reference_num_nodes": int(reference_summary.get("num_nodes", 0)),
        "reference_num_edges": int(reference_summary.get("num_edges", 0)),
        "global_clean_f1": float(clean_metrics.metrics.get("f1", clean_metrics.primary)),
        "reference_clean_malware_prob": float(ref_clean.metrics.get("malware_prob", ref_clean.primary)),
        "reference_adv_malware_prob": float(adv_metrics.get("malware_prob", info.get("adv_primary", 0.0))),
        "reference_clean_benign_prob": float(ref_clean.metrics.get("benign_prob", 0.0)),
        "reference_adv_benign_prob": float(adv_metrics.get("benign_prob", 0.0)),
        "reference_adv_success": float(adv_metrics.get("success", 0.0)),
        "reference_adv_margin": float(adv_metrics.get("margin", 0.0)),
        "budget_used": int(info.get("budget_used", 0)),
        "candidate_count": int(info.get("candidate_count", candidate_count)),
        "active_candidates": int(info.get("active_candidates", 0)),
        "steps": int(info.get("steps", 0)),
        "eps": float(info.get("eps", 0.0)),
        "lr": float(info.get("lr", 0.0)),
        "circuit_loss_weight": float(info.get("circuit_loss_weight", 0.0)),
        "circuit_loss_requested": bool(info.get("circuit_loss_requested", False)),
        "circuit_loss_required": bool(info.get("circuit_loss_required", False)),
        "circuit_loss_active": bool(info.get("circuit_loss_active", False)),
        "circuit_loss_clean_reference_id": info.get("circuit_loss_clean_reference_id"),
        "circuit_loss_attack_reference_id": info.get("circuit_loss_attack_reference_id"),
        "circuit_loss_expected_keys": list(info.get("circuit_loss_expected_keys", [])),
        "circuit_loss_captured_keys": list(info.get("circuit_loss_captured_keys", [])),
        "attack_scheme": info.get("attack_scheme"),
        "candidate_selection": candidate_selection,
        "target_circuits": [
            f"{c.module}:{c.layer}" + (f":{c.tag}" if c.tag else "")
            for c in top_circuits
        ],
    }


def aggregate_rows(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"condition": name, "num_samples": 0}

    def mean(key: str) -> float:
        values = [float(row.get(key, 0.0)) for row in rows]
        return float(sum(values) / len(values))

    return {
        "condition": name,
        "num_samples": len(rows),
        "success_rate": mean("reference_adv_success"),
        "mean_clean_malware_prob": mean("reference_clean_malware_prob"),
        "mean_adv_malware_prob": mean("reference_adv_malware_prob"),
        "mean_clean_benign_prob": mean("reference_clean_benign_prob"),
        "mean_adv_benign_prob": mean("reference_adv_benign_prob"),
        "mean_malware_prob_drop": mean("reference_clean_malware_prob") - mean("reference_adv_malware_prob"),
        "mean_adv_margin": mean("reference_adv_margin"),
        "mean_budget_used": mean("budget_used"),
        "mean_candidate_count": mean("candidate_count"),
        "mean_active_candidates": mean("active_candidates"),
        "mean_steps": mean("steps"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=BLUEPRINT_ROOT / "configs" / "mevgt" / "node_latent_smoke.yaml",
    )
    parser.add_argument(
        "--condition-set",
        choices=("factorial", "legacy"),
        default="factorial",
        help="Run the reviewer-safe factorial ablation or the historical three-way comparison.",
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        help="Frozen cross-checkpoint sample_id cohort; disables per-seed reference sampling.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Override config out_dir so reviewer-safe reruns never overwrite historical reports.",
    )
    parser.add_argument(
        "--reference-limit",
        type=int,
        help="Use only the first N manifest references for implementation smoke tests.",
    )
    parser.add_argument("--device", choices=("cpu", "mps"), help="Override task device.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_external_imports(cfg)

    from mevgt.algorithms.alg1_circuit_discovery import discover_circuits  # noqa: E402
    from mevgt.algorithms.alg2_circuit_guided_edit import craft_circuit_guided_adv  # noqa: E402

    task_cfg = dict(cfg.get("task", {}))
    task_cfg.pop("mevgt_root", None)
    if args.device:
        task_cfg["device"] = args.device
    adapter = build_android_mevgt_adapter(task_cfg)
    adapter.setup()

    split = cfg.get("eval_split", "test")
    clean_metrics = adapter.evaluate(split)

    base_selection = str(cfg.get("alg2", {}).get("candidate_selection", "degree_internal_first"))
    guided_weight = float(cfg.get("alg2", {}).get("circuit_loss_weight", 0.15))
    if args.condition_set == "legacy":
        conditions = [
            (
                "random_noise",
                {"alg2": {"attack_scheme": "random_input", "candidate_selection": base_selection, "circuit_loss_weight": 0.0}},
            ),
            (
                "plain",
                {"alg2": {"attack_scheme": "pgd_latent", "candidate_selection": base_selection, "circuit_loss_weight": 0.0}},
            ),
            (
                "selection_plus_loss",
                {"alg2": {"attack_scheme": "pgd_latent", "candidate_selection": "circuit_guided", "circuit_loss_weight": guided_weight}},
            ),
        ]
    else:
        # Factorial decomposition: candidate selection and circuit-aware loss
        # are varied independently. Random-node PGD and random-noise are kept
        # separate because they answer different control questions.
        conditions = [
            (
                "random_noise",
                {"alg2": {"attack_scheme": "random_input", "candidate_selection": base_selection, "circuit_loss_weight": 0.0}},
            ),
            (
                "random_node",
                {"alg2": {"attack_scheme": "pgd_latent", "candidate_selection": "random_node", "circuit_loss_weight": 0.0}},
            ),
            (
                "plain",
                {"alg2": {"attack_scheme": "pgd_latent", "candidate_selection": base_selection, "circuit_loss_weight": 0.0}},
            ),
            (
                "selection_only",
                {"alg2": {"attack_scheme": "pgd_latent", "candidate_selection": "circuit_guided", "circuit_loss_weight": 0.0}},
            ),
            (
                "loss_only",
                {"alg2": {"attack_scheme": "pgd_latent", "candidate_selection": base_selection, "circuit_loss_weight": guided_weight}},
            ),
            (
                "selection_plus_loss",
                {"alg2": {"attack_scheme": "pgd_latent", "candidate_selection": "circuit_guided", "circuit_loss_weight": guided_weight}},
            ),
        ]

    reference_summaries = adapter.reference_candidate_summaries(split)
    subset_size = adapter.cfg.attack_subset_size or len(reference_summaries)
    sampling_mode = str(task_cfg.get("reference_sampling", "deterministic_random"))
    sampling_seed = int(cfg.get("seed", task_cfg.get("random_seed", 0)))
    reference_manifest = None
    if args.reference_manifest:
        reference_manifest = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
        selected_summaries = select_reference_summaries_from_manifest(
            reference_summaries, reference_manifest
        )
        if args.reference_limit is not None:
            if args.reference_limit < 1:
                raise ValueError("--reference-limit must be positive")
            selected_summaries = selected_summaries[: args.reference_limit]
        sampling_mode = "shared_manifest"
    else:
        selected_summaries = select_reference_summaries(
            reference_summaries,
            subset_size,
            seed=sampling_seed,
            mode=sampling_mode,
        )
    if not selected_summaries:
        raise ValueError(f"No attackable references available for split={split}")

    aggregated_by_condition: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _ in conditions}
    per_sample: List[Dict[str, Any]] = []
    replay_states: Dict[str, Dict[str, Any]] = {}

    for sample_meta in selected_summaries:
        reference_index = int(sample_meta["reference_index"])
        adapter.set_reference_index(split, reference_index)
        # Primary mechanism claims are restricted to node-aligned GNN stages.
        # Restoring a graph-level head tensor would trivially replace the whole
        # downstream representation and cannot support a localized control.
        circuits = [circuit for circuit in adapter.list_circuits() if circuit.module == "gnn"]
        if not circuits:
            raise ValueError("No node-aligned GNN circuits are exposed by the adapter")
        discovery_cfg = copy.deepcopy(cfg)
        discovery_cfg.setdefault("alg1", {})["topk"] = len(circuits)
        if hasattr(adapter, "discover_circuits_via_clean_ablation"):
            ranked_scores = adapter.discover_circuits_via_clean_ablation(
                split=split, circuits=circuits, cfg=discovery_cfg
            )
        else:
            ranked_scores = discover_circuits(
                model=adapter.model(),
                hook_specs=adapter.hook_specs(),
                eval_fn=lambda: adapter.evaluate_with_node_override(split, overrides={}),
                circuits=circuits,
                ablate_fn=adapter.ablate_circuit,
                topk=len(circuits),
            )
        primary_topk = max(1, int(cfg.get("alg1", {}).get("topk", 1)))
        top_scores = ranked_scores[:primary_topk]
        top_circuits = [score.circuit for score in top_scores]

        sample_detail: Dict[str, Any] = {
            "reference_index": reference_index,
            "reference_summary": sample_meta,
            "top_scores": [
                {
                    "circuit": {
                        "module": score.circuit.module,
                        "layer": score.circuit.layer,
                        "head": score.circuit.head,
                        "tag": score.circuit.tag,
                    },
                    "score": score.score,
                    "detail": score.detail,
                }
                for score in ranked_scores
            ],
            "attack_topk": primary_topk,
            "conditions": {},
        }

        for name, overrides in conditions:
            condition_cfg = copy.deepcopy(cfg)
            _deep_update(condition_cfg, overrides)
            attack_problem = adapter.build_attack_problem(split=split, cfg=condition_cfg, top_scores=top_scores)
            attack_problem["space"].setdefault("split", split)
            attack_problem["space"].setdefault(
                "circuit_loss_weight",
                float(condition_cfg.get("alg2", {}).get("circuit_loss_weight", 0.0)),
            )
            ref_clean = attack_problem["objective_fn"]({})
            adv_result = craft_circuit_guided_adv(
                task_adapter=adapter,
                g=attack_problem.get("graph"),
                space=attack_problem["space"],
                target_circuits=top_circuits,
                objective_fn=attack_problem["objective_fn"],
                attack_loss_fn=attack_problem["attack_loss_fn"],
                targets=attack_problem.get("targets", []),
                max_steps=condition_cfg.get("alg2", {}).get("max_steps", 20),
                selection_objective_fn=attack_problem.get("selection_objective_fn"),
                final_eval_fn=attack_problem.get("adv_eval_fn") or attack_problem.get("objective_fn"),
                clean_forward_fn=attack_problem.get("clean_forward_fn"),
            )
            adv_eval = (attack_problem.get("adv_eval_fn") or attack_problem.get("objective_fn"))(
                adv_result.delta_overrides
            )
            row = summarize_condition(
                name=name,
                reference_index=reference_index,
                reference_summary=sample_meta,
                clean_metrics=clean_metrics,
                ref_clean=ref_clean,
                adv_result=adv_result,
                adv_eval=adv_eval,
                top_circuits=top_circuits,
                candidate_count=len(attack_problem["space"].get("candidate_ids", [])),
                candidate_selection=str(attack_problem["space"].get("candidate_selection", "unknown")),
            )
            expects_component_loss = float(
                condition_cfg.get("alg2", {}).get("circuit_loss_weight", 0.0)
            ) > 0.0
            if expects_component_loss:
                if not row["circuit_loss_active"]:
                    raise RuntimeError(
                        f"{name}: component loss requested but inactive for "
                        f"sample={sample_meta.get('sample_id')}"
                    )
                if row["circuit_loss_clean_reference_id"] != sample_meta.get("sample_id"):
                    raise RuntimeError(
                        f"{name}: clean activation source mismatch for "
                        f"sample={sample_meta.get('sample_id')}"
                    )
                if row["circuit_loss_attack_reference_id"] != sample_meta.get("sample_id"):
                    raise RuntimeError(
                        f"{name}: attack activation source mismatch for "
                        f"sample={sample_meta.get('sample_id')}"
                    )
                if set(row["circuit_loss_expected_keys"]) != set(row["circuit_loss_captured_keys"]):
                    raise RuntimeError(
                        f"{name}: incomplete component hooks for "
                        f"sample={sample_meta.get('sample_id')}"
                    )
            aggregated_by_condition[name].append(row)
            sample_detail["conditions"][name] = {
                "summary": row,
                "adv_info": adv_result.info,
                "reference_clean": {
                    "primary": ref_clean.primary,
                    "metrics": ref_clean.metrics,
                },
                "reference_adv": {
                    "primary": adv_eval.primary,
                    "metrics": adv_eval.metrics,
                },
            }
            if name == "selection_plus_loss":
                sample_id = sample_meta.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError("Every replayable attack reference must have a stable sample_id")
                replay_states[sample_id] = {
                    "sample_id": sample_id,
                    "reference_index": int(reference_index),
                    "condition": name,
                    "candidate_selection": row["candidate_selection"],
                    "reference_clean_malware_prob": row["reference_clean_malware_prob"],
                    "reference_adv_malware_prob": row["reference_adv_malware_prob"],
                    "reference_adv_success": row["reference_adv_success"],
                    "delta_overrides": {
                        int(node_idx): delta.detach().cpu().clone()
                        for node_idx, delta in adv_result.delta_overrides.items()
                    },
                }

        per_sample.append(sample_detail)

    adapter.teardown()

    rows = [aggregate_rows(name, aggregated_by_condition[name]) for name, _ in conditions]
    out_dir = args.out_dir or Path(cfg.get("out_dir", BLUEPRINT_ROOT / "outputs" / "mevgt"))
    out_dir.mkdir(parents=True, exist_ok=True)
    attack_states_path = out_dir / "selection_plus_loss_attack_states.pt"
    torch.save(
        {
            "schema_version": 1,
            "condition": "selection_plus_loss",
            "sample_ids": [row["sample_id"] for row in selected_summaries],
            "states": replay_states,
        },
        attack_states_path,
    )
    checkpoint_path = Path(task_cfg["checkpoint_path"])
    dataset_stats_path = Path(task_cfg["train_dataset"]).parent / "stats.json"
    dataset_stats = json.loads(dataset_stats_path.read_text()) if dataset_stats_path.is_file() else None
    detail = {
        "schema_version": 2,
        "config": artifact_path(args.config),
        "checkpoint": artifact_path(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset_stats": dataset_stats,
        "condition_set": args.condition_set,
        "condition_definitions": {name: overrides for name, overrides in conditions},
        "split": split,
        "global_clean_metrics": clean_metrics.metrics,
        "reference_candidate_count": len(reference_summaries),
        "selected_reference_count": len(selected_summaries),
        "reference_sampling": sampling_mode,
        "reference_sampling_seed": sampling_seed,
        "reference_manifest": (
            artifact_path(args.reference_manifest) if args.reference_manifest else None
        ),
        "reference_manifest_sha256": (
            sha256_file(args.reference_manifest) if args.reference_manifest else None
        ),
        "shared_cohort_id": (
            reference_manifest.get("cohort_id") if reference_manifest else None
        ),
        "selected_references": selected_summaries,
        "per_sample": per_sample,
        "per_condition_rows": aggregated_by_condition,
        "attack_states": artifact_path(attack_states_path),
        "attack_states_sha256": sha256_file(attack_states_path),
    }

    out_path = out_dir / "node_latent_small_benchmark.json"
    out_path.write_text(json.dumps({"rows": rows, "detail": detail}, indent=2))
    print(json.dumps({"rows": rows, "report_path": str(out_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
