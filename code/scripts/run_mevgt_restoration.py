#!/usr/bin/env python3
"""Replay factorial attacks and perform node-local, independently controlled restoration."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from android_llm_gnn.mevgt import (  # noqa: E402
    AndroidMalwareNodeLatentAdapter,
    AndroidMalwareNodeLatentConfig,
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return resolved.name


def resolve_artifact_path(value: str, report_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root_candidate = ROOT / path
    if root_candidate.exists():
        return root_candidate
    return report_path.parent / path


def ensure_external_imports(cfg: dict[str, Any]) -> None:
    mevgt_root = cfg.get("task", {}).get("mevgt_root")
    if mevgt_root:
        resolved = str(Path(mevgt_root).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def canonical_circuit(circuit: Any) -> str:
    base = f"{circuit.module}:{circuit.layer}"
    return f"{base}:{circuit.tag}" if getattr(circuit, "tag", None) else base


def circuit_name_from_score(score: dict[str, Any]) -> str:
    circuit = score["circuit"]
    base = f"{circuit['module']}:{circuit['layer']}"
    return f"{base}:{circuit['tag']}" if circuit.get("tag") else base


def capture_clean_activations(
    adapter: AndroidMalwareNodeLatentAdapter, row: Any, circuits: List[Any]
) -> Dict[str, torch.Tensor]:
    captured: Dict[str, torch.Tensor] = {}
    handles = []
    try:
        for circuit in circuits:
            module = adapter._resolve_circuit_module(circuit)
            key = canonical_circuit(circuit)

            def hook(_module, _inputs, output, hook_key=key):
                if not torch.is_tensor(output):
                    raise TypeError(f"Circuit {hook_key} does not expose a tensor output")
                captured[hook_key] = output.detach().clone()

            handles.append(module.register_forward_hook(hook))
        adapter._forward_single(row, overrides={})
    finally:
        for handle in handles:
            handle.remove()
    missing = [canonical_circuit(circuit) for circuit in circuits if canonical_circuit(circuit) not in captured]
    if missing:
        raise ValueError(f"Failed to capture clean activations for: {missing}")
    return captured


def evaluate_with_localized_restoration(
    *,
    adapter: AndroidMalwareNodeLatentAdapter,
    row: Any,
    overrides: Dict[int, torch.Tensor],
    circuit: Any,
    clean_activation: torch.Tensor,
    node_ids: List[int],
):
    """Restore only selected node rows at one GNN stage.

    Replacing an entire layer creates a trivial computational cut and makes any
    layer recover the clean prediction. Row-local replacement preserves the
    adversarial state everywhere else and admits matched random-node controls.
    """
    if circuit.module != "gnn":
        raise ValueError("Localized restoration is defined only for node-aligned GNN circuits")
    module = adapter._resolve_circuit_module(circuit)

    def hook(_module, _inputs, output):
        if not torch.is_tensor(output):
            raise TypeError("GNN circuit output must be a tensor")
        if output.shape != clean_activation.shape:
            raise ValueError(
                f"Clean/adversarial activation shape mismatch: {clean_activation.shape} vs {output.shape}"
            )
        restored = output.clone()
        valid_ids = [index for index in node_ids if 0 <= index < restored.shape[0]]
        if len(valid_ids) != len(node_ids):
            raise IndexError("Localized restoration node index is outside the graph")
        if valid_ids:
            index = torch.tensor(valid_ids, dtype=torch.long, device=restored.device)
            restored[index] = clean_activation.to(restored.device, restored.dtype)[index]
        return restored

    handle = module.register_forward_hook(hook)
    try:
        logits, labels = adapter._forward_single(row, overrides=overrides)
    finally:
        handle.remove()

    probs = torch.softmax(logits, dim=1)
    pred = int(torch.argmax(probs, dim=1).item())
    malware_prob = float(probs[0, 1].item())
    benign_prob = float(probs[0, 0].item())
    from mevgt.types import MetricBundle  # noqa: E402

    return MetricBundle(
        primary=malware_prob,
        metrics={
            "benign_prob": benign_prob,
            "malware_prob": malware_prob,
            "predicted_label": float(pred),
            "true_label": float(labels.item()),
            "success": float(pred == int(adapter.cfg.target_label)),
            "margin": float((probs[0, 0] - probs[0, 1]).item()),
        },
    )


def recovery_ratio(clean_primary: float, adv_primary: float, restored_primary: float) -> float | None:
    lost_score = clean_primary - adv_primary
    if lost_score <= 1e-8:
        return None
    return float((restored_primary - adv_primary) / lost_score)


def mean_or_none(values: List[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attack-states", type=Path)
    parser.add_argument("--random-control-seed", type=int, default=0)
    parser.add_argument("--replay-tolerance", type=float, default=1e-5)
    parser.add_argument("--device", choices=("cpu", "mps"), help="Override task device.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ensure_external_imports(cfg)
    report = load_json(args.report)
    detail = report.get("detail", {})
    if detail.get("condition_set") != "factorial":
        raise ValueError("Localized restoration requires a factorial attack report")
    if detail.get("reference_sampling") != "shared_manifest":
        raise ValueError("Localized restoration requires a frozen shared-reference manifest")

    state_value = detail.get("attack_states")
    if args.attack_states:
        state_path = args.attack_states
    elif isinstance(state_value, str) and state_value:
        state_path = resolve_artifact_path(state_value, args.report)
    else:
        raise ValueError("Factorial report does not identify replayable attack states")
    expected_state_hash = detail.get("attack_states_sha256")
    actual_state_hash = sha256_file(state_path)
    if expected_state_hash != actual_state_hash:
        raise ValueError("Attack-state sidecar hash does not match the factorial report")
    state_payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if state_payload.get("schema_version") != 1:
        raise ValueError("Unsupported attack-state schema")
    attack_states = state_payload.get("states")
    if not isinstance(attack_states, dict):
        raise ValueError("Attack-state sidecar has no states mapping")

    task_cfg = dict(cfg.get("task", {}))
    task_cfg.pop("mevgt_root", None)
    if args.device:
        task_cfg["device"] = args.device
    adapter = AndroidMalwareNodeLatentAdapter(AndroidMalwareNodeLatentConfig.from_dict(task_cfg))
    adapter.setup()

    selected = detail.get("selected_references", [])
    per_sample_by_id = {}
    for sample in detail.get("per_sample", []):
        sample_id = sample.get("reference_summary", {}).get("sample_id")
        if sample_id:
            per_sample_by_id[sample_id] = sample

    sample_reports: List[Dict[str, Any]] = []
    hotspot_values: List[float] = []
    random_layer_values: List[float] = []
    random_node_values: List[float] = []
    successful_attack_count = 0
    hotspot_label_recovery_count = 0
    random_layer_label_recovery_count = 0
    random_node_label_recovery_count = 0
    non_positive_drop_count = 0
    missing_local_control_count = 0

    try:
        split = str(cfg.get("eval_split", "test"))
        for sample_meta in selected:
            sample_id = sample_meta.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("Selected reference lacks a stable sample_id")
            if sample_id not in attack_states:
                raise ValueError(f"Missing replay state for {sample_id}")
            factorial_sample = per_sample_by_id.get(sample_id)
            if not factorial_sample:
                raise ValueError(f"Missing factorial discovery record for {sample_id}")

            reference_index = int(sample_meta["reference_index"])
            adapter.set_reference_index(split, reference_index)
            row = adapter._reference_row(split)
            circuits = [circuit for circuit in adapter.list_circuits() if circuit.module == "gnn"]
            circuit_by_name = {canonical_circuit(circuit): circuit for circuit in circuits}
            if len(circuit_by_name) < 2:
                raise ValueError("At least two node-aligned GNN stages are required for layer controls")

            ranked_names = [
                circuit_name_from_score(score)
                for score in factorial_sample.get("top_scores", [])
                if score.get("circuit", {}).get("module") == "gnn"
            ]
            if not ranked_names or ranked_names[0] not in circuit_by_name:
                raise ValueError(f"No replayable preselected GNN hotspot for {sample_id}")
            hotspot_stage = ranked_names[0]
            hotspot_circuit = circuit_by_name[hotspot_stage]

            replay = attack_states[sample_id]
            overrides = {
                int(node_idx): tensor.detach().cpu().clone()
                for node_idx, tensor in replay.get("delta_overrides", {}).items()
            }
            target_node_ids = sorted(overrides)
            clean_metric = adapter.evaluate_with_node_override(split, overrides={})
            adv_metric = adapter.evaluate_with_node_override(split, overrides=overrides)
            clean_primary = float(clean_metric.primary)
            adv_primary = float(adv_metric.primary)
            if abs(adv_primary - float(replay["reference_adv_malware_prob"])) > args.replay_tolerance:
                raise ValueError(f"Adversarial replay mismatch for {sample_id}")

            clean_prediction = int(clean_metric.metrics.get("predicted_label", -1))
            attack_success = bool(float(adv_metric.metrics.get("success", 0.0)))
            if attack_success:
                successful_attack_count += 1
            if clean_primary - adv_primary <= 1e-8:
                non_positive_drop_count += 1

            clean_activations = capture_clean_activations(adapter, row, circuits)
            rng = random.Random(args.random_control_seed + reference_index)
            nonhotspot_stages = sorted(set(circuit_by_name) - {hotspot_stage})
            random_layer_stage = rng.choice(nonhotspot_stages)
            random_layer_circuit = circuit_by_name[random_layer_stage]
            available_random_nodes = sorted(set(range(int(row.x.shape[0]))) - set(target_node_ids))
            random_node_ids = (
                rng.sample(available_random_nodes, len(target_node_ids))
                if target_node_ids and len(available_random_nodes) >= len(target_node_ids)
                else []
            )
            if not target_node_ids or not random_node_ids:
                missing_local_control_count += 1

            stage_target_ratios: Dict[str, float | None] = {}
            stage_target_metrics: Dict[str, float] = {}
            for stage, circuit in circuit_by_name.items():
                metric = evaluate_with_localized_restoration(
                    adapter=adapter,
                    row=row,
                    overrides=overrides,
                    circuit=circuit,
                    clean_activation=clean_activations[stage],
                    node_ids=target_node_ids,
                )
                stage_target_metrics[stage] = float(metric.primary)
                stage_target_ratios[stage] = recovery_ratio(
                    clean_primary, adv_primary, float(metric.primary)
                )

            hotspot_random_node_metric = evaluate_with_localized_restoration(
                adapter=adapter,
                row=row,
                overrides=overrides,
                circuit=hotspot_circuit,
                clean_activation=clean_activations[hotspot_stage],
                node_ids=random_node_ids,
            )
            hotspot_ratio = stage_target_ratios[hotspot_stage]
            random_layer_ratio = stage_target_ratios[random_layer_stage]
            random_node_ratio = recovery_ratio(
                clean_primary, adv_primary, float(hotspot_random_node_metric.primary)
            )
            if hotspot_ratio is not None:
                hotspot_values.append(float(hotspot_ratio))
            if random_layer_ratio is not None:
                random_layer_values.append(float(random_layer_ratio))
            if random_node_ratio is not None:
                random_node_values.append(float(random_node_ratio))

            hotspot_label_recovered = bool(
                stage_target_metrics[hotspot_stage] >= 0.5
                and clean_prediction == 1
            )
            random_layer_label_recovered = bool(
                stage_target_metrics[random_layer_stage] >= 0.5
                and clean_prediction == 1
            )
            random_node_label_recovered = bool(
                float(hotspot_random_node_metric.primary) >= 0.5
                and clean_prediction == 1
            )
            if attack_success and hotspot_label_recovered:
                hotspot_label_recovery_count += 1
            if attack_success and random_layer_label_recovered:
                random_layer_label_recovery_count += 1
            if attack_success and random_node_label_recovered:
                random_node_label_recovery_count += 1

            eligible = hotspot_ratio is not None
            oracle_stage = None
            oracle_ratio = None
            eligible_stage_ratios = {
                stage: ratio for stage, ratio in stage_target_ratios.items() if ratio is not None
            }
            if eligible_stage_ratios:
                oracle_stage = max(eligible_stage_ratios, key=eligible_stage_ratios.get)
                oracle_ratio = eligible_stage_ratios[oracle_stage]

            sample_reports.append({
                "reference_index": reference_index,
                "sample_id": sample_id,
                "file_name": sample_meta.get("file_name"),
                "clean_primary": clean_primary,
                "adv_primary": adv_primary,
                "attack_success": attack_success,
                "restoration_eligible": eligible,
                "target_node_ids": target_node_ids,
                "matched_random_node_ids": random_node_ids,
                "preselected_hotspot_stage": hotspot_stage,
                "hotspot_target_node_recovery_ratio": hotspot_ratio,
                "hotspot_target_node_label_recovered": hotspot_label_recovered,
                "random_nonhotspot_stage": random_layer_stage,
                "random_layer_target_node_recovery_ratio": random_layer_ratio,
                "random_layer_target_node_label_recovered": random_layer_label_recovered,
                "hotspot_random_node_recovery_ratio": random_node_ratio,
                "hotspot_random_node_label_recovered": random_node_label_recovered,
                "oracle_best_stage": oracle_stage,
                "oracle_best_recovery_ratio": oracle_ratio,
                "trace": {
                    "mode": "independent_local_node_activation",
                    "target_node_recovery_by_stage": stage_target_ratios,
                    "target_node_metric_by_stage": stage_target_metrics,
                    "hotspot_random_node_metric": float(hotspot_random_node_metric.primary),
                },
            })
    finally:
        adapter.teardown()

    payload = {
        "schema_version": 3,
        "restoration_mode": "independent_local_node_activation",
        "config": artifact_path(args.config),
        "report": artifact_path(args.report),
        "attack_states": artifact_path(state_path),
        "attack_states_sha256": actual_state_hash,
        "shared_cohort_id": detail.get("shared_cohort_id"),
        "sample_count": len(sample_reports),
        "eligible_sample_count": sum(bool(sample["restoration_eligible"]) for sample in sample_reports),
        "non_positive_drop_count": non_positive_drop_count,
        "missing_local_control_count": missing_local_control_count,
        "successful_attack_count": successful_attack_count,
        "mean_hotspot_target_node_recovery_ratio": mean_or_none(hotspot_values),
        "mean_random_layer_target_node_recovery_ratio": mean_or_none(random_layer_values),
        "mean_hotspot_random_node_recovery_ratio": mean_or_none(random_node_values),
        "hotspot_label_recovery_count": hotspot_label_recovery_count,
        "random_layer_label_recovery_count": random_layer_label_recovery_count,
        "random_node_label_recovery_count": random_node_label_recovery_count,
        "samples": sample_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
