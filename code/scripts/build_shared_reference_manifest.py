#!/usr/bin/env python3
"""Build one clean-correct malware cohort shared by all leakage-safe checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from android_llm_gnn.mevgt import build_android_mevgt_adapter  # noqa: E402


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


def candidate_map(
    config_path: Path, *, device: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    cfg = load_config(config_path)
    task_cfg = dict(cfg.get("task", {}))
    mevgt_root = task_cfg.pop("mevgt_root", None)
    if mevgt_root:
        resolved = str(Path(mevgt_root).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    task_cfg["device"] = device

    adapter = build_android_mevgt_adapter(task_cfg)
    adapter.setup()
    try:
        split = str(cfg.get("eval_split", "test"))
        summaries = adapter.reference_candidate_summaries(split)
        by_id: dict[str, dict[str, Any]] = {}
        for summary in summaries:
            sample_id = summary.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Missing stable sample_id under {config_path}")
            if sample_id in by_id:
                raise ValueError(f"Duplicate sample_id {sample_id} under {config_path}")
            by_id[sample_id] = summary
        checkpoint = Path(task_cfg["checkpoint_path"])
        metadata = {
            "config": artifact_path(config_path),
            "checkpoint": artifact_path(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "clean_correct_malware_count": len(by_id),
        }
        return by_id, metadata
    finally:
        adapter.teardown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=[
            ROOT / "configs" / "mevgt" / f"node_latent_leakage_safe_seed{seed}.yaml"
            for seed in (42, 43, 44)
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--subset-size", type=int, default=512)
    parser.add_argument("--sampling-seed", type=int, default=2027)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        default="cpu",
        help="Inference device used only to compute the shared clean-correct intersection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "leakage_safe_shared_three_seed" / "shared_reference_manifest.json",
    )
    args = parser.parse_args()

    if len(args.configs) != len(args.seeds):
        raise ValueError("--configs and --seeds must have the same length")
    if args.subset_size < 1:
        raise ValueError("--subset-size must be positive")

    pools: dict[int, dict[str, dict[str, Any]]] = {}
    checkpoint_metadata: dict[str, dict[str, Any]] = {}
    for seed, config_path in zip(args.seeds, args.configs):
        pool, metadata = candidate_map(config_path, device=args.device)
        pools[int(seed)] = pool
        checkpoint_metadata[str(seed)] = metadata

    shared_ids = set.intersection(*(set(pool) for pool in pools.values()))
    if len(shared_ids) < args.subset_size:
        raise ValueError(
            f"Only {len(shared_ids)} samples are clean-correct for every checkpoint; "
            f"cannot freeze requested subset of {args.subset_size}"
        )
    ordered_shared = sorted(shared_ids)
    selected_ids = random.Random(args.sampling_seed).sample(ordered_shared, args.subset_size)
    cohort_material = "\n".join(selected_ids).encode("utf-8")
    cohort_id = hashlib.sha256(cohort_material).hexdigest()

    rows = []
    for sample_id in selected_ids:
        rows.append(
            {
                "sample_id": sample_id,
                "per_seed_reference_index": {
                    str(seed): int(pools[seed][sample_id]["reference_index"])
                    for seed in args.seeds
                },
                "file_name": next(
                    (
                        pools[seed][sample_id].get("file_name")
                        for seed in args.seeds
                        if pools[seed][sample_id].get("file_name")
                    ),
                    None,
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "cohort_id": cohort_id,
        "selection": "intersection_clean_correct_malware_then_deterministic_sample",
        "sampling_seed": int(args.sampling_seed),
        "shared_candidate_count": len(shared_ids),
        "selected_count": len(selected_ids),
        "sample_ids": selected_ids,
        "rows": rows,
        "checkpoints": checkpoint_metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": artifact_path(args.output),
        "cohort_id": cohort_id,
        "shared_candidate_count": len(shared_ids),
        "selected_count": len(selected_ids),
    }, indent=2))


if __name__ == "__main__":
    main()
