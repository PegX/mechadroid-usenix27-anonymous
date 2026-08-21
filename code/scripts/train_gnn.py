#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.loader import DataLoader

HERE = Path(__file__).resolve().parent
SRC_ROOT = HERE.parent / "src"
REPO_ROOT = HERE.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from android_llm_gnn.datasets.pyg_android_dataset import PyGAndroidDataset  # noqa: E402
from android_llm_gnn.models.gnn_classifier import GraphClassifier  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_rows(dataset_path: Path) -> list:
    dataset = PyGAndroidDataset(dataset_path)
    return [dataset[i] for i in range(len(dataset))]


def filter_valid_rows(rows: list, split_name: str) -> list:
    valid_rows = []
    expected_dim = 0
    for row in rows:
        if row.x.ndim != 2 or row.x.shape[0] <= 0 or row.x.shape[1] <= 0:
            continue
        if expected_dim == 0:
            expected_dim = int(row.x.shape[1])
        if int(row.x.shape[1]) != expected_dim:
            continue
        valid_rows.append(row)

    rejected = len(rows) - len(valid_rows)
    if rejected:
        print(
            f"Filtered {rejected} invalid {split_name} graphs; "
            f"using {len(valid_rows)} with feature_dim={expected_dim}"
        )
    return valid_rows


def audit_group_disjoint_rows(train_rows: list, test_rows: list) -> dict[str, object]:
    missing_train = sum(not hasattr(row, "leakage_group_id") for row in train_rows)
    missing_test = sum(not hasattr(row, "leakage_group_id") for row in test_rows)
    if missing_train or missing_test:
        raise ValueError(
            "Leakage-safe training requires leakage_group_id on every graph; "
            f"missing train={missing_train}, test={missing_test}."
        )
    train_ids = [str(row.leakage_group_id) for row in train_rows]
    test_ids = [str(row.leakage_group_id) for row in test_rows]
    duplicate_train = len(train_ids) - len(set(train_ids))
    duplicate_test = len(test_ids) - len(set(test_ids))
    overlap = set(train_ids) & set(test_ids)
    if duplicate_train or duplicate_test or overlap:
        raise ValueError(
            "Dataset group audit failed: "
            f"duplicate_train={duplicate_train}, duplicate_test={duplicate_test}, "
            f"cross_split_overlap={len(overlap)}."
        )
    return {
        "train_group_count": len(train_ids),
        "test_group_count": len(test_ids),
        "duplicate_train_group_count": duplicate_train,
        "duplicate_test_group_count": duplicate_test,
        "cross_split_group_count": len(overlap),
    }


def stratified_train_val_split(rows: list, val_ratio: float, seed: int) -> tuple[list, list]:
    if not rows:
        return [], []
    if val_ratio <= 0 or len(rows) < 4:
        return rows, []

    by_label: dict[int, list] = {}
    for row in rows:
        label = int(row.y.item())
        by_label.setdefault(label, []).append(row)

    rng = random.Random(seed)
    train_rows: list = []
    val_rows: list = []

    for label_rows in by_label.values():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        if len(shuffled) < 2:
            train_rows.extend(shuffled)
            continue
        split_index = int(len(shuffled) * (1 - val_ratio))
        split_index = max(1, min(len(shuffled) - 1, split_index))
        train_rows.extend(shuffled[:split_index])
        val_rows.extend(shuffled[split_index:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def compute_class_weights(rows: list, device: torch.device) -> torch.Tensor:
    labels = [int(row.y.item()) for row in rows]
    counts = Counter(labels)
    total = sum(counts.values())
    weights = []
    for label in range(2):
        count = counts.get(label, 1)
        weights.append(total / (2.0 * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def move_batch(batch, device: torch.device):
    if hasattr(batch, "node_metadata"):
        del batch.node_metadata
    batch = batch.to(device)
    batch.x = batch.x.float()
    return batch


def fmt_metric(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.4f}"


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def artifact_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return resolved.name


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def release_accelerator_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state().cpu()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "mps" in state and torch.backends.mps.is_available() and hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(state["mps"])


def evaluate(model: GraphClassifier, loader: DataLoader, device: torch.device) -> dict[str, object]:
    if loader is None:
        return {
            "loss": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "confusion_matrix": None,
            "size": 0,
        }

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    losses: list[float] = []
    criterion = torch.nn.CrossEntropyLoss()

    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.view(-1))
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            losses.append(float(loss.item()))
            y_true.extend(batch.y.view(-1).tolist())
            y_pred.extend(preds.tolist())
            y_score.extend(probs.tolist())
            del batch, logits, loss, probs, preds

    release_accelerator_cache(device)

    metrics = {
        "loss": float(np.mean(losses)) if losses else None,
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if y_true else None,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if y_true else None,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if y_true else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist() if y_true else None,
        "size": len(y_true),
    }

    unique_labels = set(y_true)
    if len(unique_labels) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["roc_auc"] = None
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dataset",
        type=Path,
        default=Path(
            "pyg_datasets/train_dataset.pt"
        ),
    )
    parser.add_argument(
        "--test-dataset",
        type=Path,
        default=Path(
            "pyg_datasets/test_dataset.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "training_runs/default"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout-ratio", type=float, default=0.2)
    parser.add_argument("--gnn-type", choices=("gcn", "gat", "sage", "gin"), default="gin")
    parser.add_argument("--jk", choices=("last", "sum", "concat"), default="last")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--train-metrics-every",
        type=int,
        default=1,
        help="Evaluate the full training split every N epochs; 0 disables it.",
    )
    parser.add_argument(
        "--require-group-disjoint",
        action="store_true",
        help="Fail unless every graph has a unique leakage_group_id and train/test are disjoint.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/last_state.pt and save a recoverable state after every epoch.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = json_ready(vars(args))
    for path_key in ("train_dataset", "test_dataset", "output_dir"):
        config_dict[path_key] = artifact_path(Path(config_dict[path_key]))

    train_rows = filter_valid_rows(load_rows(args.train_dataset), "train")
    test_rows = (
        filter_valid_rows(load_rows(args.test_dataset), "test")
        if args.test_dataset.exists()
        else []
    )
    group_audit = None
    if args.require_group_disjoint:
        if not test_rows:
            raise ValueError("--require-group-disjoint requires a non-empty test dataset.")
        group_audit = audit_group_disjoint_rows(train_rows, test_rows)
        print(f"group_audit={json.dumps(group_audit, sort_keys=True)}")
    train_rows, val_rows = stratified_train_val_split(train_rows, args.val_ratio, args.seed)
    if not val_rows:
        val_rows = list(test_rows) if test_rows else list(train_rows)

    device = get_device(args.device)
    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_rows, batch_size=args.batch_size, shuffle=False) if val_rows else None
    test_loader = DataLoader(test_rows, batch_size=args.batch_size, shuffle=False) if test_rows else None

    if not train_rows:
        raise SystemExit("Training dataset is empty.")

    in_channels = int(train_rows[0].x.shape[1])
    model = GraphClassifier(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=2,
        num_layers=args.num_layers,
        dropout_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type,
        jk=args.jk,
    ).to(device)

    class_weights = compute_class_weights(train_rows, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_f1 = -1.0
    best_epoch = -1
    no_improvement = 0
    history: list[dict[str, object]] = []
    start_epoch = 1
    last_state_path = args.output_dir / "last_state.pt"

    if args.resume and last_state_path.exists():
        resume_state = torch.load(last_state_path, map_location="cpu", weights_only=False)
        if int(resume_state.get("seed", -1)) != args.seed:
            raise ValueError(f"Resume checkpoint seed mismatch in {last_state_path}")
        if int(resume_state.get("input_dim", -1)) != in_channels:
            raise ValueError(f"Resume checkpoint input dimension mismatch in {last_state_path}")
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        optimizer_to_device(optimizer, device)
        best_state = resume_state.get("best_state")
        best_val_f1 = float(resume_state.get("best_val_f1", -1.0))
        best_epoch = int(resume_state.get("best_epoch", -1))
        no_improvement = int(resume_state.get("no_improvement", 0))
        history = list(resume_state.get("history", []))
        start_epoch = int(resume_state["epoch"]) + 1
        restore_rng_state(resume_state["rng_state"])
        print(f"Resuming from {last_state_path} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            del batch, logits, loss

        compute_train_metrics = args.train_metrics_every > 0 and (
            epoch == 1 or epoch % args.train_metrics_every == 0
        )
        train_metrics = evaluate(model, train_eval_loader, device) if compute_train_metrics else {
            "loss": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "confusion_matrix": None,
            "size": len(train_rows),
        }
        val_metrics = evaluate(model, val_loader, device)
        release_accelerator_cache(device)
        epoch_record = {
            "epoch": epoch,
            "train_loss_epoch_mean": float(np.mean(epoch_losses)) if epoch_losses else None,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        history.append(epoch_record)

        val_f1 = val_metrics["f1"] if val_metrics["f1"] is not None else -1.0
        print(
            f"epoch={epoch} "
            f"train_loss={epoch_record['train_loss_epoch_mean']:.4f} "
            f"train_f1={fmt_metric(train_metrics['f1'])} "
            f"val_f1={fmt_metric(val_metrics['f1'])}"
        )

        should_stop = False
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                should_stop = True

        if args.resume:
            atomic_torch_save(
                {
                    "schema_version": 1,
                    "seed": args.seed,
                    "input_dim": in_channels,
                    "epoch": epoch,
                    "model_state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "optimizer_state_dict": cpu_tree(optimizer.state_dict()),
                    "best_state": best_state,
                    "best_val_f1": best_val_f1,
                    "best_epoch": best_epoch,
                    "no_improvement": no_improvement,
                    "history": history,
                    "rng_state": capture_rng_state(),
                },
                last_state_path,
            )
            print(f"Saved resumable state to {last_state_path}")
        if should_stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device) if test_loader else None

    checkpoint_path = args.output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_dict,
            "input_dim": in_channels,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
        },
        checkpoint_path,
    )

    results = {
        "config": config_dict,
        "device": str(device),
        "input_dim": in_channels,
        "class_weights": class_weights.detach().cpu().tolist(),
        "dataset_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "group_audit": group_audit,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint_path": artifact_path(checkpoint_path),
    }
    results_path = args.output_dir / "metrics.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {results_path}")


if __name__ == "__main__":
    main()
