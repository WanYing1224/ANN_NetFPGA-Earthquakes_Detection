import argparse
import json
import os
import time
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import xgboost as xgb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_tsunami_ann_balanced import (
    DEFAULT_NUMERIC_KEYS,
    DERIVED_FEATURE_KEYS,
    SimpleANN,
    SplitData,
    build_feature_matrix,
    compute_average_precision,
    compute_auroc,
    infer_region_codes,
    load_events,
    normalize_splits,
    select_best_accuracy_threshold,
    set_seed,
)


def stratified_split_indices(
    labels: np.ndarray,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for class_value in (0, 1):
        class_indices = np.where(labels == class_value)[0]
        rng.shuffle(class_indices)
        n_total = len(class_indices)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        n_train = min(max(n_train, 1), n_total - 2)
        n_val = min(max(n_val, 1), n_total - n_train - 1)
        train_parts.append(class_indices[:n_train])
        val_parts.append(class_indices[n_train : n_train + n_val])
        test_parts.append(class_indices[n_train + n_val :])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def make_split(features: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> SplitData:
    return SplitData(features[indices], labels[indices])


def make_loader(
    split: SplitData,
    batch_size: int,
    shuffle: bool,
    teacher_logits: np.ndarray,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(split.features, dtype=torch.float32),
        torch.tensor(split.labels, dtype=torch.float32),
        torch.tensor(teacher_logits, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def predict_probabilities(model: nn.Module, split: SplitData, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        features = torch.tensor(split.features, dtype=torch.float32, device=device)
        logits = model(features).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def evaluate_probabilities(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    accuracy = (tp + tn) / max(len(labels), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    loss = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    return {
        "threshold": threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(compute_auroc(labels, probabilities)),
        "average_precision": float(compute_average_precision(labels, probabilities)),
        "loss": float(loss),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_xgb_teacher(
    train_data: SplitData,
    val_data: SplitData,
    seed: int,
) -> xgb.Booster:
    dtrain = xgb.DMatrix(train_data.features, label=train_data.labels)
    dval = xgb.DMatrix(val_data.features, label=val_data.labels)
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "eta": 0.03,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "tree_method": "hist",
        "seed": seed,
    }
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=600,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )
    return booster


def xgb_predict_probabilities(booster: xgb.Booster, split: SplitData) -> np.ndarray:
    dmatrix = xgb.DMatrix(split.features)
    if booster.best_iteration is not None:
        return booster.predict(dmatrix, iteration_range=(0, booster.best_iteration + 1))
    return booster.predict(dmatrix)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    hard_loss = nn.functional.binary_cross_entropy_with_logits(student_logits, labels)
    teacher_probs = torch.sigmoid(teacher_logits / temperature)
    soft_loss = nn.functional.binary_cross_entropy_with_logits(
        student_logits / temperature,
        teacher_probs,
    ) * (temperature ** 2)
    return alpha * hard_loss + (1.0 - alpha) * soft_loss


def train_student_for_acc(
    train_data: SplitData,
    val_data: SplitData,
    test_data: SplitData,
    train_teacher_logits: np.ndarray,
    input_dim: int,
    hidden_dims: Sequence[int],
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    dropout: float,
    seed: int,
    alpha: float,
    temperature: float,
    device: torch.device,
) -> Dict:
    set_seed(seed)
    model = SimpleANN(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = make_loader(train_data, batch_size=batch_size, shuffle=True, teacher_logits=train_teacher_logits)

    best_state = None
    best_epoch = 0
    best_val_accuracy = -float("inf")
    best_val_loss = float("inf")
    best_threshold = 0.5
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for features, labels, teacher_logits in loader:
            features = features.to(device)
            labels = labels.to(device)
            teacher_logits = teacher_logits.to(device)
            optimizer.zero_grad(set_to_none=True)
            student_logits = model(features)
            loss = distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                alpha=alpha,
                temperature=temperature,
            )
            loss.backward()
            optimizer.step()

        val_probabilities = predict_probabilities(model, val_data, device)
        tuned = select_best_accuracy_threshold(val_data.labels, val_probabilities)
        val_metrics = evaluate_probabilities(val_data.labels, val_probabilities, threshold=tuned["threshold"])

        better = (
            val_metrics["accuracy"] > best_val_accuracy + 1e-9
            or (
                abs(val_metrics["accuracy"] - best_val_accuracy) <= 1e-9
                and val_metrics["loss"] < best_val_loss
            )
        )
        if better:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_threshold = tuned["threshold"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_probabilities = predict_probabilities(model, train_data, device)
    val_probabilities = predict_probabilities(model, val_data, device)
    test_probabilities = predict_probabilities(model, test_data, device)

    return {
        "best_epoch": best_epoch,
        "selected_threshold": best_threshold,
        "train_metrics": evaluate_probabilities(train_data.labels, train_probabilities, threshold=best_threshold),
        "val_metrics": evaluate_probabilities(val_data.labels, val_probabilities, threshold=best_threshold),
        "test_metrics": evaluate_probabilities(test_data.labels, test_probabilities, threshold=best_threshold),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACC-maximized distillation for tsunami prediction.")
    parser.add_argument("--data-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--student-hidden-dims", type=int, nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    events = load_events(args.data_json)
    region_codes = infer_region_codes(events)
    feature_keys = list(DEFAULT_NUMERIC_KEYS) + list(DERIVED_FEATURE_KEYS)
    features, labels, feature_names, _, _ = build_feature_matrix(
        events,
        region_codes,
        feature_keys=feature_keys,
        include_missing_indicators=True,
        include_region_one_hot=True,
    )

    train_idx, val_idx, test_idx = stratified_split_indices(labels, seed=args.seed)
    teacher_train = make_split(features, labels, train_idx)
    teacher_val = make_split(features, labels, val_idx)
    teacher_test = make_split(features, labels, test_idx)
    student_train, student_val, student_test, normalization = normalize_splits(
        teacher_train,
        teacher_val,
        teacher_test,
        region_codes=region_codes,
        include_region_one_hot=True,
    )

    teacher = train_xgb_teacher(teacher_train, teacher_val, seed=args.seed)
    teacher_train_probs = xgb_predict_probabilities(teacher, teacher_train)
    teacher_val_probs = xgb_predict_probabilities(teacher, teacher_val)
    teacher_test_probs = xgb_predict_probabilities(teacher, teacher_test)
    teacher_threshold = select_best_accuracy_threshold(teacher_val.labels, teacher_val_probs)["threshold"]
    teacher_metrics = {
        "train": evaluate_probabilities(teacher_train.labels, teacher_train_probs, teacher_threshold),
        "val": evaluate_probabilities(teacher_val.labels, teacher_val_probs, teacher_threshold),
        "test": evaluate_probabilities(teacher_test.labels, teacher_test_probs, teacher_threshold),
        "best_iteration": int(teacher.best_iteration if teacher.best_iteration is not None else -1),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.student_hidden_dims is not None:
        candidate_configs = [{
            "hidden_dims": list(args.student_hidden_dims),
            "alpha": 0.3 if args.alpha is None else args.alpha,
            "temperature": 2.0 if args.temperature is None else args.temperature,
            "dropout": 0.1 if args.dropout is None else args.dropout,
        }]
    else:
        candidate_configs = [
            {"hidden_dims": [256, 128, 64], "alpha": 0.3, "temperature": 2.0, "dropout": 0.1},
            {"hidden_dims": [256, 128, 64, 32], "alpha": 0.5, "temperature": 2.0, "dropout": 0.1},
            {"hidden_dims": [256, 128, 64, 32], "alpha": 0.7, "temperature": 2.0, "dropout": 0.1},
            {"hidden_dims": [512, 256, 128, 64], "alpha": 0.5, "temperature": 2.0, "dropout": 0.1},
            {"hidden_dims": [512, 256, 128, 64], "alpha": 0.5, "temperature": 3.0, "dropout": 0.1},
            {"hidden_dims": [512, 256, 128, 64, 32], "alpha": 0.3, "temperature": 2.0, "dropout": 0.0},
        ]

    train_teacher_logits = probabilities_to_logits(teacher_train_probs)
    search_results: List[Dict] = []
    best_result = None
    best_config = None
    for idx, config in enumerate(candidate_configs, start=1):
        result = train_student_for_acc(
            train_data=student_train,
            val_data=student_val,
            test_data=student_test,
            train_teacher_logits=train_teacher_logits,
            input_dim=student_train.features.shape[1],
            hidden_dims=config["hidden_dims"],
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            dropout=config["dropout"],
            seed=args.seed + idx,
            alpha=config["alpha"],
            temperature=config["temperature"],
            device=device,
        )
        search_entry = {
            "config": config,
            "best_epoch": result["best_epoch"],
            "selected_threshold": result["selected_threshold"],
            "val_accuracy": result["val_metrics"]["accuracy"],
            "test_accuracy": result["test_metrics"]["accuracy"],
            "test_auroc": result["test_metrics"]["auroc"],
            "test_f1": result["test_metrics"]["f1"],
        }
        search_results.append(search_entry)
        if best_result is None or result["val_metrics"]["accuracy"] > best_result["val_metrics"]["accuracy"] + 1e-9:
            best_result = result
            best_config = config

    if best_result is None or best_config is None:
        raise RuntimeError("No student configuration finished.")

    payload = {
        "config": {
            "data_json": args.data_json,
            "seed": args.seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "feature_keys": feature_keys,
            "feature_dim": int(features.shape[1]),
        },
        "teacher_metrics": teacher_metrics,
        "student_search_results": search_results,
        "best_student_config": best_config,
        "best_student_result": best_result,
        "normalization": normalization,
        "runtime_seconds": float(time.time() - start_time),
    }

    results_path = os.path.join(args.output_dir, "results.json")
    summary_path = os.path.join(args.output_dir, "summary.md")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    lines = [
        "# ACC-Maximized Distillation",
        "",
        f"- Feature dim: {payload['config']['feature_dim']}",
        f"- Teacher test ACC: {teacher_metrics['test']['accuracy']:.4f}",
        f"- Teacher test AUROC: {teacher_metrics['test']['auroc']:.4f}",
        f"- Best student hidden dims: {best_config['hidden_dims']}",
        f"- Best student alpha: {best_config['alpha']}",
        f"- Best student temperature: {best_config['temperature']}",
        f"- Best student threshold: {best_result['selected_threshold']:.4f}",
        f"- Best student val ACC: {best_result['val_metrics']['accuracy']:.4f}",
        f"- Best student test ACC: {best_result['test_metrics']['accuracy']:.4f}",
        f"- Best student test AUROC: {best_result['test_metrics']['auroc']:.4f}",
        f"- Best student test F1: {best_result['test_metrics']['f1']:.4f}",
    ]
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(json.dumps(
        {
            "results_json": results_path,
            "summary_md": summary_path,
            "teacher_test_acc": teacher_metrics["test"]["accuracy"],
            "student_test_acc": best_result["test_metrics"]["accuracy"],
            "student_test_auroc": best_result["test_metrics"]["auroc"],
            "runtime_seconds": payload["runtime_seconds"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
