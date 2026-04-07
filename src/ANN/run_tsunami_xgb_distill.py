import argparse
import json
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import xgboost as xgb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_tsunami_ann_balanced import (
    DEFAULT_NUMERIC_KEYS,
    SimpleANN,
    SplitData,
    build_feature_matrix,
    evaluate_model,
    infer_region_codes,
    load_events,
    normalize_splits,
    recommended_sample_per_class,
    select_balanced_indices,
    set_seed,
)


STUDENT_FEATURE_KEYS = [
    "latitude",
    "longitude",
    "eqMagnitude",
    "eqDepth",
    "intensity",
    "abs_latitude",
    "abs_longitude",
    "magnitude_depth_product",
    "magnitude_depth_ratio",
]


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
    teacher_logits: np.ndarray | None = None,
) -> DataLoader:
    tensors = [
        torch.tensor(split.features, dtype=torch.float32),
        torch.tensor(split.labels, dtype=torch.float32),
    ]
    if teacher_logits is not None:
        tensors.append(torch.tensor(teacher_logits, dtype=torch.float32))
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_xgb_teacher(
    train_data: SplitData,
    val_data: SplitData,
    seed: int,
    eta: float,
    max_depth: int,
    subsample: float,
    colsample_bytree: float,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> xgb.Booster:
    dtrain = xgb.DMatrix(train_data.features, label=train_data.labels)
    dval = xgb.DMatrix(val_data.features, label=val_data.labels)
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "tree_method": "hist",
        "seed": seed,
    }
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    return booster


def xgb_predict_probabilities(booster: xgb.Booster, split: SplitData) -> np.ndarray:
    dmatrix = xgb.DMatrix(split.features)
    iteration_range = (0, booster.best_iteration + 1) if booster.best_iteration is not None else None
    probabilities = booster.predict(dmatrix, iteration_range=iteration_range)
    return np.asarray(probabilities, dtype=np.float32)


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def xgb_metrics(booster: xgb.Booster, split: SplitData) -> Dict[str, float]:
    probabilities = xgb_predict_probabilities(booster, split)
    metrics = evaluate_probabilities(split.labels, probabilities)
    return metrics


def evaluate_probabilities(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    from run_tsunami_ann_balanced import binary_classification_metrics

    metrics = binary_classification_metrics(labels, probabilities, threshold=threshold)
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    loss = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    metrics["loss"] = float(loss)
    return metrics


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


def train_student_distilled(
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
    best_score = -float("inf")
    best_loss = float("inf")
    best_epoch = 0
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
        val_metrics = evaluate_model(model, val_data, device, threshold=0.5)
        better = (
            val_metrics["auroc"] > best_score + 1e-6
            or (abs(val_metrics["auroc"] - best_score) <= 1e-6 and val_metrics["loss"] < best_loss)
        )
        if better:
            best_score = val_metrics["auroc"]
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_epoch": best_epoch,
        "train_metrics": evaluate_model(model, train_data, device, threshold=0.5),
        "val_metrics": evaluate_model(model, val_data, device, threshold=0.5),
        "test_metrics": evaluate_model(model, test_data, device, threshold=0.5),
    }


def summarize_runs(run_results: Sequence[Dict]) -> Dict[str, float]:
    keys = ["accuracy", "precision", "recall", "f1", "auroc", "average_precision"]
    summary: Dict[str, float] = {}
    for split_name in ("teacher_test_metrics", "student_test_metrics"):
        for key in keys:
            values = [run[split_name][key] for run in run_results]
            summary[f"{split_name}_{key}_mean"] = float(np.mean(values))
            summary[f"{split_name}_{key}_std"] = float(np.std(values))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost teacher to ReLU student distillation.")
    parser.add_argument("--data-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sample-per-class", type=int, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--student-hidden-dims", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--xgb-eta", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=5)
    parser.add_argument("--xgb-subsample", type=float, default=0.9)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.9)
    parser.add_argument("--xgb-num-boost-round", type=int, default=400)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()

    events = load_events(args.data_json)
    region_codes = infer_region_codes(events)
    teacher_features, teacher_labels, teacher_feature_names, kept_indices, _ = build_feature_matrix(
        events,
        region_codes,
        feature_keys=DEFAULT_NUMERIC_KEYS,
        include_missing_indicators=True,
        include_region_one_hot=True,
    )
    student_features, student_labels, student_feature_names, student_kept_indices, _ = build_feature_matrix(
        events,
        region_codes,
        feature_keys=STUDENT_FEATURE_KEYS,
        include_missing_indicators=True,
        include_region_one_hot=False,
    )
    if kept_indices != student_kept_indices:
        raise RuntimeError("Teacher and student event indices are misaligned.")
    if not np.array_equal(teacher_labels, student_labels):
        raise RuntimeError("Teacher and student labels are misaligned.")

    labels = teacher_labels
    sample_per_class = args.sample_per_class or recommended_sample_per_class(labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_results: List[Dict] = []

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        selected_indices = select_balanced_indices(labels, sample_per_class=sample_per_class, seed=run_seed)
        selected_labels = labels[selected_indices]
        train_idx, val_idx, test_idx = stratified_split_indices(selected_labels, seed=run_seed)

        teacher_train = make_split(teacher_features[selected_indices], selected_labels, train_idx)
        teacher_val = make_split(teacher_features[selected_indices], selected_labels, val_idx)
        teacher_test = make_split(teacher_features[selected_indices], selected_labels, test_idx)
        student_train, student_val, student_test, _ = normalize_splits(
            make_split(student_features[selected_indices], selected_labels, train_idx),
            make_split(student_features[selected_indices], selected_labels, val_idx),
            make_split(student_features[selected_indices], selected_labels, test_idx),
            region_codes=region_codes,
            include_region_one_hot=False,
        )

        teacher = train_xgb_teacher(
            train_data=teacher_train,
            val_data=teacher_val,
            seed=run_seed,
            eta=args.xgb_eta,
            max_depth=args.xgb_max_depth,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            num_boost_round=args.xgb_num_boost_round,
            early_stopping_rounds=args.xgb_early_stopping_rounds,
        )
        teacher_train_probs = xgb_predict_probabilities(teacher, teacher_train)
        teacher_test_metrics = xgb_metrics(teacher, teacher_test)

        student_result = train_student_distilled(
            train_data=student_train,
            val_data=student_val,
            test_data=student_test,
            train_teacher_logits=probabilities_to_logits(teacher_train_probs),
            input_dim=student_train.features.shape[1],
            hidden_dims=args.student_hidden_dims,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            dropout=args.dropout,
            seed=run_seed,
            alpha=args.alpha,
            temperature=args.temperature,
            device=device,
        )

        run_results.append(
            {
                "seed": run_seed,
                "teacher_best_iteration": int(teacher.best_iteration if teacher.best_iteration is not None else -1),
                "teacher_test_metrics": teacher_test_metrics,
                "student_best_epoch": student_result["best_epoch"],
                "student_test_metrics": student_result["test_metrics"],
                "student_val_metrics": student_result["val_metrics"],
            }
        )

    aggregate = summarize_runs(run_results)
    payload = {
        "config": {
            "data_json": args.data_json,
            "sample_per_class": sample_per_class,
            "runs": args.runs,
            "seed": args.seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "student_hidden_dims": args.student_hidden_dims,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "xgb_eta": args.xgb_eta,
            "xgb_max_depth": args.xgb_max_depth,
            "xgb_subsample": args.xgb_subsample,
            "xgb_colsample_bytree": args.xgb_colsample_bytree,
            "xgb_num_boost_round": args.xgb_num_boost_round,
            "xgb_early_stopping_rounds": args.xgb_early_stopping_rounds,
            "device": str(device),
        },
        "teacher_feature_dim": int(teacher_features.shape[1]),
        "teacher_feature_names": teacher_feature_names,
        "student_feature_dim": int(student_features.shape[1]),
        "student_feature_names": student_feature_names,
        "aggregate": aggregate,
        "runs": run_results,
        "runtime_seconds": float(time.time() - start_time),
    }

    results_path = os.path.join(args.output_dir, "results.json")
    summary_path = os.path.join(args.output_dir, "summary.md")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    lines = [
        "# Tsunami XGBoost Distillation",
        "",
        f"- Teacher feature dim: {payload['teacher_feature_dim']}",
        f"- Student feature dim: {payload['student_feature_dim']}",
        f"- Teacher test ACC: {aggregate['teacher_test_metrics_accuracy_mean']:.4f} +/- {aggregate['teacher_test_metrics_accuracy_std']:.4f}",
        f"- Teacher test AUROC: {aggregate['teacher_test_metrics_auroc_mean']:.4f} +/- {aggregate['teacher_test_metrics_auroc_std']:.4f}",
        f"- Student test ACC: {aggregate['student_test_metrics_accuracy_mean']:.4f} +/- {aggregate['student_test_metrics_accuracy_std']:.4f}",
        f"- Student test AUROC: {aggregate['student_test_metrics_auroc_mean']:.4f} +/- {aggregate['student_test_metrics_auroc_std']:.4f}",
        f"- Student test F1: {aggregate['student_test_metrics_f1_mean']:.4f} +/- {aggregate['student_test_metrics_f1_std']:.4f}",
    ]
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(json.dumps(
        {
            "results_json": results_path,
            "summary_md": summary_path,
            "teacher_test_acc_mean": aggregate["teacher_test_metrics_accuracy_mean"],
            "student_test_acc_mean": aggregate["student_test_metrics_accuracy_mean"],
            "student_test_auroc_mean": aggregate["student_test_metrics_auroc_mean"],
            "runtime_seconds": payload["runtime_seconds"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
