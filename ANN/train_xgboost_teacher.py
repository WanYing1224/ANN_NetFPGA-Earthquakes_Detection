import argparse
import json
import os
import time

import xgboost as xgb

from tsunami_training_common import (
    evaluate_probabilities,
    load_dataset,
    make_split,
    save_json,
    stratified_split_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test tsunami XGBoost teacher.")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--data-json", type=str, required=True)
    parser.add_argument("--artifact-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=20260406)
    return parser.parse_args()


def xgb_predict_probabilities(booster: xgb.Booster, features):
    dmatrix = xgb.DMatrix(features)
    if booster.best_iteration is not None:
        return booster.predict(dmatrix, iteration_range=(0, booster.best_iteration + 1))
    return booster.predict(dmatrix)


def print_metrics_block(title: str, metrics: dict) -> None:
    print(title)
    print(f"  Threshold : {metrics['threshold']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  AUROC     : {metrics['auroc']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  AvgPrec   : {metrics['average_precision']:.4f}")
    print(f"  LogLoss   : {metrics['loss']:.4f}")
    print(f"  Confusion : TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")


def train(args: argparse.Namespace) -> None:
    start = time.time()
    features, labels, feature_names, _ = load_dataset(args.data_json)
    train_idx, val_idx, test_idx = stratified_split_indices(labels, seed=args.seed)
    train_split = make_split(features, labels, train_idx)
    val_split = make_split(features, labels, val_idx)
    test_split = make_split(features, labels, test_idx)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "eta": 0.03,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "seed": args.seed,
    }
    booster = xgb.train(
        params=params,
        dtrain=xgb.DMatrix(train_split.features, label=train_split.labels),
        num_boost_round=600,
        evals=[
            (xgb.DMatrix(train_split.features, label=train_split.labels), "train"),
            (xgb.DMatrix(val_split.features, label=val_split.labels), "val"),
        ],
        early_stopping_rounds=40,
        verbose_eval=False,
    )

    val_probs = xgb_predict_probabilities(booster, val_split.features)
    from run_tsunami_ann_balanced import select_best_accuracy_threshold

    threshold = select_best_accuracy_threshold(val_split.labels, val_probs)["threshold"]
    train_metrics = evaluate_probabilities(train_split.labels, xgb_predict_probabilities(booster, train_split.features), threshold)
    val_metrics = evaluate_probabilities(val_split.labels, val_probs, threshold)
    test_metrics = evaluate_probabilities(test_split.labels, xgb_predict_probabilities(booster, test_split.features), threshold)

    os.makedirs(args.artifact_dir, exist_ok=True)
    model_path = os.path.join(args.artifact_dir, "model.json")
    metadata_path = os.path.join(args.artifact_dir, "metadata.json")
    booster.save_model(model_path)
    save_json(metadata_path, {
        "model_type": "xgboost",
        "seed": args.seed,
        "feature_names": feature_names,
        "threshold": threshold,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_iteration": int(booster.best_iteration if booster.best_iteration is not None else -1),
        "runtime_seconds": time.time() - start,
    })
    print("XGBoost Teacher Training Complete")
    print(f"  Artifact Dir   : {args.artifact_dir}")
    print(f"  Model Path     : {model_path}")
    print(f"  Metadata Path  : {metadata_path}")
    print(f"  Feature Dim    : {len(feature_names)}")
    print(f"  Best Iteration : {int(booster.best_iteration if booster.best_iteration is not None else -1)}")
    print(f"  Runtime Sec    : {time.time() - start:.2f}")
    print_metrics_block("Train Metrics", train_metrics)
    print_metrics_block("Validation Metrics", val_metrics)
    print_metrics_block("Test Metrics", test_metrics)


def test(args: argparse.Namespace) -> None:
    features, labels, _, _ = load_dataset(args.data_json)
    metadata_path = os.path.join(args.artifact_dir, "metadata.json")
    model_path = os.path.join(args.artifact_dir, "model.json")

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    test_idx = metadata["test_idx"]
    test_split = make_split(features, labels, np.array(test_idx, dtype=np.int64))
    booster = xgb.Booster()
    booster.load_model(model_path)
    metrics = evaluate_probabilities(test_split.labels, xgb_predict_probabilities(booster, test_split.features), metadata["threshold"])
    print("XGBoost Teacher Test Complete")
    print(f"  Artifact Dir  : {args.artifact_dir}")
    print(f"  Model Path    : {model_path}")
    print(f"  Metadata Path : {metadata_path}")
    print_metrics_block("Test Metrics", metrics)


if __name__ == "__main__":
    import numpy as np
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        test(args)
