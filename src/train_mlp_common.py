import argparse
import json
import os
import time
from typing import Sequence

import numpy as np
import torch

from tsunami_training_common import (
    SimpleANN,
    apply_saved_normalization,
    count_mlp_params,
    load_dataset,
    make_split,
    normalize_splits,
    predict_probabilities,
    save_json,
    stratified_split_indices,
    train_supervised_mlp,
)


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


def train_mlp_artifact(
    data_json: str,
    artifact_dir: str,
    hidden_dims: Sequence[int],
    seed: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    dropout: float,
) -> None:
    start = time.time()
    features, labels, feature_names, _ = load_dataset(data_json)
    train_idx, val_idx, test_idx = stratified_split_indices(labels, seed=seed)
    train_split = make_split(features, labels, train_idx)
    val_split = make_split(features, labels, val_idx)
    test_split = make_split(features, labels, test_idx)
    train_norm, val_norm, test_norm, normalization = normalize_splits(
        train_split,
        val_split,
        test_split,
        region_codes=[],
        include_region_one_hot=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = train_supervised_mlp(
        train_data=train_norm,
        val_data=val_norm,
        test_data=test_norm,
        hidden_dims=hidden_dims,
        seed=seed,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        dropout=dropout,
        device=device,
    )

    os.makedirs(artifact_dir, exist_ok=True)
    model_path = os.path.join(artifact_dir, "model.pt")
    metadata_path = os.path.join(artifact_dir, "metadata.json")
    torch.save(result["model"].state_dict(), model_path)
    metadata = {
        "model_type": "mlp",
        "seed": seed,
        "feature_dim": int(features.shape[1]),
        "feature_names": feature_names,
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "threshold": result["threshold"],
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "normalization": normalization,
        "train_metrics": result["train_metrics"],
        "val_metrics": result["val_metrics"],
        "test_metrics": result["test_metrics"],
        "best_epoch": result["best_epoch"],
        "param_count": count_mlp_params(int(features.shape[1]), hidden_dims),
        "runtime_seconds": time.time() - start,
    }
    save_json(metadata_path, metadata)

    print("MLP Training Complete")
    print(f"  Artifact Dir   : {artifact_dir}")
    print(f"  Model Path     : {model_path}")
    print(f"  Metadata Path  : {metadata_path}")
    print(f"  Feature Dim    : {features.shape[1]}")
    print(f"  Hidden Dims    : {list(hidden_dims)}")
    print(f"  Param Count    : {metadata['param_count']}")
    print(f"  Best Epoch     : {result['best_epoch']}")
    print(f"  Runtime Sec    : {metadata['runtime_seconds']:.2f}")
    print_metrics_block("Train Metrics", result["train_metrics"])
    print_metrics_block("Validation Metrics", result["val_metrics"])
    print_metrics_block("Test Metrics", result["test_metrics"])


def test_mlp_artifact(
    data_json: str,
    artifact_dir: str,
) -> None:
    features, labels, _, _ = load_dataset(data_json)
    metadata_path = os.path.join(artifact_dir, "metadata.json")
    model_path = os.path.join(artifact_dir, "model.pt")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    test_idx = np.asarray(metadata["test_idx"], dtype=np.int64)
    test_split = make_split(features, labels, test_idx)
    normalization = metadata["normalization"]
    numeric_dim = normalization["numeric_dim"]
    region_dim = features.shape[1] - numeric_dim
    test_norm = apply_saved_normalization(test_split, normalization["mean"], normalization["std"], region_dim)
    model = SimpleANN(
        input_dim=int(features.shape[1]),
        hidden_dims=metadata["hidden_dims"],
        dropout=metadata["dropout"],
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    probabilities = predict_probabilities(model, test_norm, device)
    from tsunami_training_common import evaluate_probabilities

    metrics = evaluate_probabilities(test_norm.labels, probabilities, metadata["threshold"])
    print("MLP Test Complete")
    print(f"  Artifact Dir  : {artifact_dir}")
    print(f"  Model Path    : {model_path}")
    print(f"  Metadata Path : {metadata_path}")
    print(f"  Hidden Dims   : {metadata['hidden_dims']}")
    print(f"  Param Count   : {metadata['param_count']}")
    print_metrics_block("Test Metrics", metrics)
