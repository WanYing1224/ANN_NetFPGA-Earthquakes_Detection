import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
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


FEATURE_KEYS = list(DEFAULT_NUMERIC_KEYS) + list(DERIVED_FEATURE_KEYS)


def load_dataset(data_json: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    events = load_events(data_json)
    region_codes = infer_region_codes(events)
    features, labels, feature_names, kept_indices, _ = build_feature_matrix(
        events,
        region_codes,
        feature_keys=FEATURE_KEYS,
        include_missing_indicators=True,
        include_region_one_hot=True,
    )
    return features, labels, feature_names, kept_indices


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


def apply_saved_normalization(
    split: SplitData,
    mean: Sequence[float],
    std: Sequence[float],
    region_dim: int,
) -> SplitData:
    features = split.features.copy()
    numeric_dim = features.shape[1] - region_dim
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    features[:, :numeric_dim] = (features[:, :numeric_dim] - mean_arr) / std_arr
    return SplitData(features, split.labels)


def make_loader(split: SplitData, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(split.features, dtype=torch.float32),
        torch.tensor(split.labels, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


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
        "threshold": float(threshold),
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


def train_supervised_mlp(
    train_data: SplitData,
    val_data: SplitData,
    test_data: SplitData,
    hidden_dims: Sequence[int],
    seed: int,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    patience: int,
    dropout: float,
    device: torch.device,
) -> Dict:
    set_seed(seed)
    model = SimpleANN(input_dim=train_data.features.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    loader = make_loader(train_data, batch_size=batch_size, shuffle=True)

    best_state = None
    best_epoch = 0
    best_val_accuracy = -float("inf")
    best_val_loss = float("inf")
    best_threshold = 0.5
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_probabilities = predict_probabilities(model, val_data, device)
        tuned = select_best_accuracy_threshold(val_data.labels, val_probabilities)
        val_metrics = evaluate_probabilities(val_data.labels, val_probabilities, tuned["threshold"])
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
        "model": model,
        "best_epoch": best_epoch,
        "threshold": float(best_threshold),
        "train_metrics": evaluate_probabilities(train_data.labels, train_probabilities, best_threshold),
        "val_metrics": evaluate_probabilities(val_data.labels, val_probabilities, best_threshold),
        "test_metrics": evaluate_probabilities(test_data.labels, test_probabilities, best_threshold),
    }


def count_mlp_params(input_dim: int, hidden_dims: Sequence[int]) -> int:
    total = 0
    prev = input_dim
    for h in hidden_dims:
        total += prev * h + h
        prev = h
    total += prev * 1 + 1
    return total


def save_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
