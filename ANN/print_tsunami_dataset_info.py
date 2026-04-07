import argparse
import numpy as np

from tsunami_training_common import FEATURE_KEYS, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print NOAA/NCEI tsunami dataset summary.")
    parser.add_argument("--data-json", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    features, labels, feature_names, _ = load_dataset(args.data_json)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    missing_rates = []
    for idx, name in enumerate(feature_names):
        if name.endswith("_missing"):
            rate = float(features[:, idx].mean())
            missing_rates.append((name, rate))
    missing_rates = sorted(missing_rates, key=lambda x: x[1], reverse=True)

    print("NOAA/NCEI Tsunami Dataset Summary")
    print(f"  Total Events        : {len(labels)}")
    print(f"  Positive Events     : {positives}")
    print(f"  Negative Events     : {negatives}")
    print(f"  Positive Ratio      : {positives / len(labels):.4f}")
    print(f"  Feature Dimension   : {features.shape[1]}")
    print(f"  Base Feature Keys   : {FEATURE_KEYS}")
    print("  Top Missing Features:")
    for name, rate in missing_rates[:10]:
        print(f"    {name:<32} {rate:.4f}")
