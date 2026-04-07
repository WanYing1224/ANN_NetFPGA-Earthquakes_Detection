import argparse

from train_mlp_common import test_mlp_artifact, train_mlp_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test larger MLP 46->256->128->64->32->1.")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--data-json", type=str, required=True)
    parser.add_argument("--artifact-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train_mlp_artifact(
            data_json=args.data_json,
            artifact_dir=args.artifact_dir,
            hidden_dims=[256, 128, 64, 32],
            seed=args.seed,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            dropout=args.dropout,
        )
    else:
        test_mlp_artifact(
            data_json=args.data_json,
            artifact_dir=args.artifact_dir,
        )
