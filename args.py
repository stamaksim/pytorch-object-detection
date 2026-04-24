import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Object detection training options")

    parser.add_argument("--csv_dir", type=str, default="unified_dataset/CSVs")
    parser.add_argument("--train_csv", type=str, default="train_df.csv")
    parser.add_argument("--val_csv", type=str, default="val_df.csv")

    parser.add_argument("--outdir", type=str, default="output")

    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--num_classes", type=int, default=2, help="Background + object classes")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()