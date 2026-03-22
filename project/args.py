import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Object detection training options")

    parser.add_argument("--csv_dir", type=str, default="data/CSVs")
    parser.add_argument("--outdir", type=str, default="output")

    parser.add_argument("--batch_size", type=int, default=8, choices=[8, 16, 32, 64])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)

    return parser.parse_args()