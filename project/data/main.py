import pandas as pd
import os
from torch.utils.data import DataLoader

from args import get_args
from dataset import ObjDetectionDataset
from model import build_model
from trainer import train
from utils import collate_fn, ensure_dir, select_device, set_seed


def _resolve_existing_path(path_value: str, base_dir: str) -> str:
    candidates = []
    if os.path.isabs(path_value):
        candidates.append(path_value)
    else:
        candidates.append(os.path.join(os.getcwd(), path_value))
        candidates.append(os.path.join(base_dir, path_value))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


def _load_dataframes(args, base_dir: str):
    csv_dir_candidates = [
        _resolve_existing_path(args.csv_dir, base_dir),
        os.path.join(base_dir, "CSVs"),
        os.path.join(os.getcwd(), "CSVs"),
    ]
    csv_dir = next((path for path in csv_dir_candidates if os.path.isdir(path)), csv_dir_candidates[0])

    dataset_path = os.path.join(csv_dir, args.csv_file)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Could not find {args.csv_file} in {csv_dir}."
        )

    full_df = pd.read_csv(dataset_path).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if len(full_df) < 2 or args.val_split <= 0:
        return full_df, None

    split_ratio = min(max(args.val_split, 0.0), 0.9)
    split_idx = int(len(full_df) * (1 - split_ratio))
    split_idx = max(1, min(split_idx, len(full_df) - 1))
    train_df = full_df.iloc[:split_idx].reset_index(drop=True)
    val_df = full_df.iloc[split_idx:].reset_index(drop=True)
    print(f"Using {args.csv_file}: train={len(train_df)} | val={len(val_df)}")
    return train_df, val_df

def main():
    args = get_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    set_seed(args.seed)
    outdir = _resolve_existing_path(args.outdir, base_dir)
    ensure_dir(outdir)
    device = select_device(args.device)

    train_df, val_df = _load_dataframes(args, base_dir)

    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df) if val_df is not None else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    model = build_model(num_classes=args.num_classes)
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        outdir=outdir,
    )

if __name__ == "__main__":
    main()