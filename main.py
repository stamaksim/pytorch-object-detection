import pandas as pd
from torch.utils.data import DataLoader
import os
import torch

from args import get_args
from dataset import ObjDetectionDataset
from utils import collate_fn, show_batch
from model import build_model
from trainer import train_model
from augmentation import build_val_transforms, build_train_transforms


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train_df = pd.read_csv(os.path.join(args.csv_dir, args.train_csv))
    val_df = pd.read_csv(os.path.join(args.csv_dir, args.val_csv))

    print(f"Train dataset size: {len(train_df)}")
    print(f"Validation dataset size: {len(val_df)}")

    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if args.debug:
        batch_images, batch_targets = next(iter(train_loader))
        show_batch(batch_images, batch_targets)
        return

    model = build_model(
        backbone="fasterrcnn_resnet50_fpn",
        num_classes=args.num_classes
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    train_model(
        model,
        train_loader,
        val_loader,
        device
    )


if __name__ == "__main__":
    main()