import pandas as pd
from torch.utils.data import DataLoader
import os
import torch

from args import get_args
from dataset import ObjDetectionDataset
from utils import collate_fn
from model import build_model
from trainer import train_model


def main():

    args = get_args()

    train_df = pd.read_csv(os.path.join(args.csv_dir, "train_df.csv"))
    val_df = pd.read_csv(os.path.join(args.csv_dir, "val_df.csv"))

    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = build_model(
        backbone="fasterrcnn_resnet50_fpn",
        num_classes=args.num_classes
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(
        model,
        train_loader,
        val_loader,
        device
    )


if __name__ == "__main__":
    main()
