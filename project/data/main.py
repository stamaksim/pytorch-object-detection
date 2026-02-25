from args import get_args
from dataset import ObjDetectionDataset
import pandas as pd
import os
from torch.utils.data import DataLoader

def main():
    args = get_args()
    print(args)

    train_df = pd.read_csv(os.path.join(args.csv_dir, "train_df.csv"))
    val_df = pd.read_csv(os.path.join(args.csv_dir, "val_df.csv"))

    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

if __name__ == "__main__":
    main()