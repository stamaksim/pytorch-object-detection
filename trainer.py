from args import get_args
import os
import torch
import torch.optim as optim
from tqdm import tqdm


def validate_model(model, val_loader, device):
    model.train()

    val_loss_sum = 0.0
    val_count = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    "boxes": target["boxes"].to(device=device, dtype=torch.float32),
                    "labels": target["labels"].to(device=device, dtype=torch.int64),
                }
                for target in targets
            ]

            loss_dict = model(images, targets)

            # 🔥 ФІКС
            if isinstance(loss_dict, dict):
                loss = sum(loss_value for loss_value in loss_dict.values())
                val_loss_sum += loss.item() * len(images)
                val_count += len(images)
            else:
                continue

    return val_loss_sum / val_count if val_count > 0 else 0

def train_model(model, train_loader, val_loader, device):
    args = get_args()

    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    best_val_loss = float("inf")

    os.makedirs(args.outdir, exist_ok=True)


    history = {
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for images, targets in pbar:
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    "boxes": target["boxes"].to(device=device, dtype=torch.float32),
                    "labels": target["labels"].to(device=device, dtype=torch.int64),
                }
                for target in targets
            ]

            loss_dict = model(images, targets)
            loss = sum(loss_value for loss_value in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_epoch_loss = running_loss / len(train_loader.dataset)

        val_loss = validate_model(model, val_loader, device)

        history['train_loss'].append(train_epoch_loss)
        history['val_loss'].append(val_loss)

        print(
            f"\nEpoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Best Val Loss: {best_val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history
            }

            torch.save(
                checkpoint,
                os.path.join(args.outdir, "best_model.pth")
            )

            print(f"✓ Saved best model with val_loss: {val_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.outdir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_epoch_loss,
                    'val_loss': val_loss,
                    'history': history
                },
                checkpoint_path
            )
            print(f"✓ Saved checkpoint: {checkpoint_path}")

    final_path = os.path.join(args.outdir, "final_model.pth")
    torch.save(
        {
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        },
        final_path
    )
    print(f"\n✓ Training completed! Final model saved to: {final_path}")
    print(f"✓ Best validation loss: {best_val_loss:.4f}")