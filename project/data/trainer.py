import math

import torch

from utils import save_checkpoint


def _mean_loss(loss_dict):
    if not loss_dict:
        return torch.tensor(0.0)
    return sum(loss for loss in loss_dict.values())


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        loss_dict = model(images, targets)
        loss = _mean_loss(loss_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


def validate_one_epoch(model, loader, device):
    model.train()
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            loss_dict = model(images, targets)
            loss = _mean_loss(loss_dict)
            running_loss += loss.item()

    return running_loss / max(len(loader), 1)


def train(model, train_loader, val_loader, device, epochs, lr, weight_decay, outdir):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = math.inf

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = None
        if val_loader is not None and len(val_loader) > 0:
            val_loss = validate_one_epoch(model, val_loader, device)

        if val_loss is None:
            print(f"Epoch {epoch:03d}/{epochs:03d} | train_loss={train_loss:.4f}")
            save_checkpoint(model, optimizer, epoch, outdir)
            continue

        print(f"Epoch {epoch:03d}/{epochs:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, outdir)
            print(f"Saved best checkpoint with val_loss={val_loss:.4f}")

    return model
