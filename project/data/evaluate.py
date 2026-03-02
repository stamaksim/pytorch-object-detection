import torch


def evaluate(model, data_loader, device, score_thresh=0.5):
    model.eval()
    model.to(device)

    total_images = 0
    total_kept_boxes = 0
    total_score = 0.0

    with torch.no_grad():
        for images, _ in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            for output in outputs:
                scores = output.get("scores", torch.tensor([], device=device))
                keep = scores >= score_thresh

                kept_scores = scores[keep]
                total_kept_boxes += int(keep.sum().item())
                if kept_scores.numel() > 0:
                    total_score += float(kept_scores.sum().item())
                total_images += 1

    avg_boxes = total_kept_boxes / max(total_images, 1)
    avg_score = total_score / max(total_kept_boxes, 1)

    metrics = {
        "images": total_images,
        "boxes_above_thresh": total_kept_boxes,
        "avg_boxes_per_image": avg_boxes,
        "avg_score": avg_score,
    }
    print("Evaluation:", metrics)
    return metrics
