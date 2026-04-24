import matplotlib.pyplot as plt
import matplotlib.patches as patches


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def show_batch(images, targets):
    for i in range(len(images)):
        image = images[i].detach().cpu().permute(1, 2, 0).numpy()
        boxes = targets[i]["boxes"].detach().cpu().numpy()
        labels = targets[i]["labels"].detach().cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image)

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )
            ax.add_patch(rect)

            ax.text(
                x1,
                y1 - 5,
                f"class {label}",
                fontsize=10,
                bbox=dict(facecolor="yellow", alpha=0.5)
            )

        ax.set_title(f"Sample {i + 1} in batch")
        ax.axis("off")
        plt.show()