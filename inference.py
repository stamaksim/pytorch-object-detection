import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from model import build_model  # твоя модель

MODEL_PATH = "output/best_model.pth"
IMAGE_FOLDER = "test_images"
DEVICE = torch.device("cpu")


def load_model():
    model = build_model(
        backbone="fasterrcnn_resnet50_fpn",
        num_classes=2
    )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model

def predict(image_path, model):
    image = Image.open(image_path).convert("RGB")

    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(image).to(DEVICE)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    return image, prediction


def show(image, prediction, filename, threshold=0.5):
    fig, ax = plt.subplots(1, figsize=(10, 7))
    ax.imshow(image)

    boxes = prediction["boxes"].cpu()
    scores = prediction["scores"].cpu()

    found = 0

    for box, score in zip(boxes, scores):
        if score > threshold:
            x1, y1, x2, y2 = box.numpy()

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            ax.text(x1, y1, f"{score:.2f}", color='red')
            found += 1

    print(f"{filename}: Detected cups = {found}")

    plt.axis('off')

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{filename}")
    plt.close()


def main():
    model = load_model()

    for file in os.listdir(IMAGE_FOLDER):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(IMAGE_FOLDER, file)

            image, prediction = predict(path, model)
            show(image, prediction, file)


if __name__ == "__main__":
    main()
