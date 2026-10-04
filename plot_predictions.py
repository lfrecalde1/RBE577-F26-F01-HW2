"""Show one reproducibly selected validation image from each vehicle class."""

import hashlib
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights, resnet50


def select_images(dataset, seed):
    """Choose one image per class before inspecting the model's predictions."""
    generator = random.Random(seed)
    selected_indices = []
    for label in range(len(dataset.classes)):
        candidates = []
        for index, target in enumerate(dataset.targets):
            if target == label:
                candidates.append(index)
        selected_indices.append(generator.choice(candidates))
    return selected_indices


def main():
    seed = 42
    checkpoint_path = Path("results/best_model.pth")
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # A small CPU inference batch avoids needing a GPU just to create a figure.
    torch.set_num_threads(2)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weights = ResNet50_Weights[checkpoint["weights"]]
    preprocessing = weights.transforms()
    dataset = ImageFolder(Path("data/val"), transform=preprocessing)
    if dataset.class_to_idx != checkpoint["class_to_idx"]:
        raise ValueError("Validation labels do not match the saved model's labels.")
    if len(dataset.classes) != 10:
        raise ValueError("This 2 by 5 figure expects ten vehicle classes.")

    selected_indices = select_images(dataset, seed)
    images = []
    for index in selected_indices:
        image, label = dataset[index]
        images.append(image)
    inputs = torch.stack(images)

    # All learned parameters come from the saved checkpoint; no download is needed.
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1)

    mean = torch.tensor(preprocessing.mean).view(3, 1, 1)
    std = torch.tensor(preprocessing.std).view(3, 1, 1)
    figure, axes = plt.subplots(2, 5, figsize=(13, 7.5))
    records = []
    for panel, index in enumerate(selected_indices):
        axis = axes.flat[panel]
        true_label = dataset.targets[index]
        predicted_label = predictions[panel].item()
        correct = predicted_label == true_label
        color = "#267343" if correct else "#B3261E"

        # Display the exact model input crop, with normalization reversed.
        image = (inputs[panel] * std + mean).clamp(0, 1)
        axis.imshow(image.permute(1, 2, 0).numpy())
        axis.set_xticks([])
        axis.set_yticks([])
        for border in axis.spines.values():
            border.set_color(color)
            border.set_linewidth(2.5)
        axis.set_title(
            f"True: {dataset.classes[true_label]}\n"
            f"Predicted: {dataset.classes[predicted_label]}",
            fontsize=10, loc="left", pad=8,
        )
        axis.set_xlabel("✓ Correct" if correct else "✗ Incorrect", color=color, fontsize=10)
        records.append({
            "image": dataset.samples[index][0],
            "true_class": dataset.classes[true_label],
            "predicted_class": dataset.classes[predicted_label],
            "correct": correct,
        })

    figure.suptitle(f"Vehicle predictions — checkpoint at epoch {checkpoint['epoch']}", fontsize=16)
    # Leave room between rows for both prediction labels and correctness labels.
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.07, top=0.85,
                           wspace=0.12, hspace=0.55)
    plt.rcParams["pdf.fonttype"] = 42
    figure.savefig(output_dir / "vehicle_predictions.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "vehicle_predictions.png", dpi=600, bbox_inches="tight")
    plt.close(figure)

    # Preserve the exact image selection and model identity behind the figure.
    metadata = {
        "selection_seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_epoch": checkpoint["epoch"],
        "validation_accuracy": checkpoint["val_accuracy"],
        "images": records,
    }
    (output_dir / "vehicle_predictions.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print("Saved prediction PDF and PNG to:", output_dir)


if __name__ == "__main__":
    main()
