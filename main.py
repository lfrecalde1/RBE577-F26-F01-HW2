from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.utils import make_grid


def add_gaussian_noise(image, std=0.02):
    noise = torch.randn_like(image) * std
    noisy_image = image + noise
    return noisy_image.clamp(0, 1)


def create_transforms(weights: ResNet50_Weights):
    ## Here we are using the values directly from the weights so everything is compatible with the Resnet
    val_transform = weights.transforms()

    train_transform = transforms.Compose([
        transforms.Resize(
            val_transform.resize_size,
            interpolation=val_transform.interpolation,
            antialias=True,
        ),
        transforms.RandomCrop(val_transform.crop_size),
        transforms.RandomAffine(
            degrees=10,             
            translate=(0.05, 0.05), 
            scale=(0.95, 1.05),     
            shear=5,                
            interpolation=val_transform.interpolation,
            fill=128,       
        ),
        transforms.ToTensor(),
        transforms.RandomApply([add_gaussian_noise], p=0.5),
        transforms.Normalize(mean=val_transform.mean, std=val_transform.std),
    ])

    return train_transform, val_transform


def create_model(weights: ResNet50_Weights, num_classes: int, freeze_backbone=False):
    # We only replace last layer
    model = resnet50(weights=weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    num_features = model.fc.in_features
    # This creates the last layer with the number of classes we need
    model.fc = nn.Linear(num_features, num_classes)
    return model


def load_datasets(
    data_dir: Path, train_transform=None, val_transform=None
) -> tuple[ImageFolder, ImageFolder]:
    for split in ("train", "val"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset folder not found: {split_dir}. "
                "Run .venv/bin/python download_data.py from the project folder."
            )

    train_dataset = ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = ImageFolder(data_dir / "val", transform=val_transform)

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Training and validation must have identical class-to-label mappings.\n"
            f"Training: {train_dataset.class_to_idx}\n"
            f"Validation: {val_dataset.class_to_idx}"
        )
    return train_dataset, val_dataset


def train_one_epoch(
    model, train_loader, criterion, optimizer, device, freeze_backbone=False
):
    model.train()
    if freeze_backbone:
        model.eval()
        model.fc.train()
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    progress = tqdm(train_loader, desc="Training", unit="batch")
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_images += batch_size
        progress.set_postfix(
            loss=f"{total_loss / total_images:.4f}",
            accuracy=f"{total_correct / total_images:.1%}",
            refresh=False,
        )

    return total_loss / total_images, total_correct / total_images


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    with torch.no_grad():
        progress = tqdm(val_loader, desc="Validation", unit="batch")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_images += batch_size
            progress.set_postfix(
                loss=f"{total_loss / total_images:.4f}",
                accuracy=f"{total_correct / total_images:.1%}",
                refresh=False,
            )

    return total_loss / total_images, total_correct / total_images


def print_dataset_summary(
    train_dataset: ImageFolder, val_dataset: ImageFolder
) -> None:
    train_counts = Counter(train_dataset.targets)
    val_counts = Counter(val_dataset.targets)

    print(f"{'Label':<7} {'Class':<20} {'Train':>8} {'Val':>8}")
    for class_name, label in train_dataset.class_to_idx.items():
        print(
            f"{label:<7} {class_name:<20} "
            f"{train_counts[label]:>8} {val_counts[label]:>8}"
        )

    print(f"\nTotal: {len(train_dataset)} training, {len(val_dataset)} validation")


def imshow(image, mean, std, title=None):
    image = image.detach().cpu().numpy().transpose((1, 2, 0))
    image = image * np.array(std) + np.array(mean)
    image = np.clip(image, 0, 1)

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.imshow(image)
    axis.axis("off")
    if title is not None:
        axis.set_title(title)
    figure.tight_layout()
    return figure


def main() -> None:
    batch_size = 20
    num_epochs = 30
    learning_rate = 0.005
    freeze_backbone = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_dir = Path.cwd() / "data"
    results_dir = Path.cwd() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    weights = ResNet50_Weights.IMAGENET1K_V2
    train_transform, val_transform = create_transforms(weights)
    train_dataset, val_dataset = load_datasets(
        data_dir, train_transform, val_transform
    )
    print_dataset_summary(train_dataset, val_dataset)
    num_classes = len(train_dataset.classes)

    model = create_model(weights, num_classes, freeze_backbone)
    model = model.to(device)
    print("\nLoaded pretrained ResNet50.")
    print("Current classification head:", model.fc)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Close the preview window to continue to training.
    #for images, labels in train_loader:
    #    class_names = []
    #    for label in labels:
    #        class_name = train_dataset.classes[label.item()]
    #        class_names.append(class_name)

    #    grid = make_grid(images, nrow=10)
    #    imshow(
    #        grid,
    #        mean=val_transform.mean,
    #        std=val_transform.std,
    #        title=" | ".join(class_names),
    #    )
    #    plt.show()
    #    break  # Show only the first batch.
    
    ## This is the section for the training
    criterion = nn.CrossEntropyLoss()
    if freeze_backbone:
        parameters = model.fc.parameters()
    else:
        parameters = model.parameters()
    optimizer = torch.optim.SGD(
        parameters, lr=learning_rate, momentum=0.9, weight_decay=0.0001
    )

    # Each training run gets its own folder so curves do not mix across runs.
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_dir = results_dir / "tensorboard" / run_name
    print("TensorBoard logs:", log_dir)

    # The context manager closes the writer when training finishes or fails.
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        best_val_accuracy = -1.0
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}", flush=True)
            train_loss, train_accuracy = train_one_epoch(
                model, train_loader, criterion, optimizer, device, freeze_backbone
            )
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)

            # Two curves on one chart; the horizontal axis is the epoch number.
            writer.add_scalars(
                "Loss",
                {"Training": train_loss, "Validation": val_loss},
                global_step=epoch + 1,
            )
            writer.add_scalars(
                "Accuracy",
                {"Training": train_accuracy, "Validation": val_accuracy},
                global_step=epoch + 1,
            )
            writer.flush()  # Make this epoch's values available immediately.

            print(f"Train: loss={train_loss:.4f}, accuracy={train_accuracy:.1%}")
            print(f"Val:   loss={val_loss:.4f}, accuracy={val_accuracy:.1%}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_to_idx": train_dataset.class_to_idx,
                        "weights": weights.name,
                        "epoch": epoch + 1,
                        "val_accuracy": val_accuracy,
                    },
                    results_dir / "best_model.pth",
                )
                print("Saved best model to:", results_dir / "best_model.pth")


if __name__ == "__main__":
    main()
