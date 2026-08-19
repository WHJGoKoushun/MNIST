import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class CompactCNN(nn.Module):
    def __init__(self, channels=(16, 32, 64)) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.channels = tuple(channels)
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(c3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(x), 1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_loaders(data_dir: Path, batch_size: int, seed: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08)),
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )

    # Both objects use the official training split. The official test split is
    # deliberately not constructed or downloaded by this training program.
    train_source = datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
    val_source = datasets.MNIST(data_dir, train=True, download=True, transform=eval_transform)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_source), generator=generator).tolist()
    val_size = 6000
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_loader = DataLoader(
        Subset(train_source, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        Subset(val_source, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader, train_indices, val_indices


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    labels_all = []
    probabilities_all = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            if not training:
                labels_all.append(labels.cpu().numpy())
                probabilities_all.append(torch.softmax(logits, dim=1).cpu().numpy())

    size = len(loader.dataset)
    auc = None
    if not training:
        auc = roc_auc_score(
            np.concatenate(labels_all),
            np.concatenate(probabilities_all),
            multi_class="ovr",
            average="macro",
        )
    return loss_sum / size, correct / size, auc


def save_plot(history, output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_auc"] for row in history], label="Macro OvR AUC")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation scores")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate CompactCNN on MNIST without touching the test split.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channels", type=int, nargs=3, default=(16, 32, 64))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_indices, val_indices = build_loaders(
        args.data_dir, args.batch_size, args.seed
    )

    channels = tuple(args.channels)
    model = CompactCNN(channels).to(device)
    params = parameter_count(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    history = []
    best_auc = -1.0
    epochs_without_improvement = 0
    best_path = args.output_dir / "best_model.pt"

    architecture = f"CompactCNN-{'-'.join(map(str, channels))}-GAP"
    print(f"Device: {device}")
    print(f"Architecture: {architecture}")
    print(f"Train/validation sizes: {len(train_loader.dataset)}/{len(val_loader.dataset)}")
    print(f"Trainable parameters: {params:,}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _ = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy, val_auc = run_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_auc)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_auc": val_auc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: train loss={train_loss:.4f}, "
            f"train acc={train_accuracy:.4f}, val loss={val_loss:.4f}, "
            f"val acc={val_accuracy:.4f}, val AUC={val_auc:.6f}"
        )

        if val_auc > best_auc + 1e-6:
            best_auc = val_auc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "parameter_count": params,
                    "best_val_auc": best_auc,
                    "epoch": epoch,
                    "seed": args.seed,
                    "architecture": architecture,
                    "channels": channels,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    np.savez_compressed(
        args.output_dir / "split_indices.npz",
        train_indices=np.asarray(train_indices),
        val_indices=np.asarray(val_indices),
    )
    best_checkpoint = torch.load(best_path, map_location="cpu")
    best_history_row = next(row for row in history if row["epoch"] == best_checkpoint["epoch"])
    summary = {
        "architecture": architecture,
        "channels": channels,
        "parameter_count": params,
        "best_validation_auc_macro_ovr": best_auc,
        "best_validation_accuracy": best_history_row["val_accuracy"],
        "best_validation_loss": best_history_row["val_loss"],
        "best_epoch": int(best_checkpoint["epoch"]),
        "test_split_used": False,
        "device": str(device),
        "seed": args.seed,
    }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_plot(history, args.output_dir / "training_curves.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
