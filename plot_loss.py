import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import scienceplots
from matplotlib.ticker import MaxNLocator, PercentFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_metric(log_dir: Path, metric: str):
    """Read a metric's recorded epoch numbers and values from TensorBoard."""
    events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    events.Reload()
    if metric not in events.Tags()["scalars"]:
        raise ValueError(f"No {metric} values found in {log_dir}")

    # Keep the latest value if an epoch was logged more than once.
    values_by_epoch = {}
    for event in events.Scalars(metric):
        values_by_epoch[event.step] = event.value

    epochs = sorted(values_by_epoch)
    values = []
    for epoch in epochs:
        values.append(values_by_epoch[epoch])
    return epochs, values


def plot_metric(run_dir: Path, metric: str):
    train_epochs, train_values = read_metric(run_dir / f"{metric}_Training", metric)
    val_epochs, val_values = read_metric(run_dir / f"{metric}_Validation", metric)

    # no-latex keeps this script usable without a separate LaTeX installation.
    with plt.style.context(["science", "no-latex", "bright"]):
        plt.rcParams["pdf.fonttype"] = 42
        figure, axis = plt.subplots(figsize=(6, 3.8))
        axis.plot(train_epochs, train_values, "o-", label="Training", linewidth=1.7)
        axis.plot(val_epochs, val_values, "s--", label="Validation", linewidth=1.7)
        axis.set_xlabel("Epoch")
        if metric == "Accuracy":
            axis.set_ylabel("Accuracy")
            axis.set_ylim(0, 1)
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        else:
            axis.set_ylabel("Cross-entropy loss")
            axis.set_ylim(bottom=0)
        axis.set_title("ResNet50 vehicle classification")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)
        axis.legend(frameon=False)
        figure.tight_layout()

        filename = f"{metric.lower()}_curves"
        pdf_path = run_dir / f"{filename}.pdf"
        figure.savefig(pdf_path, bbox_inches="tight")
        figure.savefig(run_dir / f"{filename}.png", dpi=600, bbox_inches="tight")
        plt.close(figure)

    print(f"Training epochs: {train_epochs}")
    print(f"Validation epochs: {val_epochs}")
    print(f"Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Defaults to the latest TensorBoard run.")
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        runs = sorted(Path("results/tensorboard").glob("*/Loss_Training"))
        if not runs:
            raise FileNotFoundError("No training logs found. Run main.py first.")
        run_dir = runs[-1].parent

    plot_metric(run_dir, "Loss")
    if (run_dir / "Accuracy_Training").is_dir() and (run_dir / "Accuracy_Validation").is_dir():
        plot_metric(run_dir, "Accuracy")
    else:
        print("Accuracy was not logged for this run. Run the updated main.py to record it.")


if __name__ == "__main__":
    main()
