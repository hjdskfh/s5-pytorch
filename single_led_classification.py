import copy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from s5 import SingleLEDSSM


SEQUENCE_LENGTH = 128 # nr of timesteps in the LED intensity sequence. Each step is one scalar value.
STATE_WIDTH = 32 # hidden state width of the SSM
OUTPUT_DIM = 32 # output dimension of the SSM, which is fed into the classifier
CLASS_NAMES = ("seesaw", "sinus", "block")


def make_dataset(
    samples: int,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one scalar LED-intensity sequence and one waveform label."""
    time = torch.linspace(0.0, 1.0, sequence_length)
    signals = []
    labels = []

    for index in range(samples):
        label = index % len(CLASS_NAMES)
        phase = (index // len(CLASS_NAMES)) * 0.13
        frequency = 2.0 + (index % 4) * 0.5
        position = (frequency * time + phase).remainder(1.0)

        if label == 0:
            signal = 2.0 * torch.where(position < 0.5, position, 1.0 - position)
        elif label == 1:
            signal = 0.5 + 0.5 * torch.sin(
                2.0 * torch.pi * (frequency * time + phase)
            )
        else:
            signal = (position > 0.5).to(torch.float32)

        signals.append(signal)
        labels.append(label)

    return torch.stack(signals), torch.tensor(labels, dtype=torch.long)


class LEDWaveformClassifier(nn.Module):
    def __init__(self, state_width: int = STATE_WIDTH, output_dim: int = OUTPUT_DIM):
        super().__init__()
        self.led_ssm = SingleLEDSSM(state_width=state_width, output_dim=output_dim) #state_width = hidden state width of 32 for the SSM
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 64), # takes the input vector and transforms it into 64 features
            nn.GELU(), # applies non lin activation function
            nn.Linear(64, len(CLASS_NAMES)), # hidden layer with 64 features to output layer with number of classes
        )

    def forward(self, led_intensity: torch.Tensor) -> torch.Tensor:
        # led_intensity has shape [batch, sequence]: one scalar LED value per step.
        ssm_output = self.led_ssm(led_intensity)
        return self.classifier(ssm_output[:, -1, :]) # [B, output_dim] - take the last output of the SSM for classification


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    loss_function = nn.CrossEntropyLoss()

    with torch.no_grad():
        for signals, labels in loader:
            logits = model(signals.to(device))
            labels = labels.to(device)
            total_loss += loss_function(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def plot_examples(
    model: nn.Module,
    signals: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    count: int = 9,
    output_path: str = "predictions/single_led_misclassified.png",
) -> None:
    model.eval()
    with torch.no_grad():
        predictions = model(signals.to(device)).argmax(dim=1).cpu()

    incorrect = predictions != labels
    signals = signals[incorrect][:count]
    labels = labels[incorrect][:count]
    predictions = predictions[incorrect][:count]
    if len(signals) == 0:
        print("No misclassified examples to plot")
        return

    time = torch.linspace(0.0, 1.0, signals.shape[1])
    figure, axes = plt.subplots(
        len(signals),
        1,
        figsize=(10, 2 * len(signals)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    for axis, signal, label, prediction in zip(
        axes, signals, labels, predictions
    ):
        axis.plot(time, signal)
        axis.set_ylim(-0.1, 1.1)
        is_correct = label.item() == prediction.item()
        axis.set_ylabel(
            f"true: {CLASS_NAMES[label.item()]}\n"
            f"pred: {CLASS_NAMES[prediction.item()]}",
            color="green" if is_correct else "red",
        )
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("sequence position")
    figure.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=150)
    plt.close(figure)
    print(f"Saved plot to {output_file}")


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for signals, batch_labels in loader:
            predictions.append(model(signals.to(device)).argmax(dim=1).cpu())
            labels.append(batch_labels)
    return torch.cat(predictions), torch.cat(labels)


def plot_training_history(
    history: dict[str, list[float]],
    output_path: str = "predictions/training_history.png",
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["validation_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history["validation_accuracy"], label="validation")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=150)
    plt.close(figure)
    print(f"Saved training history to {output_file}")


def plot_confusion_matrix(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    output_path: str = "predictions/confusion_matrix.png",
) -> None:
    matrix = torch.zeros(
        (len(CLASS_NAMES), len(CLASS_NAMES)),
        dtype=torch.int64,
    )
    for label, prediction in zip(labels, predictions):
        matrix[label, prediction] += 1

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    axis.set_xlabel("predicted class")
    axis.set_ylabel("true class")
    axis.set_title("Test confusion matrix")

    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            axis.text(
                column,
                row,
                str(matrix[row, column].item()),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )

    figure.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=150)
    plt.close(figure)
    print(f"Saved confusion matrix to {output_file}")


def export_parameters(
    model: LEDWaveformClassifier,
    output_path: str = "predictions/single_led_parameters.pt",
) -> None:
    parameters = model.led_ssm.export_parameters()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(parameters, output_file)
    shapes = {name: tuple(value.shape) for name, value in parameters.items()}
    print(f"Saved S5 parameters to {output_file}")
    print(f"parameter shapes: {shapes}")


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_signals, all_labels = make_dataset(900)
    full_dataset = TensorDataset(all_signals, all_labels)
    train_dataset, validation_dataset, test_dataset = random_split(
        full_dataset,
        [600, 150, 150],
        generator=torch.Generator().manual_seed(7),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=64,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
    )

    model = LEDWaveformClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_function = nn.CrossEntropyLoss()
    best_validation_accuracy = -1.0
    best_model_state = None
    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    for epoch in range(60):
        model.train()
        train_loss = 0.0
        train_samples = 0
        for signals, labels in train_loader:
            logits = model(signals.to(device))
            loss = loss_function(logits, labels.to(device))
            train_loss += loss.item() * labels.size(0)
            train_samples += labels.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            device,
        )
        history["train_loss"].append(train_loss / train_samples)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_model_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0:
            print(
                f"epoch {epoch + 1}: "
                f"validation loss = {validation_loss:.4f}, "
                f"validation accuracy = {validation_accuracy:.2%}"
            )

    if best_model_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_model_state)

    test_loss, test_accuracy = evaluate(model, test_loader, device)
    print(f"device: {device}")
    print(f"best validation accuracy: {best_validation_accuracy:.2%}")
    print(f"test loss: {test_loss:.4f}")
    print(f"test accuracy: {test_accuracy:.2%}")
    export_parameters(model)
    test_indices = test_dataset.indices
    test_signals = all_signals[test_indices]
    test_labels = all_labels[test_indices]
    test_predictions, test_labels = collect_predictions(model, test_loader, device)
    plot_training_history(history)
    plot_confusion_matrix(test_labels, test_predictions)
    plot_examples(model, test_signals, test_labels, device)


if __name__ == "__main__":
    main()
