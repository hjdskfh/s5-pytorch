from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from s5 import S5Block


class MNISTS5(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = S5Block(
            dim=28,
            state_dim=128,
            block_count=4,
            bidir=False,
        )
        self.classifier = nn.Linear(28, 10)

    def forward(self, images):
        # Treat each image row as one sequence step with 28 features.
        x = images.squeeze(1)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1) # get the index of the max log-probability
            correct += (predictions == labels.to(device)).sum().item() # count the number of correct predictions
            total += labels.size(0)

    return correct / total

def show_predictions(model, dataset, device, count=12, output_path="predictions/mnist.png"):
    model.eval()
    wrong_predictions = []
    first_predictions = []

    with torch.no_grad():
        for index in range(len(dataset)):
            image, label = dataset[index]
            logits = model(image.unsqueeze(0).to(device))
            prediction = logits.argmax(dim=1).item()

            prediction_data = (image, label, prediction)
            if len(first_predictions) < count:
                first_predictions.append(prediction_data)
            if prediction != label:
                wrong_predictions.append(prediction_data)
                if len(wrong_predictions) == count:
                    break

    predictions_to_show = wrong_predictions or first_predictions
    columns = 4
    rows = (len(predictions_to_show) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(8, 2 * rows))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for axis, (image, label, prediction) in zip(axes_list, predictions_to_show):
        axis.imshow(image.squeeze(), cmap="gray")
        axis.set_title(f"true: {label}, predicted: {prediction}")
        axis.axis("off")

    for axis in axes_list[len(predictions_to_show):]:
        axis.axis("off")

    plt.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved predictions to {output_file}")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor() # transforms from PIL image to tensor

    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256)

    model = MNISTS5().to(device) # move to cuda or cpu depending on availability
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    accuracy_file = Path("predictions/accuracy.txt")
    accuracy_file.parent.mkdir(parents=True, exist_ok=True)

    with accuracy_file.open("w", encoding="utf-8") as file:
        for epoch in range(5):
            model.train()
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = loss_function(logits, labels)

                optimizer.zero_grad() # clear gradients before backward pass so that they don't add up
                loss.backward() # calculate gradients of loss w.r.t. model parameters
                optimizer.step() # update model parameters based on gradients

            accuracy = evaluate(model, test_loader, device)
            result = f"epoch {epoch + 1}: test accuracy = {accuracy:.2%}"
            print(result)
            file.write(result + "\n")

    show_predictions(model, test_data, device)


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")
