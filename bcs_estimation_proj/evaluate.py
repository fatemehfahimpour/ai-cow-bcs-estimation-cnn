import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data.dataset_dataloader import get_dataloaders
from bcs_estimation_proj.model import BCSResNet18


def load_saved_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    trainable_layers = config['trainable_layers']
    history = checkpoint['history']

    model = BCSResNet18(trainable_layers=trainable_layers)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    print(f"Model loaded successfully with config: {config}")
    return model, config, history


def train_val_loss(best_history):
    plt.figure(figsize=(10, 6))

    plt.plot(
        best_history['train_loss'],
        label='Train Loss'
    )

    plt.plot(
        best_history['val_loss'],
        label='Validation Loss'
    )

    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Loss Curve of Best Model')

    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()


def testing(model, test_loader, device):
    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            outputs = outputs.squeeze(1)

            predictions.extend(outputs.cpu().numpy())
            actual.extend(labels.cpu().numpy())

    predictions = np.asarray(predictions)
    actual = np.asarray(actual)
    return predictions, actual


def testing_results(best_model, test_loader, device):
    predictions, actual = testing(best_model, test_loader, device)
    print(f"actual values: {actual[:10]}")
    print(f"predicted values: {predictions[:10]}")

    mse = mean_squared_error(actual, predictions)
    mae = mean_absolute_error(actual, predictions)
    print(f"MSE: {mse}")
    print(f"RMSE: {np.sqrt(mse)}")
    print(f"MAE: {mae}")

    # actual vs predictions
    plt.figure(figsize=(8, 8))

    plt.scatter(
        actual,
        predictions,
        alpha=0.6
    )

    min_value = min(actual.min(), predictions.min())
    max_value = max(actual.max(), predictions.max())

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle='--'
    )

    plt.xlabel("Actual BCS")
    plt.ylabel("Predicted BCS")
    plt.title("Actual vs Predicted BCS")
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    PATH = 'results/best_model.pth'

    model, config, history = load_saved_model(PATH)
    train_val_loss(history)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders()
    testing_results(model, test_loader, device)
