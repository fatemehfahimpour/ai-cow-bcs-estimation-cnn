import os

import numpy as np
import torch
from trainer import Trainer
from bcs_estimation_proj.model import BCSResNet18
from data.dataset_dataloader import get_dataloaders


def find_parameters():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainable_layers = ['fc', 'layer4_fc', 'layer3_layer4_fc']
    optimizer_names = ['adam', 'momentum']
    adam_learning_rates = [3e-5, 1e-4, 3e-4, 1e-3]
    momentum_learning_rates = [0.001, 0.01, 0.03]

    # trainable_layers = ['layer4_fc']
    # optimizer_names = ['adam']
    # adam_learning_rates = [1e-3]
    # momentum_learning_rates = [0.001, 0.01, 0.03]

    best_model = None
    best_loss = np.inf
    best_history = None
    best_config = None

    test_counter = 0
    train_loader, val_loader, test_loader = get_dataloaders()
    for trainable_layer in trainable_layers:
        for optimizer_name in optimizer_names:
            learning_rates = None
            if optimizer_name == 'adam':
                learning_rates = adam_learning_rates
            elif optimizer_name == 'momentum':
                learning_rates = momentum_learning_rates
            for learning_rate in learning_rates:
                print(
                    f"....................................test number: {test_counter}.................................... ")
                test_counter += 1

                model = BCSResNet18(trainable_layers=trainable_layer).to(device)
                trainer = Trainer(model, train_loader, val_loader, device, learning_rate, optimizer_name, )
                history, train_loss, val_loss = trainer.fit()

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = model
                    best_history = history
                    best_config = {'trainable_layers': trainable_layer,
                                   'optimizer': optimizer_name,
                                   'learning_rate': learning_rate}

    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    checkpoint = {
        'model_state_dict': best_model.state_dict(),
        'config': best_config,
        'history': best_history
    }

    torch.save(checkpoint, os.path.join(results_dir, 'best_model.pth'))

    print(best_config)
    print(best_loss)


if __name__ == '__main__':
    find_parameters()
