import os

import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

BCS_COLUMN_NAME = 'bcs_mean'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_PATH = os.path.join(BASE_DIR, 'BCS_dataset')

class BCSDataset(Dataset):
    def __init__(self, df, transform):
        self.data_frame = df
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx]['local_path']
        label = self.data_frame.iloc[idx][BCS_COLUMN_NAME]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


def get_transformers():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),

        transforms.ToTensor(),
        # mean and std of ImageNet used in ResNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_val_transform


def get_datasets():
    train_transform, test_val_transform = get_transformers()
    train_df = pd.read_csv(f'{MAIN_PATH}/train.csv')
    test_df = pd.read_csv(f'{MAIN_PATH}/test.csv')
    val_df = pd.read_csv(f'{MAIN_PATH}/val.csv')
    train_dataset = BCSDataset(train_df, train_transform)
    test_dataset = BCSDataset(test_df, test_val_transform)
    val_dataset = BCSDataset(val_df, test_val_transform)

    return train_dataset, test_dataset, val_dataset


def get_dataloaders():
    batch_size = 32
    train_dataset, test_dataset, val_dataset = get_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
