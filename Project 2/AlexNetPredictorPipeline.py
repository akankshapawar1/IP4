import os
import random
import torch
from torch import nn, optim
from torchvision import transforms
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from PIL import Image
from torch.utils.data import Dataset, DataLoader

device = ("cuda" if torch.cuda.is_available() else "cpu")


class AlexNet(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(AlexNet, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.linear = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutional(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = [0 if label == 'melanoma' else 1 for label in labels]  # 0 for melanoma, 1 for naevus
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def balance_data(path):
    files = os.listdir(path)
    if len(files) == 70:
        print("Dataset already balanced.")
        return
    assert len(files) == 100, "There are not exactly 100 images in the folder."
    files_to_delete = random.sample(files, 30)
    for file_name in files_to_delete:
        file_path = os.path.join(path, file_name)
        os.remove(file_path)
        print(f"Deleted {file_path}")
    print(f"Number of images left in the naevus folder: {len(os.listdir(path))}")


def train_test(folder_path, train_size, test_size):
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]
    random.shuffle(images)
    train_images = images[:train_size]
    test_images = images[train_size:train_size + test_size]
    return train_images, test_images


def prepare_datasets(melanoma_folder_path, naevus_folder_path, transform):
    train_melanoma, test_melanoma = train_test(melanoma_folder_path, 50, 20)
    train_naevus, test_naevus = train_test(naevus_folder_path, 50, 20)

    train_images = train_melanoma + train_naevus
    train_labels = ['melanoma'] * len(train_melanoma) + ['naevus'] * len(train_naevus)

    test_images = test_melanoma + test_naevus
    test_labels = ['melanoma'] * len(test_melanoma) + ['naevus'] * len(test_naevus)

    train_dataset = CustomImageDataset(train_images, train_labels, transform=transform)
    test_dataset = CustomImageDataset(test_images, test_labels, transform=transform)

    return train_dataset, test_dataset


def train_model(model, train_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    training_losses = []

    # Train the model
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())

    return model, training_losses


def model_test(model, test_loader, device):
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def cross_validate_with_dropout(train_dataset, dropout_rates, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    best_accuracy = 0
    best_dropout_rate = 0

    for dropout_rate in dropout_rates:
        accuracies = []
        for train_idx, val_idx in kf.split(train_dataset):
            train_sub = Subset(train_dataset, train_idx)
            val_sub = Subset(train_dataset, val_idx)

            train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_sub, batch_size=32)

            model = AlexNet(dropout_rate=dropout_rate).to(device)
            train_model(model, train_loader, device)
            accuracy = model_test(model, val_loader, device)
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        print(f"Dropout: {dropout_rate}, Accuracy: {mean_accuracy}")

        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_dropout_rate = dropout_rate

    return best_dropout_rate, best_accuracy


def main():
    # step 3
    naevus_folder_path = 'complete_mednode_dataset/naevus'
    balance_data(naevus_folder_path)
    melanoma_folder_path = 'complete_mednode_dataset/melanoma'

    # step 4
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset, test_dataset = prepare_datasets(melanoma_folder_path, naevus_folder_path, transform)

    dropout_rates = [0.6, 0.75, 1.0]
    best_dropout_rate, best_accuracy = cross_validate_with_dropout(train_dataset, dropout_rates)
    print(f"Best Dropout Rate: {best_dropout_rate}, Best Accuracy: {best_accuracy}")

    best_model = AlexNet(dropout_rate=best_dropout_rate).to(device)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    best_model, training_losses = train_model(best_model, train_loader, device)

    print("Training losses:", training_losses)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_accuracy = model_test(best_model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()
