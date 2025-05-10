import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm
import sys
import os
from torch.utils.data import DataLoader, Dataset, random_split
from ClassModels import ConvNextClassificationModel

# Configuration parameters
images_folder = 'dataset/CassavaLeafDisease-data/images'
labels_csv = 'dataset/CassavaLeafDisease-data/label.csv'
batch_size = 64
learning_rate = 0.0001
n_epochs = 20
num_classes = 5
image_size = (96, 96)
train_split_ratio = 0.8

# Function to determine the device to use for training
def set_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = set_device()

# Load the labels from the CSV file
labels_df = pd.read_csv(labels_csv)

# Define the image transformations for data augmentation
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# Custom dataset class for loading images and labels
class CassavaDataset(Dataset):
    def __init__(self, images_folder, labels_df, transform=None):
        self.images_folder = images_folder
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        label = self.labels_df.iloc[idx, 1]
        img_path = os.path.join(self.images_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Create the full dataset
full_dataset = CassavaDataset(images_folder, labels_df, transform=transform)

# Split the dataset into training and testing sets
train_size = int(train_split_ratio * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Class to adjust the learning rate based on training progress
class LearningRateAdjuster:
    def __init__(self, model, patience, stop_patience, threshold, factor, dwell, model_name, batches, epochs, ask_epoch):
        self.model = model
        self.patience = patience
        self.stop_patience = stop_patience
        self.threshold = threshold
        self.factor = factor
        self.dwell = dwell
        self.model_name = model_name
        self.batches = batches
        self.epochs = epochs
        self.ask_epoch = ask_epoch
        self.best_weights = model.state_dict()
        self.best_accuracy = 0
        self.no_improvement_count = 0
        self.lr_adjustment_count = 0

    def adjust_learning_rate(self, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= self.factor
        print(f"Learning rate adjusted by factor {self.factor}")

    def check_early_stopping(self):
        if self.lr_adjustment_count >= self.stop_patience:
            print("Early stopping triggered.")
            return True
        return False

    def restore_best_weights(self):
        if self.dwell:
            self.model.load_state_dict(self.best_weights)
            print("Restored best model weights.")

    def prompt_user(self):
        response = input("Do you want to continue training? (y/n): ")
        return response.lower() == 'y'

    def update(self, epoch, epoch_acc, optimizer, train_acc):
        if epoch_acc > self.best_accuracy:
            self.best_accuracy = epoch_acc
            self.best_weights = self.model.state_dict()
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if train_acc >= self.threshold and self.no_improvement_count >= self.patience:
            self.adjust_learning_rate(optimizer)
            self.lr_adjustment_count += 1
            self.no_improvement_count = 0
            self.restore_best_weights()

        if epoch >= self.ask_epoch and not self.prompt_user():
            return True

        return self.check_early_stopping()

# Function to evaluate the model on the test set
def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = data 
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_correctly_on_epoch += (predicted == labels).sum().item()

    epoch_acc = 100.0 * predicted_correctly_on_epoch / total 
    print(f"    - Testing dataset. Got {predicted_correctly_on_epoch} out of {total} images correctly ({epoch_acc:.3f}%)")
    return epoch_acc

# Function to train the neural network
def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs, lra):
    for epoch in range(n_epochs):
        print(f"Epoch number {epoch + 1}")
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0 

        for data in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
            images, labels = data 
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.00 * running_correct / total 

        print(f"   - Training dataset got {running_correct} out of {total} images correctly ({epoch_acc:.3f}%). Epoch loss: {epoch_loss:.3f}")
        test_acc = evaluate_model_on_test_set(model, test_loader)

        if lra.update(epoch, test_acc, optimizer, epoch_acc):
            break

    print("Finished")
    return model 

# Main function to set up and start the training process
def main():
    model = ConvNextClassificationModel(num_classes=num_classes, pretrained=True, keep_n_layers=5, freeze_layers=False)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    lra = LearningRateAdjuster(
        model=model,
        patience=3,
        stop_patience=5,
        threshold=97.0,
        factor=0.5,
        dwell=True,
        model_name="ConvNext",
        batches=len(train_loader),
        epochs=n_epochs,
        ask_epoch=10
    )

    trained_model = train_nn(model, train_loader, test_loader, loss_fn, optimizer, n_epochs, lra)

if __name__ == "__main__":
    main()
