import os
import glob
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


BATCH_SIZE = 66
EPOCHS=5

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
model_dir = os.path.join(os.getcwd(), "model")
os.makedirs(model_dir, exist_ok=True)

train_path = os.getcwd()+"\\dataset\\train"
val_path = os.getcwd()+"\\dataset\\validation"
test_path = os.getcwd()+"\\dataset\\test"

# Image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # small size due to deep layers
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])



# Datasets and loaders
train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset = datasets.ImageFolder(val_path, transform=transform)

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

# Custom Test Dataset
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(test_dir, '*'))
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self.loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_path



test_dataset = TestDataset(test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# CNN Model Definition
class ConvBlock(nn.Module):
    def _init_(self, in_channels, out_channels):
        super(ConvBlock, self)._init_()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)


class GenderClassificationCNN(nn.Module):
    def __init__(self):
        super(GenderClassificationCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),  # Adjust dimensions properly
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def fit(model, train_loader, val_loader, optimizer, criterion, epochs=EPOCHS, device="cuda"):
    model.to(device)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_accuracy)

        # Logging results
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    print("\n✅ Training Complete")
    return history  # Returning history dictionary


# Initialize model
model = GenderClassificationCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training function
def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    accuracy = 100 * correct / total
    return running_loss / len(loader), accuracy

# Validation function
def evaluate(model, loader, criterion, mode="Validation"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=mode):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    return running_loss / len(loader), accuracy

history = fit(model, train_loader, val_loader, optimizer, criterion, epochs=EPOCHS, device=device)

model_path = os.path.join(model_dir, "gender_classifier_cnn.pth")
torch.save(model.state_dict(), model_path)

predictions = []

with torch.no_grad():
    for images, paths in tqdm(test_loader, desc="Predicting Test Set"):
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        label = "Male" if predicted.item() == 0 else "Female"
        filename = os.path.basename(paths[0])
        predictions.append((filename, label))

# Save to CSV
df = pd.DataFrame(predictions, columns=['filename', 'predicted_label'])
df.to_csv('test_predictions.csv', index=False)

print("\n✅ Predictions saved to test_predictions.csv")



validation_results = evaluate(model, val_loader, criterion)


# Plot accuracy graph
plt.figure(figsize=(8, 6))
plt.plot(history["train_acc"], label='Train Accuracy')
plt.plot(history["val_acc"], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs. Iteration")
plt.legend()
plt.grid()
plt.show()

# Plot loss graph
plt.figure(figsize=(8, 6))
plt.plot(history["train_loss"], label='Train Loss')
plt.plot(history["val_loss"], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs. Iteration")
plt.legend()
plt.grid()
plt.show()

# True and predicted labels
true_labels = []
pred_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, pred_labels)
class_names = train_dataset.classes  # ['MEN', 'WOMAN']

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Pick 3 random validation images since test has no labels
indices = random.sample(range(len(val_dataset)), 3)

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for i in indices:
        # Load image & true label from validation set
        img, true_label = val_dataset[i]
        img_tensor = img.unsqueeze(0).to(device)  # Add batch dimension

        # Predict label
        output = model(img_tensor)
        _, pred_label = torch.max(output, 1)

        # Get class names
        true_label_str = train_dataset.classes[true_label]  # Validation provides labels
        pred_label_str = train_dataset.classes[pred_label.item()]

        # Plot image
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        plt.title(f"True: {true_label_str}, Predicted: {pred_label_str}")
        plt.axis("off")
        plt.show()


from sklearn.metrics import classification_report

# Compute classification report
report = classification_report(true_labels, pred_labels, target_names=class_names)
print("\n📊 Performance Metrics:\n")
print(report)
