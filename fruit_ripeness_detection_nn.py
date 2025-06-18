import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 32
img_size = 100
num_epochs = 5
learning_rate = 0.001

# Transformations
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# Load dataset
dataset = datasets.ImageFolder(root='images/Strawberries', transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
class_names = dataset.classes  # ['apple_ripe', ..., 'strawberry_rotten']


# Define CNN
class FruitCNN(nn.Module):
    def __init__(self):
        super(FruitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 25 * 25, 64), nn.ReLU(),
            nn.Linear(64, len(class_names))  # 9 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Initializing CNN
model = FruitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training CNN
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")


# --- Predict Function ---
def predict_image(image_path):
    model.eval()

    # Preprocessing- Load and transform image
    image = Image.open(image_path).convert('RGB')  # preprocessing image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()]
        parts = label.split('_')
        if len(parts) == 2:
            fruit, condition = parts
        else:
            fruit, condition = parts[0], " "

    # Display image with title
    plt.imshow(image)
    plt.title(f"{fruit.capitalize()} {condition}", fontsize=16)
    plt.axis('off')
    plt.show()


# Call Predict Function to evaluate and display an image from a directory
predict_image("images/Apples/RipeApples/RipeApple (23).jpg")


import torch
import numpy as np
import matplotlib.pyplot as plt

# Set the model to evaluation mode
model.eval()

# Initialize confusion matrix
num_classes = len(class_names)
confusion = np.zeros((num_classes, num_classes), dtype=int)

# Generate predictions and build confusion matrix
with torch.no_grad():
    for images, labels in data_loader:  # Replace with test_loader if available
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            confusion[true][pred] += 1

# --- Plotting the Confusion Matrix ---
plt.figure(figsize=(10, 8))
plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Annotate the matrix cells
thresh = confusion.max() / 2
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, str(confusion[i, j]),
                 ha="center", va="center",
                 color="white" if confusion[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()
