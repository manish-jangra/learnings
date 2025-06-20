import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load MNIST dataset
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)

# Build the Neural Network
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 output classes (0–9)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No softmax here; it's handled by loss function

model = DigitClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(5):
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

correct = 0
total = 0
model.eval()  # Set model to eval mode

# Test the Model
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")


# View one sample image and prediction
image, label = test_data[0]
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Actual: {label}")
plt.show()

model.eval()
with torch.no_grad():
    output = model(image.unsqueeze(0))
    pred = torch.argmax(output, 1).item()
    print("Model Prediction:", pred)

# Saving the model
torch.save(model.state_dict(), "pytorch-learn/data/handwritten-number.pth")
