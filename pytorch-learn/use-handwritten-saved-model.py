import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Define the model structure again
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Load the saved model
model = DigitClassifier()
model.load_state_dict(torch.load("pytorch-learn/data/handwritten-number.pth"))
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),              # Convert to 1 channel
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

def load_own_image(path):
    image = Image.open(path).convert("L")  # Grayscale
    image = ImageOps.invert(image)         # Invert (white bg, black digit)
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)  # Resize & center

    # Auto contrast helps with pixel values (optional but helps)
    image = ImageOps.autocontrast(image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(image).unsqueeze(0)

# Load and predict
img = load_own_image("pytorch-learn/data/9.png")  # Your image here

with torch.no_grad():
    output = model(img)
    predicted = torch.argmax(output, 1).item()
    print("Model Prediction:", predicted)

def show_tensor_image(tensor):
    plt.imshow(tensor.squeeze(0).squeeze(0), cmap="gray")
    plt.title("Preprocessed Image Sent to Model")
    plt.show()

img = load_own_image("pytorch-learn/data/9.png")
show_tensor_image(img)
