import torch
import torch.nn as nn
import torch.optim as optim

# Example inputs and outputs for teaching addition
X = torch.tensor([[2.0, 3.0], [10.0, 4.0], [1.0, 7.0], [6.0, 5.0]])
y = torch.tensor([[5.0], [14.0], [8.0], [11.0]])  # correct answers

# Create a simple neural network
model = nn.Sequential(
    nn.Linear(2, 1)  # Input: 2 numbers, Output: 1 number (sum)
)

# Loss Function
loss_fn = nn.MSELoss()  # Mean Squared Error (how wrong it is)

# Optimizer (Brain Helper)
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Learning rate = 0.01

# Training Loop, Now the robot will practice 1000 times.
for epoch in range(10000):
    # 1. Predict
    y_pred = model(X)

    # 2. Calculate loss (how wrong)
    loss = loss_fn(y_pred, y)

    # 3. Reset gradients
    optimizer.zero_grad()

    # 4. Backpropagate (learn from mistake)
    loss.backward()

    # 5. Update model
    optimizer.step()

    # Print loss every 100 steps
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


torch.save(model.state_dict(), "pytorch-learn/data/model_weights_2.pth")

# Test-1
# test = torch.tensor([8.0, 3.0])  # 8 + 3 = 11
# output = model(test)
# print("Model says:", output.item())

# # Test-2
# test = torch.tensor([1.0, 1.0])  # 1 + 1 = 2
# output = model(test)
# print("Model says:", output.item())

# # Test-3
# test = torch.tensor([10.0, 10.0])  # 10 + 10 = 20
# output = model(test)
# print("Model says:", output.item())
