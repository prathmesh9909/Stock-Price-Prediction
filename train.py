import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Create model folder
os.makedirs("model", exist_ok=True)

# Load data
X = np.load("data/X.npy")
y = np.load("data/y.npy")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=False
)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.FloatTensor(y_train).view(-1, 1)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# LSTM Model
class StockLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Create model
model = StockLSTM()

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 30

print("\nTraining Started...\n")

# Training Loop
for epoch in range(epochs):

    model.train()
    running_loss = 0

    for inputs, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

# Save Model
torch.save(model.state_dict(), "model/lstm_model.pth")

print("\nModel saved successfully!")

# Evaluate
model.eval()

test_loss = 0

with torch.no_grad():

    for inputs, labels in test_loader:

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        test_loss += loss.item()

test_loss /= len(test_loader)

print(f"\nTest Loss: {test_loss:.6f}")

print("\nTraining Completed Successfully!")