import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# -----------------------------
# LSTM Model (same as train.py)
# -----------------------------
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

# -----------------------------
# Load Model
# -----------------------------
model = StockLSTM()
model.load_state_dict(torch.load("model/lstm_model.pth"))
model.eval()

# -----------------------------
# Load Scaler
# -----------------------------
scaler = joblib.load("model/scaler.pkl")

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("data/final_data.csv")

features = [
    "Close",
    "RSI",
    "MACD",
    "MACD_Signal",
    "BB_Upper",
    "BB_Lower",
    "BB_Middle",
    "Volume"
]

data = df[features]

scaled = scaler.transform(data)

# Last 60 Days
last_60 = scaled[-60:]

X = np.array([last_60])

X = torch.FloatTensor(X)

# -----------------------------
# Predict
# -----------------------------
with torch.no_grad():
    prediction = model(X).item()

# Convert back to original price
dummy = np.zeros((1, 8))
dummy[0, 0] = prediction

predicted_price = scaler.inverse_transform(dummy)[0][0]

current_price = df["Close"].iloc[-1]

print("\n==========================")
print("Current Price :", round(current_price, 2))
print("Predicted Price:", round(predicted_price, 2))

if predicted_price > current_price:
    print("Prediction : UP 📈")
else:
    print("Prediction : DOWN 📉")

print("==========================")