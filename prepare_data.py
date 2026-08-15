import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

os.makedirs("model", exist_ok=True)

# Load processed data
df = pd.read_csv("data/final_data.csv")

# Features for LSTM
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

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")

sequence_length = 60

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# Save
np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("X Shape:", X.shape)
print("y Shape:", y.shape)

print("\nData Prepared Successfully!")