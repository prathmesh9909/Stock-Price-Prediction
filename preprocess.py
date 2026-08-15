import pandas as pd
import ta
import os

os.makedirs("data", exist_ok=True)

# Read CSV
df = pd.read_csv("data/stock_data.csv")

# Rename first column to Date
df.rename(columns={"Price": "Date"}, inplace=True)

# Remove the extra header rows
df = df.iloc[2:].reset_index(drop=True)

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Convert numeric columns
numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col])

# Calculate RSI
df["RSI"] = ta.momentum.RSIIndicator(close=df["Close"]).rsi()

# Calculate MACD
macd = ta.trend.MACD(close=df["Close"])
df["MACD"] = macd.macd()
df["MACD_Signal"] = macd.macd_signal()

# Calculate Bollinger Bands
bb = ta.volatility.BollingerBands(close=df["Close"])

df["BB_Upper"] = bb.bollinger_hband()
df["BB_Lower"] = bb.bollinger_lband()
df["BB_Middle"] = bb.bollinger_mavg()

# Remove NaN values
df.dropna(inplace=True)

# Save processed data
df.to_csv("data/final_data.csv", index=False)

print(df.head())

print("\n✅ Preprocessing Completed Successfully!")
print("Rows:", len(df))