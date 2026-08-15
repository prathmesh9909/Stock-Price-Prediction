import yfinance as yf
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

ticker = "AAPL"      # Change to RELIANCE.NS, TCS.NS, INFY.NS if needed

df = yf.download(
    ticker,
    start="2015-01-01",
    end="2025-01-01"
)

print(df.head())

df.to_csv("data/stock_data.csv")

print(f"\nDownloaded {len(df)} rows successfully.")