from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import yfinance as yf
import ta
import plotly.graph_objects as go

app = Flask(__name__)

# ===========================
# LSTM Model
# ===========================
class StockLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# ===========================
# Load Model
# ===========================
model = StockLSTM()
model.load_state_dict(torch.load("model/lstm_model.pth"))
model.eval()

# Load Scaler
scaler = joblib.load("model/scaler.pkl")


# ===========================
# Home Page
# ===========================
@app.route("/")
def home():
    return render_template("index.html")


# ===========================
# Prediction Route
# ===========================
@app.route("/predict", methods=["POST"])
def predict():

    ticker = request.form["ticker"].upper()

    try:

        # Download Live Stock Data
        df = yf.download(
            ticker,
            start="2015-01-01",
            progress=False
        )

        if df.empty:
            return "Invalid Stock Symbol!"

        df = df.reset_index()

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Technical Indicators
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()

        bb = ta.volatility.BollingerBands(df["Close"])
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Middle"] = bb.bollinger_mavg()
        df["BB_Lower"] = bb.bollinger_lband()

        df.dropna(inplace=True)

        # Features
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

        scaled = scaler.transform(df[features])

        last60 = scaled[-60:]

        X = np.array([last60])

        X = torch.FloatTensor(X)

        # Prediction
        with torch.no_grad():
            prediction = model(X).item()

        dummy = np.zeros((1, 8))
        dummy[0, 0] = prediction

        predicted_price = scaler.inverse_transform(dummy)[0][0]

        current_price = float(df["Close"].iloc[-1])

        # Trend
        if predicted_price > current_price:
            trend = "📈 UP"
        else:
            trend = "📉 DOWN"

        # Percentage Change
        change = ((predicted_price - current_price) / current_price) * 100

        # Recommendation
        if change > 1:
            recommendation = "🟢 BUY"
        elif change < -1:
            recommendation = "🔴 SELL"
        else:
            recommendation = "🟡 HOLD"

        # Confidence Score (temporary estimate)
        confidence = round( min(95, max(55, 100 - abs(change))),2)

        # ===========================
        # Candlestick Chart
        # ===========================
        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price"
            )
        )

        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["BB_Upper"],
            mode="lines",
            name="Upper Band"
        ))

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["BB_Middle"],
            mode="lines",
            name="Middle Band"
        ))

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["BB_Lower"],
            mode="lines",
            name="Lower Band"
        ))

        fig.update_layout(
            title=f"{ticker} Stock Price",
            template="plotly_white",
            xaxis_rangeslider_visible=False,
            height=600
        )

        chart = fig.to_html(full_html=False)

        return render_template(
            "result.html",
            ticker=ticker,
            current=round(current_price, 2),
            predicted=round(predicted_price, 2),
            trend=trend,
            recommendation=recommendation,
            confidence=confidence,
            change=round(change, 2),
            chart=chart
        )

    except Exception as e:
        return f"Error: {e}"


# ===========================
# Run Flask
# ===========================
if __name__ == "__main__":
    app.run(debug=True)