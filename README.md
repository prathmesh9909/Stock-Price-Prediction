# 📈 Stock Price Movement Predictor

An end-to-end **Deep Learning + Flask** project that uses historical stock-market data, technical indicators, and an **LSTM (Long Short-Term Memory)** neural network to predict the next stock closing price and derive an expected market trend.

> **Educational disclaimer:** This project is intended for academic/educational use. Stock-market predictions are uncertain and the generated BUY/SELL/HOLD labels are not financial advice.

---

## 📌 Project Overview

This project was developed as a practical application of **Machine Learning / Deep Learning for financial time-series data**.

The system:

1. Downloads historical stock data using `yfinance`.
2. Preprocesses the OHLCV data.
3. Calculates technical indicators:
   - RSI
   - MACD
   - MACD Signal
   - Bollinger Bands
4. Scales eight features using `MinMaxScaler`.
5. Creates 60-day sliding-window sequences.
6. Trains a two-layer LSTM model using PyTorch.
7. Saves the trained model and scaler.
8. Provides a Flask web application where a user can enter a stock ticker.
9. Fetches current historical data for the ticker and generates a predicted price.
10. Displays the current price, predicted price, trend, recommendation, estimated confidence score, and an interactive candlestick chart with Bollinger Bands.

---

## 🎯 Problem Statement

Stock prices are time-series data affected by many changing factors. The objective of this project is to build a hands-on deep learning system that learns patterns from historical stock prices and technical indicators and provides a simple web interface for generating a predicted next price.

---

## 🧠 How the Model Works

The LSTM model receives sequences containing **60 previous trading observations**.

### Input features

The model uses 8 features:

```text
1. Close
2. RSI
3. MACD
4. MACD_Signal
5. BB_Upper
6. BB_Lower
7. BB_Middle
8. Volume
```

The features are normalized using `MinMaxScaler`.

### LSTM architecture

The implemented model contains:

```text
Input size     : 8
Hidden size    : 64
LSTM layers    : 2
Dropout        : 0.2
Output layer   : Linear(64 → 1)
Loss function  : Mean Squared Error (MSE)
Optimizer      : Adam
Learning rate  : 0.001
Epochs         : 30
Batch size     : 32
Sequence length: 60
```

The model predicts a **normalized next closing price**. The prediction is then converted back to the original price scale using the saved scaler.

---

## 📊 Dataset

The included training data is based on **Apple Inc. (AAPL)** historical stock data downloaded through Yahoo Finance using `yfinance`.

The included raw dataset covers approximately:

```text
Start: 2015-01-02
End:   2024-12-31
```

The processed dataset contains **2,483 rows** after technical-indicator calculation and removal of missing values.

The prepared sequence data contains:

```text
X shape: (2423, 60, 8)
y shape: (2423,)
```

### Raw data

`data/stock_data.csv`

Contains the downloaded OHLCV data.

### Processed data

`data/final_data.csv`

Contains the cleaned data and calculated technical indicators.

### Prepared NumPy data

```text
data/X.npy
data/y.npy
```

These contain the 60-step sequences and target values used for LSTM training.

---

## 📈 Technical Indicators

### RSI — Relative Strength Index

RSI is calculated using the `ta` library and provides a momentum indicator based on recent price movements.

### MACD — Moving Average Convergence Divergence

The project calculates:

- MACD
- MACD Signal

These provide trend and momentum information.

### Bollinger Bands

The project calculates:

- Upper Band
- Middle Band
- Lower Band

These are also displayed on the application's candlestick chart.

---

## 🔄 Project Workflow

```text
Yahoo Finance
     ↓
download_data.py
     ↓
stock_data.csv
     ↓
preprocess.py
     ↓
RSI + MACD + Bollinger Bands
     ↓
final_data.csv
     ↓
prepare_data.py
     ↓
60-day sequences + MinMaxScaler
     ↓
X.npy + y.npy
     ↓
train.py
     ↓
LSTM training
     ↓
lstm_model.pth + scaler.pkl
     ↓
predict.py / app.py
     ↓
Predicted Price
     ↓
Trend + Recommendation + Chart
```

---

## 🌐 Flask Web Application

The web application is implemented in `app.py`.

### Home page

Users enter a stock ticker such as:

```text
AAPL
TSLA
MSFT
TCS.NS
INFY.NS
RELIANCE.NS
```

### Prediction page

The application displays:

- Current price
- Predicted price
- Market trend
- BUY / SELL / HOLD recommendation
- Estimated confidence score
- Percentage price change
- Interactive candlestick chart
- Bollinger Bands

---

## 📁 Project Structure

```text
StockPrediction/
│
├── data/
│   ├── final_data.csv
│   ├── stock_data.csv
│   ├── X.npy
│   └── y.npy
│
├── model/
│   ├── lstm_model.pth
│   └── scaler.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── download_data.py
├── predict.py
├── prepare_data.py
├── preprocess.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📝 Description of Python Files

### `download_data.py`

Downloads AAPL historical stock data from Yahoo Finance for the period used for training and saves it as:

```text
data/stock_data.csv
```

The ticker can be changed in the script.

### `preprocess.py`

Reads the raw CSV and:

- Cleans the Yahoo Finance CSV structure.
- Converts dates and numerical columns.
- Calculates RSI.
- Calculates MACD and MACD Signal.
- Calculates Bollinger Bands.
- Removes rows containing missing indicator values.
- Saves the processed data to `data/final_data.csv`.

### `prepare_data.py`

- Selects the 8 model features.
- Fits a `MinMaxScaler`.
- Creates 60-day sequences.
- Saves the scaler as `model/scaler.pkl`.
- Saves training arrays as `data/X.npy` and `data/y.npy`.

### `train.py`

- Loads `X.npy` and `y.npy`.
- Splits the data into training and testing sets using an 80/20 chronological split.
- Trains the LSTM for 30 epochs.
- Uses MSE loss and Adam optimization.
- Saves the trained model as `model/lstm_model.pth`.
- Calculates test loss.

### `predict.py`

Loads the saved model and scaler and predicts the next closing price from the latest 60 observations in `final_data.csv`.

### `app.py`

Runs the Flask web application and performs prediction using data downloaded dynamically from Yahoo Finance.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/prathmesh9909/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment on Windows

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Web Application

Run:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

Enter a stock ticker and click **Predict Stock**.

---

## 🧪 Rebuild the Model

If you want to reproduce the training pipeline from the raw data:

### Step 1 — Download data

```bash
python download_data.py
```

### Step 2 — Preprocess data

```bash
python preprocess.py
```

### Step 3 — Prepare sequences

```bash
python prepare_data.py
```

### Step 4 — Train the LSTM

```bash
python train.py
```

### Step 5 — Test prediction

```bash
python predict.py
```

### Step 6 — Run the web application

```bash
python app.py
```

---

## ⚠️ Important Implementation Notes

### Training ticker

The included training dataset is **AAPL**. Although the Flask interface accepts other tickers, the saved model was trained on the AAPL-based feature distribution.

Therefore, predictions for other stocks should be treated as experimental unless the model is retrained appropriately for those stocks.

### Confidence score

The application's displayed confidence value is a **heuristic estimate** calculated from the predicted percentage change. It is not a statistically calibrated probability.

### Recommendation

The BUY/SELL/HOLD label is generated from the predicted percentage change:

```text
Change > +1%  → BUY
Change < -1%  → SELL
Otherwise     → HOLD
```

This is a project rule and should not be interpreted as professional investment advice.

---

## 🚀 Future Enhancements

- Train separate models for different stocks or a multi-stock dataset.
- Add proper classification for Up/Down movement.
- Add calibrated prediction probabilities.
- Add validation metrics such as MAE and RMSE.
- Add walk-forward time-series validation.
- Add model performance graphs.
- Add RSI and MACD visualizations to the web interface.
- Add backtesting and strategy comparison.
- Compare LSTM with GRU, Transformer, Random Forest, and other models.
- Deploy the Flask application online.

---

## 🛠️ Technologies Used

```text
Python
PyTorch
LSTM
Flask
Pandas
NumPy
Scikit-learn
yfinance
TA
Joblib
Plotly
HTML
CSS
Bootstrap
JavaScript
```

---

## 👩‍💻 Author

**Shravani Patil**

GitHub: https://github.com/prathmesh9909

---

## 📄 License

This project is created for educational and academic purposes.

## ⚠️ Disclaimer

Financial markets are unpredictable. Historical patterns do not guarantee future results. The predictions and recommendations produced by this project should not be considered financial advice.
