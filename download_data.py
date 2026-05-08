import yfinance as yf

tickers = ["AAPL", "TSLA", "MSFT", "SPY", "BND"]

data = yf.download(
    tickers,
    start="2024-01-01",
    end="2026-05-01",
    interval="1d"
)

prices = data["Close"]

prices.to_csv("data/prices.csv")

print(prices.head())
print("CSV saved successfully.")