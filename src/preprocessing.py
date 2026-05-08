import pandas as pd

def load_price_data(file_path: str) -> pd.DataFrame:

    data = pd.read_csv(file_path)

    if data.empty:
        raise ValueError("Dataset is empty.")
    if "Date" not in data.columns:
        raise ValueError("Missing Date column.")

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.set_index("Date")

    data = data.sort_index()

    return data

def clean_price_data(prices: pd.DataFrame) -> pd.DataFrame:

    if prices.empty:
        raise ValueError("Price data is empty.")

    prices = prices.dropna(how="all")

    prices = prices.ffill()
    prices = prices.bfill()

    prices = prices.drop_duplicates()

    return prices


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:

    if prices.empty:
        raise ValueError("Price data is empty.")

    returns = prices.pct_change()

    returns = returns.dropna()

    return returns