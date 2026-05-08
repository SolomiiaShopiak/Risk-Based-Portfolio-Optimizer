from src.preprocessing import load_price_data, clean_price_data, calculate_returns


prices = load_price_data("data/prices.csv")

print("PRICES:")
print(prices.head())

clean_prices = clean_price_data(prices)

print("\nCLEAN PRICES:")
print(clean_prices.head())

returns = calculate_returns(clean_prices)

returns.to_csv("data/returns.csv")

print("\nRETURNS:")
print(returns.head())

print("\nData shape:")
print("Prices:", prices.shape)
print("Returns:", returns.shape)