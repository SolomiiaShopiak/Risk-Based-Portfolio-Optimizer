import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_returns(file_path="data/returns.csv"):
    returns = pd.read_csv(file_path, index_col=0, parse_dates=True)

    if returns.empty:
        raise ValueError("Returns data is empty")

    if returns.isnull().values.any():
        raise ValueError("Returns contain NaN values")

    return returns


def calculate_statistics(returns):
    """
    Calculate mean returns and covariance matrix
    """
    mu = returns.mean()
    cov_matrix = returns.cov()

    if (returns.std() == 0).any():
        raise ValueError("Some assets have zero volatility")

    return mu, cov_matrix


def generate_random_weights(n_assets):
    """
    Generate random weights that sum to 1
    """
    weights = np.random.random(n_assets)
    weights = weights / np.sum(weights)
    return weights


def portfolio_performance(weights, mu, cov_matrix):
    """
    Calculate portfolio return and risk
    """
    portfolio_return = np.dot(weights, mu)

    portfolio_risk = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix, weights))
    )

    return portfolio_return, portfolio_risk


def simulate_portfolios(returns, n_portfolios=1000):
    """
    Generate many random portfolios
    """
    mu, cov_matrix = calculate_statistics(returns)

    results = []
    weights_list = []

    for _ in range(n_portfolios):
        weights = generate_random_weights(len(mu))

        port_return, port_risk = portfolio_performance(
            weights, mu, cov_matrix
        )

        results.append((port_return, port_risk))
        weights_list.append(weights)

    results_df = pd.DataFrame(results, columns=["Return", "Risk"])

    return results_df, weights_list


def add_sharpe_ratio(results_df, risk_free_rate=0.0001):
    """
    Add Sharpe ratio column
    """
    results_df["Sharpe"] = (
        results_df["Return"] - risk_free_rate
    ) / results_df["Risk"]

    return results_df


def get_best_portfolio(results_df, weights_list):
    """
    Get portfolio with highest Sharpe ratio
    """
    max_sharpe_idx = results_df["Sharpe"].idxmax()

    best_portfolio = results_df.loc[max_sharpe_idx]
    best_weights = weights_list[max_sharpe_idx]

    return best_portfolio, best_weights


def plot_efficient_frontier(results_df):
    plt.figure(figsize=(10, 6))

    plt.scatter(
        results_df["Risk"],
        results_df["Return"],
        c=results_df["Sharpe"],
        cmap="viridis"
    )

    plt.xlabel("Risk (Volatility)")
    plt.ylabel("Return")
    plt.title("Efficient Frontier")

    plt.colorbar(label="Sharpe Ratio")

    plt.show()


if __name__ == "__main__":

    returns = load_returns()
    results, weights = simulate_portfolios(returns, 1000)
    results = add_sharpe_ratio(results)

    print("\nBest portfolio:")
    best_portfolio, best_weights = get_best_portfolio(results, weights)

    print(best_portfolio)

    print("\nWeights:")
    print(best_weights)

    print("\nTop 5 portfolios:")
    print(results.sort_values(by="Sharpe", ascending=False).head())
    plot_efficient_frontier(results)
