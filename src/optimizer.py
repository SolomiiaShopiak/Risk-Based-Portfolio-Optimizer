import numpy as np
import pandas as pd
from scipy.optimize import minimize


RISK_FREE_RATE = 0.0001
TRADING_DAYS = 252

RISK_PROFILES = {
    "low":    {"max_volatility": 0.010, "min_sharpe": 0.5},
    "medium": {"max_volatility": 0.018, "min_sharpe": 0.3},
    "high":   {"max_volatility": float("inf"), "min_sharpe": 0.0},
}


def _portfolio_return(weights: np.ndarray, mu: pd.Series) -> float:
    return float(np.dot(weights, mu))


def _portfolio_volatility(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
    return float(np.sqrt(variance))


def _sharpe_ratio(weights: np.ndarray, mu: pd.Series, cov_matrix: pd.DataFrame,
                  risk_free_rate: float = RISK_FREE_RATE) -> float:
    ret = _portfolio_return(weights, mu)
    vol = _portfolio_volatility(weights, cov_matrix)
    if vol == 0:
        return 0.0
    return (ret - risk_free_rate) / vol


def _neg_sharpe(weights, mu, cov_matrix, risk_free_rate):
    """Objective function: negative Sharpe (scipy minimizes)."""
    return -_sharpe_ratio(weights, mu, cov_matrix, risk_free_rate)


def _base_constraints(n_assets: int) -> list:
    """Weights must sum to 1."""
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]


def _base_bounds(n_assets: int) -> tuple:
    """Each weight in [0, 1] — long-only portfolio."""
    return tuple((0.0, 1.0) for _ in range(n_assets))



def maximize_sharpe(mu: pd.Series, cov_matrix: pd.DataFrame,
                    risk_free_rate: float = RISK_FREE_RATE) -> dict:
    """
    Find the portfolio that maximises the Sharpe Ratio.

    Returns a dict with weights, return, volatility, and Sharpe ratio.
    """
    n = len(mu)
    w0 = np.ones(n) / n  # equal-weight starting point

    result = minimize(
        _neg_sharpe,
        w0,
        args=(mu, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=_base_bounds(n),
        constraints=_base_constraints(n),
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"Sharpe maximisation failed: {result.message}")

    weights = result.x
    return _build_result(weights, mu, cov_matrix, risk_free_rate)




def minimize_volatility(mu: pd.Series, cov_matrix: pd.DataFrame,
                        risk_free_rate: float = RISK_FREE_RATE) -> dict:
    """
    Find the minimum-variance (lowest risk) portfolio.
    Useful as the anchor for the 'low' risk profile.
    """
    n = len(mu)
    w0 = np.ones(n) / n

    result = minimize(
        lambda w: _portfolio_volatility(w, cov_matrix),
        w0,
        method="SLSQP",
        bounds=_base_bounds(n),
        constraints=_base_constraints(n),
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"Min-volatility optimisation failed: {result.message}")

    weights = result.x
    return _build_result(weights, mu, cov_matrix, risk_free_rate)




def optimize_for_target_return(mu: pd.Series, cov_matrix: pd.DataFrame,
                                target_return: float,
                                risk_free_rate: float = RISK_FREE_RATE) -> dict:
    """
    Minimise volatility subject to achieving at least `target_return`.
    Used to construct the medium-risk portfolio.
    """
    n = len(mu)
    w0 = np.ones(n) / n

    constraints = _base_constraints(n) + [
        {"type": "ineq", "fun": lambda w: _portfolio_return(w, mu) - target_return}
    ]

    result = minimize(
        lambda w: _portfolio_volatility(w, cov_matrix),
        w0,
        method="SLSQP",
        bounds=_base_bounds(n),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"Target-return optimisation failed: {result.message}")

    weights = result.x
    return _build_result(weights, mu, cov_matrix, risk_free_rate)



def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """
    Calculate CAPM beta of a single asset vs the market.

    β = Cov(R_i, R_m) / Var(R_m)
    """
    aligned = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if aligned.empty or aligned.shape[0] < 2:
        raise ValueError("Not enough overlapping data to compute beta.")

    ri = aligned.iloc[:, 0].values
    rm = aligned.iloc[:, 1].values

    cov = np.cov(ri, rm)[0, 1]
    var_m = np.var(rm, ddof=1)

    if var_m == 0:
        raise ValueError("Market returns have zero variance.")

    return cov / var_m


def calculate_all_betas(returns: pd.DataFrame,
                        market_column: str = "SPY") -> pd.Series:
    """
    Calculate beta for every asset in `returns` relative to `market_column`.

    Returns a Series indexed by ticker.
    """
    if market_column not in returns.columns:
        raise ValueError(f"Market column '{market_column}' not found in returns.")

    market = returns[market_column]
    betas = {}

    for ticker in returns.columns:
        if ticker == market_column:
            betas[ticker] = 1.0  # market beta is by definition 1
            continue
        betas[ticker] = calculate_beta(returns[ticker], market)

    return pd.Series(betas, name="Beta")


def capm_expected_return(beta: float, market_return: float,
                         risk_free_rate: float = RISK_FREE_RATE) -> float:
    """
    CAPM expected return for a single asset.

    E(R_i) = R_f + β * (R_m - R_f)
    """
    return risk_free_rate + beta * (market_return - risk_free_rate)


def capm_expected_returns_all(betas: pd.Series, market_return: float,
                               risk_free_rate: float = RISK_FREE_RATE) -> pd.Series:
    """
    CAPM expected returns for all assets given their betas.
    """
    return betas.apply(
        lambda b: capm_expected_return(b, market_return, risk_free_rate)
    ).rename("CAPM_Expected_Return")



def get_portfolio_for_risk_profile(mu: pd.Series, cov_matrix: pd.DataFrame,
                                    risk_level: str,
                                    risk_free_rate: float = RISK_FREE_RATE) -> dict:
    """
    Return an optimised portfolio for the requested risk level.

    risk_level : "low" | "medium" | "high"

    Strategy:
        low    → minimum-volatility portfolio
        medium → minimise volatility subject to reaching the midpoint
                 between min-vol return and max-Sharpe return
        high   → maximum Sharpe Ratio portfolio
    """
    risk_level = risk_level.lower()
    if risk_level not in RISK_PROFILES:
        raise ValueError(f"risk_level must be one of {list(RISK_PROFILES.keys())}")

    if risk_level == "low":
        portfolio = minimize_volatility(mu, cov_matrix, risk_free_rate)
        portfolio["profile"] = "low"
        return portfolio

    if risk_level == "high":
        portfolio = maximize_sharpe(mu, cov_matrix, risk_free_rate)
        portfolio["profile"] = "high"
        return portfolio

    # medium: aim for the midpoint return between min-vol and max-Sharpe
    min_vol = minimize_volatility(mu, cov_matrix, risk_free_rate)
    max_sr  = maximize_sharpe(mu, cov_matrix, risk_free_rate)

    target_return = (min_vol["return"] + max_sr["return"]) / 2

    try:
        portfolio = optimize_for_target_return(
            mu, cov_matrix, target_return, risk_free_rate
        )
    except RuntimeError:
        # Fallback: return max-Sharpe if the constraint is infeasible
        portfolio = max_sr

    portfolio["profile"] = "medium"
    return portfolio



def annualize_return(daily_return: float,
                     trading_days: int = TRADING_DAYS) -> float:
    return (1 + daily_return) ** trading_days - 1


def annualize_volatility(daily_volatility: float,
                         trading_days: int = TRADING_DAYS) -> float:
    return daily_volatility * np.sqrt(trading_days)



def _build_result(weights: np.ndarray, mu: pd.Series,
                  cov_matrix: pd.DataFrame,
                  risk_free_rate: float) -> dict:
    ret = _portfolio_return(weights, mu)
    vol = _portfolio_volatility(weights, cov_matrix)
    sharpe = _sharpe_ratio(weights, mu, cov_matrix, risk_free_rate)

    return {
        "weights":            dict(zip(mu.index, weights.round(6))),
        "return":             ret,
        "volatility":         vol,
        "sharpe_ratio":       sharpe,
        "annual_return":      annualize_return(ret),
        "annual_volatility":  annualize_volatility(vol),
    }


if __name__ == "__main__":
    from portfolio_model import load_returns, calculate_statistics

    returns = load_returns("data/returns.csv")
    mu, cov = calculate_statistics(returns)

    print("=" * 55)
    for level in ("low", "medium", "high"):
        p = get_portfolio_for_risk_profile(mu, cov, level)
        print(f"\n[{level.upper()} RISK]")
        print(f"  Weights        : {p['weights']}")
        print(f"  Daily Return   : {p['return']:.6f}")
        print(f"  Daily Vol      : {p['volatility']:.6f}")
        print(f"  Sharpe         : {p['sharpe_ratio']:.4f}")
        print(f"  Annual Return  : {p['annual_return']:.2%}")
        print(f"  Annual Vol     : {p['annual_volatility']:.2%}")

    print("\n" + "=" * 55)
    print("\n[CAPM]")
    market_ret = mu["SPY"]
    betas = calculate_all_betas(returns, market_column="SPY")
    capm_rets = capm_expected_returns_all(betas, market_ret)
    print(pd.concat([betas, capm_rets], axis=1).to_string())