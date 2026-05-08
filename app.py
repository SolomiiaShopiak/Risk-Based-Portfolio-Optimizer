from src.preprocessing import *
from src.portfolio_model import *
from src.optimizer import *

from src.preprocessing import load_price_data, clean_price_data, calculate_returns
from src.portfolio_model import calculate_statistics, simulate_portfolios, add_sharpe_ratio
from src.optimizer import get_portfolio_for_risk_profile


st.set_page_config(
    page_title="Risk-Based Portfolio Optimizer",
    layout="wide"
)

# ---------- HEADER ----------
st.title("💰 Risk-Based Portfolio Optimizer")

st.markdown("""
This tool helps users choose an investment portfolio based on their risk profile.

It uses:
- **Modern Portfolio Theory**
- **Sharpe Ratio**
- **Risk-based portfolio optimization**
- **Efficient Frontier visualization**
""")

st.markdown("---")


# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ User Settings")

risk_level = st.sidebar.selectbox(
    "Choose your risk level:",
    ["low", "medium", "high"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload prices CSV file",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.caption("If no file is uploaded, the app uses default prices.csv data.")


# ---------- LOAD DATA ----------
try:
    if uploaded_file is not None:
        prices = pd.read_csv(uploaded_file)

        if "Date" not in prices.columns:
            st.error("Uploaded CSV must contain a Date column.")
            st.stop()

        prices["Date"] = pd.to_datetime(prices["Date"])
        prices = prices.set_index("Date")

    else:
        prices = load_price_data("data/prices.csv")

    clean_prices = clean_price_data(prices)
    returns = calculate_returns(clean_prices)

    # ---------- DATA PREVIEW ----------
    st.subheader("1. Historical Price Data")
    st.write("This table shows the latest historical prices of selected assets.")
    st.dataframe(clean_prices.tail())

    st.subheader("2. Asset Returns")
    st.write("Returns show how much each asset price changed from one day to the next.")
    st.dataframe(returns.tail())

    # ---------- STATISTICS ----------
    mu, cov_matrix = calculate_statistics(returns)

    # ---------- SIMULATION ----------
    results_df, weights_list = simulate_portfolios(
        returns,
        n_portfolios=3000
    )

    results_df = add_sharpe_ratio(results_df)

    # ---------- OPTIMIZATION ----------
    portfolio = get_portfolio_for_risk_profile(
        mu,
        cov_matrix,
        risk_level=risk_level
    )

    # ---------- RECOMMENDED PORTFOLIO ----------
    st.subheader("3. Recommended Portfolio")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Annual Return",
        f"{portfolio['annual_return']:.2%}"
    )

    col2.metric(
        "Annual Volatility",
        f"{portfolio['annual_volatility']:.2%}"
    )

    col3.metric(
        "Sharpe Ratio",
        f"{portfolio['sharpe_ratio']:.2f}"
    )

    st.markdown("""
    **How to read these metrics:**
    - **Annual Return** — expected yearly return of the portfolio.
    - **Annual Volatility** — risk level, or how strongly the portfolio value may fluctuate.
    - **Sharpe Ratio** — return per unit of risk. Higher Sharpe Ratio means better risk-adjusted performance.
    """)

    # ---------- PORTFOLIO ALLOCATION ----------
    st.subheader("4. Portfolio Allocation")

    weights_df = pd.DataFrame(
        list(portfolio["weights"].items()),
        columns=["Asset", "Weight"]
    )

    weights_df["Weight"] = weights_df["Weight"] * 100
    weights_df["Weight"] = weights_df["Weight"].round(2)

    st.dataframe(weights_df)

    fig_pie = px.pie(
        weights_df,
        names="Asset",
        values="Weight",
        title="Recommended Portfolio Weights"
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    # ---------- BEST ASSET ----------
    best_asset = returns.mean().idxmax()
    best_asset_return = returns.mean().max()

    st.info(
        f"🏆 Best average daily performing asset in the dataset: "
        f"**{best_asset}** ({best_asset_return:.4%})"
    )

    # ---------- EFFICIENT FRONTIER ----------
    st.subheader("5. Efficient Frontier")

    st.markdown("""
    The Efficient Frontier shows many possible portfolios.

    - Each dot = one possible portfolio  
    - X-axis = risk / volatility  
    - Y-axis = expected return  
    - Color = Sharpe Ratio  
    - Red dot = selected recommended portfolio  
    """)

    fig = px.scatter(
        results_df,
        x="Risk",
        y="Return",
        color="Sharpe",
        title="Efficient Frontier",
        labels={
            "Risk": "Risk / Volatility",
            "Return": "Expected Return",
            "Sharpe": "Sharpe Ratio"
        }
    )

    fig.add_scatter(
        x=[portfolio["volatility"]],
        y=[portfolio["return"]],
        mode="markers",
        marker=dict(size=16, color="red"),
        name="Selected Portfolio"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------- EXPLANATION ----------
    st.subheader("6. Explanation")

    if risk_level == "low":
        st.success(
            "The system selected a low-risk portfolio by minimizing volatility. "
            "This profile is suitable for conservative users who prefer stability "
            "and lower price fluctuations."
        )

    elif risk_level == "medium":
        st.info(
            "The system selected a medium-risk portfolio by balancing expected return "
            "and risk. This profile is suitable for users who want moderate growth "
            "with controlled volatility."
        )

    else:
        st.warning(
            "The system selected a high-risk portfolio by maximizing the Sharpe Ratio. "
            "This profile is suitable for users who are ready to accept higher volatility "
            "for potentially higher returns."
        )

    # ---------- FINAL SUMMARY ----------
    st.markdown("---")

    st.subheader("7. Final Recommendation")

    st.markdown(f"""
    Based on the selected risk profile **{risk_level.upper()}**, the system recommends a portfolio with:

    - **Expected annual return:** {portfolio['annual_return']:.2%}
    - **Annual volatility:** {portfolio['annual_volatility']:.2%}
    - **Sharpe Ratio:** {portfolio['sharpe_ratio']:.2f}

    This recommendation is generated using historical returns, portfolio risk, and optimization logic.
    """)

    st.caption("Built using Python, Streamlit, pandas, Plotly, Modern Portfolio Theory and Sharpe Ratio.")

except Exception as e:
    st.error(f"Error: {e}")
