import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

AUTHOR = "Jose A. Marchena"
# Fixed color per company so a filter change never repaints the survivors.
COMPANIES = {
    "AAPL": ("Apple", "#2a78d6"),
    "MSFT": ("Microsoft", "#eb6834"),
    "NVDA": ("NVIDIA", "#1baf7a"),
    "GOOGL": ("Alphabet", "#eda100"),
    "AMZN": ("Amazon", "#e87ba4"),
    "META": ("Meta", "#008300"),
    "TSLA": ("Tesla", "#6250d6"),
    "JPM": ("JPMorgan Chase", "#e34948"),
}
COLORS = {t: c for t, (_, c) in COMPANIES.items()}
TRADING_DAYS = 252

st.set_page_config(page_title="Stock Comparison Dashboard", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner="Downloading latest prices from Yahoo Finance…")
def load_prices():
    raw = yf.download(list(COMPANIES), period="5y", auto_adjust=True, progress=False)
    return raw["Close"].dropna(how="all"), raw["Volume"].dropna(how="all")


st.markdown(
    f"""
    <div style="background:#002D72;padding:18px 24px;border-radius:8px;color:#fff;margin-bottom:12px">
      <div style="font-size:1.8rem;font-weight:700">Big Tech &amp; Finance Stock Comparison</div>
      <div style="opacity:.9">By <b>{AUTHOR}</b> &nbsp;·&nbsp; <b>Developed for CPS*5745</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    close_all, volume_all = load_prices()
except Exception as e:  # network / rate-limit failure: show it instead of a stack trace
    st.error(f"Could not download stock data: {e}")
    st.stop()

min_date, max_date = close_all.index.min().date(), close_all.index.max().date()

# Shared filters - one row above every chart.
f1, f2 = st.columns([2, 1])
tickers = f1.multiselect(
    "Companies (pick at least 3)",
    list(COMPANIES),
    default=["AAPL", "MSFT", "NVDA", "GOOGL"],
    format_func=lambda t: f"{COMPANIES[t][0]} ({t})",
)
date_range = f2.date_input(
    "Date range",
    value=(max(min_date, max_date - dt.timedelta(days=365)), max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(tickers) < 3:
    st.warning("Select at least three companies to compare.")
    st.stop()
if not isinstance(date_range, tuple) or len(date_range) != 2:
    st.info("Pick an end date for the range.")
    st.stop()
start, end = (pd.Timestamp(d) for d in date_range)

# Daily returns and rolling volatility use the full history so the first days of the range are valid.
returns_all = close_all[tickers].pct_change()
vol_all = returns_all.rolling(30).std() * np.sqrt(TRADING_DAYS) * 100

in_range = (close_all.index >= start) & (close_all.index <= end)
close = close_all.loc[in_range, tickers].dropna()
if len(close) < 5:
    st.warning("Not enough trading days in the selected range - widen it.")
    st.stop()
rets = returns_all.loc[close.index]
vol = vol_all.loc[close.index]
volume = volume_all.loc[close.index, tickers]

cum_ret = (close / close.iloc[0] - 1) * 100
drawdown = (close / close.cummax() - 1) * 100
years = len(close) / TRADING_DAYS
summary = pd.DataFrame(
    {
        "Company": [COMPANIES[t][0] for t in tickers],
        "Last close ($)": close.iloc[-1],
        "Total return (%)": cum_ret.iloc[-1],
        "Annualized return (%)": ((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1) * 100,
        "Annualized volatility (%)": rets.std() * np.sqrt(TRADING_DAYS) * 100,
        "Max drawdown (%)": drawdown.min(),
        "Avg daily $ volume ($B)": (close * volume).mean() / 1e9,
    }
)
summary["Return / risk"] = summary["Annualized return (%)"] / summary["Annualized volatility (%)"]

st.caption(
    f"Showing {close.index[0]:%b %d, %Y} – {close.index[-1]:%b %d, %Y} "
    f"({len(close)} trading days). Prices are split/dividend-adjusted closes in USD from Yahoo Finance; "
    f"latest data point {max_date:%b %d, %Y}."
)

# Summary indicators (not counted as visualizations).
for col, t in zip(st.columns(len(tickers)), tickers):
    col.metric(
        f"{COMPANIES[t][0]} ({t})",
        f"${summary.loc[t, 'Last close ($)']:,.2f}",
        f"{summary.loc[t, 'Total return (%)']:+.1f}% in range",
    )


def long(df, value):
    return df.reset_index().melt(id_vars="Date", var_name="Ticker", value_name=value)


def line_chart(df, value, title, ytitle):
    fig = px.line(long(df, value), x="Date", y=value, color="Ticker", color_discrete_map=COLORS, title=title)
    fig.update_traces(line_width=2, hovertemplate="%{y:.1f}%")
    fig.update_layout(hovermode="x unified", yaxis_title=ytitle, xaxis_title=None, legend_title=None)
    return fig


c1, c2 = st.columns(2)
fig = line_chart(cum_ret, "Return", "1. Cumulative return since range start", "Return (%)")
fig.add_hline(y=0, line_color="gray", line_width=1)
c1.plotly_chart(fig, width="stretch")
c1.caption("Indexed to 0% on the first day, so companies with very different share prices compare fairly.")

fig = line_chart(drawdown, "Drawdown", "2. Drawdown from running peak", "Below peak (%)")
c2.plotly_chart(fig, width="stretch")
c2.caption("How far each stock sits below its highest close so far in the range (0% = at a new high).")

c3, c4 = st.columns(2)
fig = line_chart(vol, "Volatility", "3. Rolling 30-day volatility (annualized)", "Volatility (%)")
c3.plotly_chart(fig, width="stretch")
c3.caption("Std. dev. of daily returns over the past 30 trading days × √252. Higher = bumpier ride.")

fig = px.box(
    long(rets * 100, "Daily return"), x="Ticker", y="Daily return", color="Ticker",
    color_discrete_map=COLORS, title="4. Distribution of daily returns", points="outliers",
)
fig.update_layout(showlegend=False, yaxis_title="Daily return (%)", xaxis_title=None)
fig.add_hline(y=0, line_color="gray", line_width=1)
c4.plotly_chart(fig, width="stretch")
c4.caption("Box = middle 50% of days, line = median, dots = unusually large moves.")

c5, c6 = st.columns(2)
fig = px.scatter(
    summary.reset_index(names="Ticker"), x="Annualized volatility (%)", y="Annualized return (%)",
    color="Ticker", color_discrete_map=COLORS, text="Ticker", title="5. Risk vs. return",
    hover_data={"Company": True, "Return / risk": ":.2f", "Max drawdown (%)": ":.1f"},
)
fig.update_traces(marker=dict(size=16, line=dict(width=2, color="white")), textposition="top center")
fig.update_layout(showlegend=False)
fig.add_hline(y=0, line_color="gray", line_width=1)
c5.plotly_chart(fig, width="stretch")
c5.caption("Up-and-left is best: more return per unit of risk.")

corr = rets.corr()
fig = go.Figure(
    go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
        colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]],
        text=corr.round(2).values, texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>", colorbar=dict(title="ρ"),
    )
)
fig.update_layout(title="6. Correlation of daily returns", yaxis_autorange="reversed")
c6.plotly_chart(fig, width="stretch")
c6.caption("1 = move in lockstep, 0 = unrelated. Low correlation means better diversification.")

fig = px.bar(
    summary.reset_index(names="Ticker"), x="Ticker", y="Avg daily $ volume ($B)", color="Ticker",
    color_discrete_map=COLORS, title="7. Average daily dollar volume traded", text_auto=".2f",
)
fig.update_traces(textposition="outside", cliponaxis=False)
fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Billions of USD per day")
st.plotly_chart(fig, width="stretch")
st.caption("Shares traded × price, averaged per day - a liquidity / investor-attention comparison in dollars.")

# Key differences - recomputed from whatever the filters select.
name = lambda t: f"{COMPANIES[t][0]} ({t})"
best, worst = summary["Total return (%)"].idxmax(), summary["Total return (%)"].idxmin()
calm, wild = summary["Annualized volatility (%)"].idxmin(), summary["Annualized volatility (%)"].idxmax()
deep = summary["Max drawdown (%)"].idxmin()
eff = summary["Return / risk"].idxmax()
liquid = summary["Avg daily $ volume ($B)"].idxmax()
pairs = corr.where(~np.eye(len(corr), dtype=bool)).stack()
hi_pair, lo_pair = pairs.idxmax(), pairs.idxmin()

st.subheader("Key differences")
st.markdown(
    f"""
- **Performance:** {name(best)} led with **{summary.loc[best, 'Total return (%)']:+.1f}%**, while {name(worst)} trailed at **{summary.loc[worst, 'Total return (%)']:+.1f}%** (chart 1).
- **Risk:** {name(wild)} was the most volatile (**{summary.loc[wild, 'Annualized volatility (%)']:.0f}%** annualized) and {name(calm)} the calmest (**{summary.loc[calm, 'Annualized volatility (%)']:.0f}%**) (charts 3 & 4).
- **Worst decline:** {name(deep)} fell as much as **{-summary.loc[deep, 'Max drawdown (%)']:.1f}%** below its peak (chart 2).
- **Efficiency:** {name(eff)} delivered the most return per unit of risk (ratio **{summary.loc[eff, 'Return / risk']:.2f}**) (chart 5).
- **Co-movement:** {hi_pair[0]} & {hi_pair[1]} moved most alike (ρ = {pairs.max():.2f}); {lo_pair[0]} & {lo_pair[1]} least (ρ = {pairs.min():.2f}) (chart 6).
- **Liquidity:** {name(liquid)} had the most dollars traded per day (**${summary.loc[liquid, 'Avg daily $ volume ($B)']:.1f}B**) (chart 7).
"""
)

with st.expander("Data table"):
    st.dataframe(summary.round(2), width="stretch")

st.markdown("---")
st.caption(f"{AUTHOR} · Developed for CPS*5745 · Data: Yahoo Finance via yfinance (refreshed hourly).")
