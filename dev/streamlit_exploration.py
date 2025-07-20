import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

import pandas as pd, yfinance as yf, datetime, plotly.graph_objects as go, plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
from pygooglenews import GoogleNews
import os

# ──────────────────────────────  DATA  ───────────────────────────────────────────
# Get the absolute path to the CSV file relative to this script
current_dir = os.path.dirname(__file__)  # dev/
csv_path = os.path.abspath(os.path.join(current_dir, "..", "data", "S&P500.csv"))
# Read the CSV
df_tickers = pd.read_csv(csv_path, encoding="latin1")

tickers = df_tickers["Symbol"].dropna().unique()
ticker_to_name = dict(zip(df_tickers["Symbol"], df_tickers["Security"]))

# ──────────────────────────────  SIDEBAR  ────────────────────────────────────────
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=[
            "General Information",
            "Individual Information",
            "News",
            "Portfolio Optimizer",
        ],
    )

# ──────────────────────────────  NEWS CACHE  ─────────────────────────────────────
@st.cache_data
def load_news(tkr: str, company: str, days: int = 7) -> pd.DataFrame:
    query = f'{tkr} stock "{company}"'
    gn = GoogleNews()
    search = gn.search(query, when=f"{days}d")

    if not search["entries"]:
        return pd.DataFrame()

    df = pd.DataFrame(search["entries"]).rename(columns={"link": "url"})
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df["publisher"] = df["source"].apply(
        lambda s: s.get("title", "N/A") if isinstance(s, dict) else "N/A"
    )

    final_cols = ["title", "summary", "publisher", "published", "url"]
    for col in final_cols:
        if col not in df.columns:
            df[col] = "N/A"

    return df[final_cols]

# ─────────────────────  INDIVIDUAL INFORMATION  ─────────────────────────────────
if selected == "Individual Information":
    st.title("DETAILED STOCK INFORMATION")

    ticker = st.sidebar.selectbox("Select Ticker", tickers)
    ticker = str(ticker).upper()

    # ------------- helper to flatten MultiIndex from yfinance ------------------
    def extract_single_ticker(df: pd.DataFrame, tkr: str) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Received empty DataFrame from yfinance.")

        if not isinstance(df.columns, pd.MultiIndex):
            return df  # already a single‑ticker frame

        # find which level contains the ticker symbol
        for lvl in range(df.columns.nlevels):
            if tkr in df.columns.get_level_values(lvl):
                return df.xs(tkr, level=lvl, axis=1)

        raise KeyError(f"{tkr} not found in downloaded data.")


    # ------------- today vs yesterday ------------------------------------------
    raw_latest = yf.download(ticker, period="1d", threads=False)
    latest = extract_single_ticker(raw_latest, ticker)

    if latest.empty:
        st.error("No intraday data returned.")
        st.stop()

    today_open, today_close, trading_volume = (
        float(latest["Open"].iloc[0]),
        float(latest["Close"].iloc[0]),
        int(latest["Volume"].iloc[0]),
    )

    raw_hist = yf.download(ticker, period="5d", threads=False)
    hist = extract_single_ticker(raw_hist, ticker)
    if hist.shape[0] < 2:
        st.error("Not enough history for yesterday’s metrics.")
        st.stop()
    yest = hist.iloc[-2]
    y_open, y_close, y_vol = float(yest["Open"]), float(yest["Close"]), int(yest["Volume"])

    # ------------- all‑time high / low ------------------------------------------
    # Replace this block
    raw_all_data = yf.download(ticker, start="2000-01-01", threads=False)
    all_data = extract_single_ticker(raw_all_data, ticker)
    all_data.index = pd.to_datetime(all_data.index)

    a_high, a_low = float(all_data["High"].max()), float(all_data["Low"].min())
    high_date = all_data["High"].idxmax().strftime("%B %d, %Y")
    low_date = all_data["Low"].idxmin().strftime("%B %d, %Y")

    # ------------- KPI cards ----------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Today's Open", f"${today_open:.2f}", f"{(today_open-y_open)/y_open:.2%} vs Yest")
    c2.metric("Today's Close", f"${today_close:.2f}", f"{(today_close-y_close)/y_close:.2%} vs Yest")
    c3.metric("Today's Volume", f"{trading_volume/1_000_000:.2f} M", f"{(trading_volume-y_vol)/y_vol:.2%} vs Yest")
    c4.metric("All‑Time High", f"${a_high:.2f}")
    c4.caption(f"on {high_date}")
    c5.metric("All‑Time Low", f"${a_low:.2f}")
    c5.caption(f"on {low_date}")

    # --- Date range selector ---
    start_date = st.sidebar.date_input("Start date")
    end_date   = st.sidebar.date_input("End date")

    if start_date >= end_date:
        st.error("Start Date must be earlier than End Date")
        st.stop()

    # --- Download & flatten data ---
    raw_user_data = yf.download(ticker, start=start_date, end=end_date, threads=False)
    user_data     = extract_single_ticker(raw_user_data, ticker)

    if user_data.empty or user_data.shape[0] < 2:
        st.warning("Not enough data for that range")
        st.stop()

    # --- Candlestick chart ---
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=user_data.index,
                open=user_data["Open"],
                high=user_data["High"],
                low=user_data["Low"],
                close=user_data["Close"],
            )
        ]
    )
    fig.update_layout(
        title="C A N D L E S T I C K   C H A R T",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Volume bar chart ---
    vol_fig = px.bar(
        user_data,
        x=user_data.index,
        y="Volume",
        title="V O L U M E   &   M O V I N G   A V E R A G E S",
        labels={"index": "Date"},
        height=400,
    )
    st.plotly_chart(vol_fig, use_container_width=True)


# ────────────────────────  GENERAL INFORMATION  ─────────────────────────────────
if selected == "General Information":
    st.title("GENERAL STOCK INFORMATION")

    # 1) Download today’s OHLCV for all tickers
    all_data = yf.download(tickers.tolist(), period="1d", threads=True, progress=False)

    # 2) Extract first (and only) row
    if isinstance(all_data.columns, pd.MultiIndex):
        opens = all_data["Open"].iloc[0]
        closes = all_data["Close"].iloc[0]
        volumes = all_data["Volume"].iloc[0]
    else:  # single‑ticker edge‑case
        opens = pd.Series({tickers[0]: all_data["Open"].iloc[0]})
        closes = pd.Series({tickers[0]: all_data["Close"].iloc[0]})
        volumes = pd.Series({tickers[0]: all_data["Volume"].iloc[0]})

    pct_change = (closes - opens) / opens

    # 3) Build price DataFrame
    df = pd.DataFrame({"pct_change": pct_change, "volume": volumes})
    df.index = df.index.astype(str).str.upper()

    # 4) Prepare GICS metadata
    meta = (
        df_tickers[["Symbol", "GICS Sector", "GICS Sub-Industry"]]
        .assign(Symbol=lambda x: x["Symbol"].astype(str).str.upper())
        .set_index("Symbol")
    )

    # 5) Join and clean
    df = df.join(meta, how="left").dropna(subset=["GICS Sector", "GICS Sub-Industry"])
    if df.empty:
        st.warning("No overlap between price data and GICS metadata.")
        st.stop()

    # 6) Aggregations
    sector_perf = df.groupby("GICS Sector")["pct_change"].mean()
    industry_perf = df.groupby("GICS Sub-Industry")["pct_change"].mean()

    top_sector, bottom_sector = sector_perf.idxmax(), sector_perf.idxmin()
    top_industry, bottom_industry = industry_perf.idxmax(), industry_perf.idxmin()
    total_volume = df["volume"].sum() / 1_000_000  # millions

    # 7) KPI cards
    c1, c2, c3, c4, c5 = st.columns([1, 3, 1, 2, 1])
    c1.metric("Top Sector", top_sector)
    c2.metric("Top Industry", top_industry)
    c3.metric("Bottom Sector", bottom_sector)
    c4.metric("Bottom Industry", bottom_industry)
    c5.metric("Total Volume", f"{total_volume:.2f} M")

# ───────────────────────────────  NEWS  ─────────────────────────────────────────
if selected == "News":
    ticker = st.sidebar.selectbox("Select Ticker", tickers)
    company = ticker_to_name.get(ticker, "")

    st.header(f"Recent News for {ticker}")

    df_news = load_news(ticker, company, days=7)
    if df_news.empty:
        st.info("No recent news.")
        st.stop()

    for _, row in df_news.iloc[:10].iterrows():
        st.subheader(row["title"])
        st.write(f"**{row['publisher']}** | {row['published'].strftime('%B %d, %Y')}")
        st.write(f"[Read More]({row['url']})")
        st.markdown("---")
