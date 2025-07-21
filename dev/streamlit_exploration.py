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
    # ─── Main Navigation ──────────────────────────────────────────────────────
with st.sidebar:
    selected = option_menu(
            menu_title="Navigation",
            options=[
                "General Information",
                "Individual Information",
                "News",
                "Portfolio Optimizer",
            ],
            icons=["bar-chart-line", "list-columns-reverse", "newspaper", "gear"],  # lucide icons
            menu_icon="cast",
            default_index=0,
        )

    st.markdown("---")

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

    with st.sidebar:
        st.header("🔎 Select Parameters")
        ticker = st.selectbox("Ticker", tickers, index= tickers.tolist().index("AAPL"))  # default if you like
        start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
        end_date   = st.date_input("End Date",   value=pd.Timestamp.today())
        rsi_period = st.selectbox("RSI Period (days)", [7,14,30], index=1)
        ma_period  = st.selectbox("Volume Moving Average (days)", [7,14,21,30], index=1)

    company_name = ticker_to_name.get(ticker, ticker)

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
    c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1], gap="small")
    c1.metric("Open", f"${today_open:.2f}", delta=f"{(today_open-y_open)/y_open:.2%}")
    c2.metric("Close", f"${today_close:.2f}", delta=f"{(today_close-y_close)/y_close:.2%}")
    c3.metric("Volume (M)", f"{trading_volume/1e6:.2f}", delta=f"{(trading_volume-y_vol)/y_vol:.2%}")
    c4.metric("All-Time High", f"${a_high:.2f}", help=f"on {high_date}")
    c5.metric("All-Time Low",  f"${a_low:.2f}", help=f"on {low_date}")

    st.markdown("---")
    st.subheader(f"{company_name} — {start_date:%b %d, %Y} to {end_date:%b %d, %Y}")

    if start_date >= end_date:
        st.error("Start Date must be earlier than End Date")
        st.stop()

    # --- Download & flatten data ---
    raw_user_data = yf.download(ticker, start=start_date, end=end_date, threads=False)
    user_data     = extract_single_ticker(raw_user_data, ticker)

    if user_data.empty or user_data.shape[0] < 2:
        st.warning("Not enough data for that range")
        st.stop()

    # compute rolling average
    user_data[f"Vol_MA_{ma_period}"] = (
        user_data["Volume"]
        .rolling(window=ma_period)
        .mean()
    )

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
    

     # --- Volume bar chart + MA line ---
    vol_fig = go.Figure()
    # bars for raw volume
    vol_fig.add_trace(go.Bar(
        x=user_data.index,
        y=user_data["Volume"],
        name="Volume",
        marker_color="lightgrey",
    ))
    # line for moving average
    vol_fig.add_trace(go.Scatter(
        x=user_data.index,
        y=user_data[f"Vol_MA_{ma_period}"],
        mode="lines",
        name=f"{ma_period}-Day MA",
        line=dict(width=2),
    ))

    vol_fig.update_layout(
        title=f"Volume & {ma_period}-Day MA",
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=400
    )

    

    # --- RSI calculation & chart ---
    # 1) compute price changes
    delta = user_data["Close"].diff()

    # 2) separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # 3) Wilder’s smoothing with α = 1/rsi_period
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()

    # 4) compute RSI
    rs = avg_gain / avg_loss
    user_data["RSI"] = 100 - (100 / (1 + rs))

    # 5) plot it
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(
        x=user_data.index,
        y=user_data["RSI"],
        mode="lines",
        name=f"RSI ({rsi_period})"
    ))
    # overbought / oversold lines
    rsi_fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="white",
        annotation_text="Overbought",
        annotation_position="top left",
        annotation_font_color="white"
    )
    rsi_fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="white",
        annotation_text="Oversold",
        annotation_position="bottom left",
        annotation_font_color="white"
    )

    rsi_fig.update_layout(
        title=f"R E L A T I V E   S T R E N G T H   I N D E X",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        height=350,
    )
    base_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12),
        margin=dict(l=40, r=20, t=50, b=30),
    )
    for figs in (fig, vol_fig, rsi_fig):
        figs.update_layout(**base_layout)
    
    tabs = st.tabs(["📈 Price", "📊 Volume", "⚡ RSI"])
    with tabs[0]:
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        st.plotly_chart(vol_fig, use_container_width=True)
    with tabs[2]:
        st.plotly_chart(rsi_fig, use_container_width=True)
        


    # ────────────────────────  GENERAL INFORMATION  ─────────────────────────────────
if selected == "General Information":

    # 1) Page title & sidebar controls
    st.title("GENERAL STOCK INFORMATION")
    with st.sidebar:
        st.header("🔎 Filters & Settings")
        # • How many rows to show in Top/Bottom tables
        top_n = st.slider("Rows to show", 5, 30, 10, step=5)
        # • (Optional) Filter by Sector
        sectors = st.multiselect("Sector filter", options=df_tickers["GICS Sector"].unique(), default=None)
        # • (Optional) Filter by Industry
        industries = st.multiselect("Industry filter", options=df_tickers["GICS Sub-Industry"].unique(), default=None)

    # 2) Download today’s OHLCV for all tickers
    all_data = yf.download(tickers.tolist(), period="1d", threads=True, progress=False)

    # 3) Extract first (and only) row of Open/Close/Volume
    if isinstance(all_data.columns, pd.MultiIndex):
        opens  = all_data["Open"].iloc[0]
        closes = all_data["Close"].iloc[0]
        volumes= all_data["Volume"].iloc[0]
    else:
        opens  = pd.Series({tickers[0]: all_data["Open"].iloc[0]})
        closes = pd.Series({tickers[0]: all_data["Close"].iloc[0]})
        volumes= pd.Series({tickers[0]: all_data["Volume"].iloc[0]})

    # 4) Build base DataFrame and apply filters
    df = pd.DataFrame({"pct_change": (closes - opens) / opens, "volume": volumes})
    df.index = df.index.astype(str).str.upper()
    # 4a) Join GICS metadata
    meta = (df_tickers[["Symbol","GICS Sector","GICS Sub-Industry"]]
              .assign(Symbol=lambda x: x["Symbol"].str.upper())
              .set_index("Symbol"))
    df = df.join(meta, how="left").dropna(subset=["GICS Sector","GICS Sub-Industry"])
    # 4b) Apply sidebar filters if any
    if sectors:
        df = df[df["GICS Sector"].isin(sectors)]
    if industries:
        df = df[df["GICS Sub-Industry"].isin(industries)]
    if df.empty:
        st.warning("No data after applying filters.")
        st.stop()

    # 5) Aggregations for metrics
    sector_perf   = df.groupby("GICS Sector")["pct_change"].mean()
    industry_perf = df.groupby("GICS Sub-Industry")["pct_change"].mean()
    top_sector    = sector_perf.idxmax()
    bottom_sector = sector_perf.idxmin()
    top_industry  = industry_perf.idxmax()
    bottom_industry=industry_perf.idxmin()
    total_volume  = df["volume"].sum() / 1_000_000  # in millions

    # 6) KPI cards (“At a glance”)
    row1, row2, row3 = st.columns([1,1,2], gap="small")
    with row1:
        st.metric("🏆 Top Sector",   top_sector,    f"{sector_perf[top_sector]:.2%}")
        st.metric("🥇 Top Industry", top_industry,  f"{industry_perf[top_industry]:.2%}")
    with row2:
        st.metric("🏴 Bottom Sector", bottom_sector, f"{sector_perf[bottom_sector]:.2%}")
        st.metric("🥈 Bottom Industry",bottom_industry, f"{industry_perf[bottom_industry]:.2%}")
    with row3:
        st.metric("📊 Total Volume (M)", f"{total_volume:.2f}")

    # 7) Optional: Bar chart of sector performance
    fig_sec = px.bar(
        sector_perf.sort_values(ascending=False).reset_index(),
        x="GICS Sector", y="pct_change",
        title="Sector % Change Today",
        labels={"pct_change": "% Change"},
    )


    fig_sec.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_tickformat=".1%",    # <-- display axis in percent (e.g. 1.2%)
        yaxis_title="% Change"
    )

    st.plotly_chart(fig_sec, use_container_width=True, height=300)

    # ─── 8) Top/Bottom N performers tables ─────────────────────────────────────
    # 8a) Prepare tables
    perf = df.assign(Price=closes)[["Price","pct_change"]].rename(columns={"pct_change":"% Change"})
    topN    = perf.sort_values("% Change", ascending=False).head(top_n)
    bottomN = perf.sort_values("% Change", ascending=True).head(top_n)

    # 8b) Render in Expanders side-by-side
    exp1, exp2 = st.columns(2)
    with exp1:
        with st.expander(f"📈 Top {top_n} Gainers"):
            styled = (topN.style
                        .format({"Price":"${:,.2f}","% Change":"{:.2%}"})
                        .applymap(lambda v: "color: green", subset=["% Change"]))
            st.dataframe(styled, use_container_width=True)
    with exp2:
        with st.expander(f"📉 Top {top_n} Losers"):
            styled = (bottomN.style
                        .format({"Price":"${:,.2f}","% Change":"{:.2%}"})
                        .applymap(lambda v: "color: red", subset=["% Change"]))
            st.dataframe(styled, use_container_width=True)

# ───────────────────────────────  NEWS  ─────────────────────────────────────────
if selected == "News":
    # Sidebar controls
    with st.sidebar:
        st.header("🔍 News Settings")
        ticker = st.selectbox("Ticker", tickers, index=0)
        days = st.number_input(
            "Fetch news from last…", min_value=1, max_value=90, value=7, step=1
        )
        max_articles = st.slider("Max articles to show", 5, 20, 10, step=5)
        keyword_filter = st.text_input("Filter by keyword", "")

    company = ticker_to_name.get(ticker, "")
    st.header(f"📰 Recent News for {ticker} (last {days} days)")

    # Fetch news with spinner
    with st.spinner(f"Loading news for {ticker}…"):
        df_news = load_news(ticker, company, days=days)

    if df_news.empty:
        st.info("No recent news found. Try increasing the date range or changing ticker.")
        st.stop()

    # Optional keyword filtering
    if keyword_filter:
        df_news = df_news[df_news["title"]
                          .str.contains(keyword_filter, case=False, na=False)]
        if df_news.empty:
            st.warning(f"No articles found containing '{keyword_filter}'.")
            st.stop()


    # Display top N articles
    for i, row in df_news.head(max_articles).iterrows():
        # Clickable title
        st.markdown(f"### [{row['title']}]({row['url']})")
        # Publisher & date metadata
        st.markdown(
            f"**{row['publisher']}** · {row['published'].strftime('%b %d, %Y')}",
            unsafe_allow_html=True,
        )

    # Download button
    st.download_button(
        "Download News as CSV",
        df_news.to_csv(index=False),
        file_name=f"{ticker}_news.csv",
        mime="text/csv",
    )