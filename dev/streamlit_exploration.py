import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
from pygooglenews import GoogleNews
import os
import boto3
import json
from io import StringIO
from textblob import TextBlob



# ──────────────────────────────  DATA  ───────────────────────────────────────────
def load_aws_credentials():
    """
    Load AWS keys from environment or ../private/config.json
    """
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if access_key and secret_key:
        return access_key, secret_key

    # fallback to config file
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "private", "config.json")
    )
    if os.path.exists(config_path):
        with open(config_path) as f:
            keys = json.load(f)
        return keys["aws_access_key_id"], keys["aws_secret_access_key"]

    st.error(
        "AWS credentials not found. "
        "Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY as environment variables "
        "or place a config.json under ../private/config.json"
    )
    st.stop()

@st.cache_data
def load_stock_data_from_s3(bucket_name: str, s3_key: str) -> pd.DataFrame:
    """
    Fetches the long-format OHLCV CSV from S3 and returns a DataFrame
    with columns [Date, Type, Ticker, Price].
    """
    access_key, secret_key = load_aws_credentials()
    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
    data = obj["Body"].read().decode("utf-8")
    df = pd.read_csv(StringIO(data), parse_dates=["Date"])
    return df

# ─────────────────────────────  TICKER METADATA  ───────────────────────────────
current_dir = os.path.dirname(__file__)
csv_path   = os.path.abspath(os.path.join(current_dir, "..", "data", "S&P500.csv"))
df_tickers = pd.read_csv(csv_path, encoding="latin1")
tickers    = df_tickers["Symbol"].dropna().astype(str).tolist()
ticker_to_name = dict(zip(df_tickers["Symbol"], df_tickers["Security"]))

# ─────────────────────── S3 SNAPSHOT CONFIG ──────────────────────────────────
bucket_name = "yfinancestockdata"
s3_key      = "snapshots/stock_data_2023-01-01_to_2025-06-29.csv"

# ──────────────────────────────  SIDEBAR  ────────────────────────────────────────

with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["General Information", "Individual Information", "News", "Portfolio Optimizer"],
        icons=["bar-chart-line", "list-columns-reverse", "newspaper", "gear"],
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

    # 1) Load full history so we know our date bounds
    full_df  = load_stock_data_from_s3(bucket_name, s3_key)
    min_date = full_df["Date"].min().date()
    max_date = full_df["Date"].max().date()

    # 2) Sidebar inputs for ticker, date range, and periods
    with st.sidebar:
        st.header("🔎 Select Parameters")
        ticker      = st.selectbox("Ticker", tickers, index=tickers.index("AAPL"))
        start_date  = st.date_input("Start Date", value=min_date,  min_value=min_date,  max_value=max_date)
        end_date    = st.date_input("End Date",   value=max_date,  min_value=min_date,  max_value=max_date)
        ma_period   = st.selectbox("Volume MA (days)", [7, 14, 21, 30], index=1)
        rsi_period  = st.selectbox("RSI Period (days)",   [7, 14, 30],       index=1)

    # 3) Pivot your S3 data into an OHLCV DataFrame for this ticker
    df_tkr = (
        full_df
        .query("Ticker == @ticker")
        .pivot(index="Date", columns="Type", values="Price")
        .sort_index()
    )

    # extra metadata
    company_name = ticker_to_name.get(ticker, ticker)
    industry_arr = df_tickers.loc[df_tickers["Symbol"] == ticker, "GICS Sub-Industry"].values
    industry     = industry_arr[0] if len(industry_arr) else "N/A"

    # 4) Compute summary metrics
    latest           = df_tkr.iloc[-1]
    yest             = df_tkr.iloc[-2]
    today_open       = float(latest["Open"])
    today_close      = float(latest["Close"])
    trading_volume   = int(latest["Volume"])
    y_open, y_close, y_vol = float(yest["Open"]), float(yest["Close"]), int(yest["Volume"])
    a_high           = float(df_tkr["High"].max())
    a_low            = float(df_tkr["Low"].min())
    high_date        = df_tkr["High"].idxmax().strftime("%B %d, %Y")
    low_date         = df_tkr["Low"].idxmin().strftime("%B %d, %Y")

    # 5) KPI cards
    c_company, c_industry, c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1,1,1], gap="large")
    with c_company:
        st.metric("Company", company_name)
    with c_industry:
        st.metric("Industry", industry)
    with c1:
        st.metric("Open", f"${today_open:.2f}", delta=f"{(today_open-y_open)/y_open:.2%}")
    with c2:
        st.metric("Close", f"${today_close:.2f}", delta=f"{(today_close-y_close)/y_close:.2%}")
    with c3:
        st.metric("Volume (M)", f"{trading_volume/1e6:.2f}", delta=f"{(trading_volume-y_vol)/y_vol:.2%}")
    with c4:
        st.metric("High", f"${a_high:.2f}", help=f"on {high_date}")
    with c5:
        st.metric("Low", f"${a_low:.2f}",  help=f"on {low_date}")

    st.caption(f"Data last updated: {datetime.datetime.now():%b %d, %Y – %H:%M:%S}")
    st.markdown("---")
    st.subheader(f"{company_name} — {start_date:%b %d, %Y} to {end_date:%b %d, %Y}")

    if start_date >= end_date:
        st.error("Start Date must be earlier than End Date")
        st.stop()

    # 6) Slice user_data for charts
    user_data = df_tkr.loc[start_date:end_date]

    # Shared layout settings
    base_layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=12),
        margin=dict(l=40, r=20, t=50, b=30),
    )

            # ────────── Similar / Opposite Tickers (side by side) ──────────
    # Build a wide Close‐price DataFrame
    close_wide = (
        full_df[full_df["Type"] == "Close"]
        .pivot(index="Date", columns="Ticker", values="Price")
        .sort_index()
    )

    # Correlate selected ticker against all others
    target = close_wide[ticker]
    corr   = close_wide.corrwith(target).dropna()
    corr   = corr.drop(labels=[ticker], errors="ignore")

    # Top 10 most alike and most opposite
    similar  = corr.nlargest(10).index.tolist()
    opposite = corr.nsmallest(10).index.tolist()

    # Badge CSS
    badge_style = (
        "display:inline-block;"
        "background-color:{bg};"
        "color:#fff;"
        "border-radius:4px;"
        "padding:4px 8px;"
        "margin:2px;"
        "font-size:0.9em;"
    )

    # Two side-by-side columns
    col_sim, col_opp = st.columns(2, gap="large")

    with col_sim:
        st.subheader("🔀 Similar Tickers")
        st.markdown(
            "".join(
                f"<span style=\"{badge_style.format(bg='#4CAF50')}\">{t}</span>"
                for t in similar
            ),
            unsafe_allow_html=True
        )

    with col_opp:
        st.subheader("↔️ Opposite Tickers")
        st.markdown(
            "".join(
                f"<span style=\"{badge_style.format(bg='#E74C3C')}\">{t}</span>"
                for t in opposite
            ),
            unsafe_allow_html=True
        )

    # ─────────────────── Candlestick Chart (full width) ───────────────────
    price_fig = go.Figure(data=[go.Candlestick(
        x=user_data.index,
        open=user_data["Open"],
        high=user_data["High"],
        low=user_data["Low"],
        close=user_data["Close"],
    )])
    st.subheader("Candlestick Chart")
    price_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        height=550,
        **base_layout
    )
    st.plotly_chart(price_fig, use_container_width=True)

    # ─────────────────── Prepare RSI & Volume+MA ───────────────────
    # Compute Volume MA
    user_data[f"Vol_MA_{ma_period}"] = user_data["Volume"].rolling(window=ma_period).mean()

    # Volume + MA chart
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Bar(
        x=user_data.index,
        y=user_data["Volume"],
        name="Volume",
        marker_color="lightgrey",
    ))
    vol_fig.add_trace(go.Scatter(
        x=user_data.index,
        y=user_data[f"Vol_MA_{ma_period}"],
        mode="lines",
        name=f"{ma_period}-Day MA",
        line=dict(width=2),
    ))
    vol_fig.update_layout(
        title=f"V O L U M E   &   {ma_period} - D A Y   M O V I N G   A V E R A G E",
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=550,
        **base_layout
    )

    # Compute RSI
    delta    = user_data["Close"].diff()
    gain     = delta.where(delta > 0, 0)
    loss     = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs       = avg_gain / avg_loss
    user_data["RSI"] = 100 - (100 / (1 + rs))

    # RSI chart
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(
        x=user_data.index,
        y=user_data["RSI"],
        mode="lines",
        name=f"RSI ({rsi_period})"
    ))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="white",
                      annotation_text="Overbought", annotation_position="top left",
                      annotation_font_color="white")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="white",
                      annotation_text="Oversold", annotation_position="bottom left",
                      annotation_font_color="white")
    rsi_fig.update_layout(
        title="R E L A T I V E   S T R E N G T H   I N D E X",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        height=550,
        **base_layout
    )

    # ─────────────────── Display RSI & Volume side by side ───────────────────
    col_rsi, col_vol = st.columns(2, gap="large")
    with col_rsi:
        st.plotly_chart(rsi_fig, use_container_width=True)
    with col_vol:
        st.plotly_chart(vol_fig, use_container_width=True)





# ────────────────────────  GENERAL INFORMATION  ─────────────────────────────────
if selected == "General Information":
    st.title("GENERAL STOCK INFORMATION")

    # ──────────── LOAD & PIVOT ────────────
    full_df = load_stock_data_from_s3(bucket_name, s3_key)
    wide = (
        full_df
        .pivot(index="Date", columns=["Ticker","Type"], values="Price")
        .sort_index()
    )
    last_dt   = wide.index.max()
    today     = wide.loc[last_dt]
    yesterday = wide.iloc[-2]

    closes      = today.xs("Close",  level="Type")
    prev_closes = yesterday.xs("Close", level="Type")
    volumes     = today.xs("Volume", level="Type")

    # ─── BUILD METRIC DF (close→close) ─────────────────────────────
    df = pd.DataFrame({
        "pct_change": (closes - prev_closes) / prev_closes,
        "volume":      volumes
    })
    df.index = df.index.astype(str).str.upper()

    # Join in sector/industry metadata
    meta = (
        df_tickers[["Symbol","GICS Sector","GICS Sub-Industry"]]
        .assign(Symbol=lambda x: x["Symbol"].str.upper())
        .set_index("Symbol")
    )
    df = df.join(meta, how="left").dropna(subset=["GICS Sector","GICS Sub-Industry"])

    # ─────────────────── BUILD SECTOR & INDUSTRY INDEX SERIES ───────────────────
    close_wide = (
        full_df[full_df["Type"] == "Close"]
        .pivot(index="Date", columns="Ticker", values="Price")
        .sort_index()
    )
    sector_map    = df_tickers.set_index("Symbol")["GICS Sector"].to_dict()
    subind_map    = df_tickers.set_index("Symbol")["GICS Sub-Industry"].to_dict()
    sector_index  = close_wide.groupby(sector_map, axis=1).mean()
    subind_index  = close_wide.groupby(subind_map, axis=1).mean()

    # overnight % change of those indices
    sector_pct_idx = (sector_index.iloc[-1] - sector_index.iloc[-2]) / sector_index.iloc[-2]
    subind_pct_idx = (subind_index.iloc[-1] - subind_index.iloc[-2]) / subind_index.iloc[-2]

    # ─────────────────── AGGREGATIONS & KPI CARDS ─────────────────────────────
    top_sector      = sector_pct_idx.idxmax()
    bottom_sector   = sector_pct_idx.idxmin()
    top_industry    = subind_pct_idx.idxmax()
    bottom_industry = subind_pct_idx.idxmin()
    total_volume    = df["volume"].sum() / 1_000_000  # millions

    row1, row2, row3 = st.columns([3,3,1], gap="large")
    with row1:
        st.metric("🏆 Top Sector",    top_sector,       f"{sector_pct_idx[top_sector]:.2%}")
        st.metric("🥇 Top Industry",  top_industry,     f"{subind_pct_idx[top_industry]:.2%}")
    with row2:
        st.metric("🏴 Bottom Sector", bottom_sector,    f"{sector_pct_idx[bottom_sector]:.2%}")
        st.metric("🥈 Bottom Industry", bottom_industry,f"{subind_pct_idx[bottom_industry]:.2%}")
    with row3:
        st.metric("📊 Total Volume (M)", f"{total_volume:.2f}")
    st.caption(f"Data last updated: {datetime.datetime.now():%b %d, %Y – %H:%M:%S}")

    st.markdown("---")

    # ─────────────────── TOP / BOTTOM TABLES ───────────────────────────────────
    top_n = 10
    perf = (
        df
        .assign(Price=closes)
        [["Price","pct_change"]]
        .rename(columns={"pct_change":"% Change"})
    )
    topN    = perf.sort_values("% Change", ascending=False).head(top_n)
    bottomN = perf.sort_values("% Change", ascending=True).head(top_n)

    cols = st.columns(2, gap="large")
    with cols[0]:
        st.subheader(f"📈 Top {top_n} Gainers")
        styled_top = (
            topN.style
                .format({"Price":"${:,.2f}","% Change":"{:.2%}"})
                .applymap(lambda v: "color: green", subset=["% Change"])
        )
        st.dataframe(styled_top, use_container_width=True)

    with cols[1]:
        st.subheader(f"📉 Top {top_n} Losers")
        styled_bot = (
            bottomN.style
                .format({"Price":"${:,.2f}","% Change":"{:.2%}"})
                .applymap(lambda v: "color: red", subset=["% Change"])
        )
        st.dataframe(styled_bot, use_container_width=True)
    
    st.markdown("---")

    # ─────────────────── SECTOR ROTATION & TODAY’S PERFORMANCE ───────────────────
    # now lay out the two charts side by side
    col1, col2 = st.columns(2, gap="large")

    # — Left: rolling‐window line chart —
    with col1:
        # nest a row for the title + selector
        title_col, select_col = st.columns([4,1], gap="small")
        with title_col:
            st.subheader("Sector Rotations")
        with select_col:
            label = st.selectbox(
            "",
            ["7 days", "15 days", "30 days", "60 days", "120 days"],
            index=2,
            help="Show % change over the last N days"
        )
        # pull the numeric value back out
        window = int(label.split()[0])
        # compute rolling pct change
        sector_pct = sector_index.pct_change(periods=window) * 100
        start_date = sector_pct.index.max() - pd.Timedelta(days=window)
        sector_pct = sector_pct.loc[start_date:]

        fig1 = px.line(
            sector_pct,
            x=sector_pct.index,
            y=sector_pct.columns,
            labels={"value":"% Change", "Date":"Date"}
        )
        fig1.update_layout(
        margin=dict(t=40, l=0, r=0, b=0),
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
        ),
        showlegend=True
        )
        st.plotly_chart(fig1, use_container_width=True)

    # — Right: today’s bar chart —
    with col2:
        st.subheader("Sector Performance Today")
        bar_df = (
            sector_pct_idx
            .rename_axis("Sector")
            .reset_index(name="Pct Change")
            .sort_values("Pct Change", ascending=False)  # best at top
        )
        mx = bar_df["Pct Change"].abs().max()

        fig2 = px.bar(
            bar_df,
            x="Pct Change",
            y="Sector",
            orientation="h",
            color="Pct Change",
            range_color=[-mx, mx],
            color_continuous_scale=["#b22222","lightgray","#228b22"],
            title="Today’s % Change by Sector",
        )
        # hide colorbar, sort categories, format axis as %
        fig2.update_coloraxes(showscale=False)
        fig2.update_layout(
            yaxis={"categoryorder":"total ascending"},
            margin=dict(t=40, l=0, r=0, b=0),
            showlegend=False
        )
        fig2.update_xaxes(tickformat=".1%")  # display e.g. 1.5% instead of 0.015

        st.plotly_chart(fig2, use_container_width=True)


        
# ───────────────────────────────  NEWS  ─────────────────────────────────────────
if selected == "News":
    # --- Configuration ---
    POSITIVE_THRESHOLD = 0.1
    NEGATIVE_THRESHOLD = -0.1

    # Sidebar controls
    with st.sidebar:
        st.header("🔍 News Settings")
        ticker         = st.selectbox("Ticker", tickers, index=tickers.index("AAPL"))
        days           = st.number_input("Fetch news from last… (days)", min_value=1, max_value=90, value=7, step=1)
        keyword_filter = st.text_input("Filter by keyword (optional)", "")

    company = ticker_to_name.get(ticker, "")
    st.header(f"📰 Recent News for {ticker} (last {days} days)")

    # Fetch & filter
    with st.spinner(f"Loading news for {ticker}…"):
        df_news = load_news(ticker, company, days=days)
    if df_news.empty:
        st.info("No recent news found. Try increasing the date range or changing the ticker.")
        st.stop()
    if keyword_filter:
        df_news = df_news[
            df_news["title"].str.contains(keyword_filter, case=False, na=False) |
            df_news["summary"].str.contains(keyword_filter, case=False, na=False)
        ]
        if df_news.empty:
            st.warning(f"No articles found containing '{keyword_filter}'.")
            st.stop()

    # We will analyze and display the top 10 articles
    df_display = df_news.head(10).copy()

    # ───────── SENTIMENT ANALYSIS ─────────────────
    df_display["text_for_analysis"] = (
        df_display["title"].astype(str) + ". " +
        df_display["summary"].astype(str)
    )
    sentiments = df_display["text_for_analysis"].apply(lambda txt: TextBlob(txt).sentiment)
    df_display["polarity"]     = sentiments.apply(lambda s: s.polarity)
    avg_polarity = df_display["polarity"].mean()

    # Decide overall feeling based on average polarity
    if avg_polarity > POSITIVE_THRESHOLD:
        feeling = "Positive"
        advice_fn = st.success
        advice    = f"✅ The average sentiment of the top {len(df_display)} articles is Positive."
    elif avg_polarity < NEGATIVE_THRESHOLD:
        feeling = "Negative"
        advice_fn = st.error
        advice    = f"⚠️ The average sentiment of the top {len(df_display)} articles is Negative."
    else:
        feeling = "Neutral"
        advice_fn = st.warning
        advice    = f"ℹ️ The average sentiment of the top {len(df_display)} articles is Neutral."

    # --- Display Metrics and Advice ---
    m1, m2, m3 = st.columns([1, 1, 2], gap="medium")
    m1.metric("Avg. Polarity",      f"{avg_polarity:.3f}", help="Sentiment score from -1 to +1")
    m2.metric("Overall Feeling",    feeling)
    with m3:
        advice_fn(advice)
    st.info(
        "💡 **Disclaimer:** This sentiment analysis is automated and for informational purposes only.",
        icon="ℹ️"
    )
    st.markdown("---")

    # --- Display Articles (title, source, date only) ---
    st.subheader(f"Top {len(df_display)} Articles")
    for _, row in df_display.iterrows():
        st.markdown(f"##### [{row['title']}]({row['url']})")
        st.caption(
            f"**Publisher:** {row['publisher']}   |   "
            f"**Published:** {row['published'].strftime('%b %d, %Y')}"
        )
        st.markdown("")  # small spacer

    # Download button for the full fetched list
    st.download_button(
        "Download All News as CSV",
        df_news.to_csv(index=False).encode('utf-8'),
        file_name=f"{ticker}_news_{datetime.date.today()}.csv",
        mime="text/csv",
    )

