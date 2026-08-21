from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# 1. 頁面設定
st.set_page_config(page_title="台股亞當理論分析", layout="wide")

st.title("📈 台股亞當理論 (Adam Theory) 順勢映射引擎")
st.caption(
    "「不預測市場，只順應趨勢」— 自動偵測近期的顯著轉折波段，進行雙軸二次映射。"
)

# 2. 參數設定
st.sidebar.header("⚙️ 參數設定")
stock_id = st.sidebar.text_input(
    "股票代碼 (台股請加 .TW 或 .TWO)", value="2344.TW"
).upper()

default_start = pd.Timestamp.now() - pd.DateOffset(years=1)
start_date = st.sidebar.date_input("歷史資料開始日期", default_start)
end_date = st.sidebar.date_input("歷史資料結束日期", pd.Timestamp.now())

# 模式選擇：自動偵測關鍵波段 vs 手動指定天數
auto_mode = st.sidebar.toggle("🤖 自動偵測最近顯著波段 (推薦)", value=True)

if not auto_mode:
    reflection_days = st.sidebar.slider(
        "手動觀察區間 (反射天數)",
        min_value=10,
        max_value=90,
        value=30,
        step=5,
    )


# 3. 快取資料抓取
@st.cache_data(ttl=3600)
def load_stock_data(symbol, start, end):
    fetch_end = pd.to_datetime(end) + pd.Timedelta(days=1)
    df = yf.download(symbol, start=start, end=fetch_end, progress=False)

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()


df = load_stock_data(stock_id, start_date, end_date)

if df.empty or len(df) < 30:
    st.error("❌ 無法取得數據或數據長度不足。請檢查股票代碼與日期範圍。")
else:
    # --- [關鍵優化] 自動計算近期的顯著代表性波段 ---
    if auto_mode:
        # 取最近 120 個交易日進行檢視
        lookback_df = df.tail(120).copy()

        # 找尋最近的波段高點與低點位置
        max_idx = lookback_df["Close"].idxmax()
        min_idx = lookback_df["Close"].idxmin()

        # 計算高點與低點離「最新日期」的交易天數距離
        days_from_max = len(lookback_df.loc[max_idx:])
        days_from_min = len(lookback_df.loc[min_idx:])

        # 優先挑選離當前最近的極值轉折點；若天數相近，則挑選變動幅度最大的點
        # 保底至少需要 10 天，最多不超過 90 天
        candidates = []
        if 10 <= days_from_max <= 90:
            volatility_max = abs(
                (lookback_df["Close"].iloc[-1] - lookback_df.loc[max_idx, "Close"])
                / lookback_df.loc[max_idx, "Close"]
            )
            candidates.append((days_from_max, volatility_max, "近期顯著高點"))

        if 10 <= days_from_min <= 90:
            volatility_min = abs(
                (lookback_df["Close"].iloc[-1] - lookback_df.loc[min_idx, "Close"])
                / lookback_df.loc[min_idx, "Close"]
            )
            candidates.append((days_from_min, volatility_min, "近期顯著低點"))

        if candidates:
            # 排序標準：變動幅度大 + 離當前時間較近
            candidates.sort(key=lambda x: (x[1], -x[0]), reverse=True)
            reflection_days = candidates[0][0]
            feature_label = candidates[0][2]
        else:
            reflection_days = 25  # 預設值
            feature_label = "近期趨勢區間"

        st.info(
            f"🔍 **系統已自動偵測**：採樣自 **{feature_label}**（追蹤近 **{reflection_days}** 個交易日的代表性強勢走勢）"
        )

    # 4. 亞當二次映射邏輯
    last_date = df.index[-1]
    last_close = float(df["Close"].iloc[-1])

    sub_df = df.tail(reflection_days).copy()
    hist_closes = sub_df["Close"].to_numpy()[::-1]

    # 時間與價格雙軸翻轉 (亞當二次映射核心)
    future_close = last_close + (last_close - hist_closes[1:])

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(future_close) * 2,
        freq="B",
    )[: len(future_close)]

    df_proj = pd.DataFrame({"Adam_Close": future_close}, index=future_dates)

    # 5. 指標與建議
    target_price = future_close[-1]
    pct_change = ((target_price - last_close) / last_close) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("最新收盤價", f"{last_close:.2f} TWD")
    col2.metric(
        f"亞當 {len(future_close)} 日預測目標價",
        f"{target_price:.2f} TWD",
        f"{pct_change:+.2f}%",
    )
    col3.metric("反射採樣天數", f"{reflection_days} 天")

    if pct_change > 5:
        st.success("🚀 **亞當趨勢訊號：強烈順勢多頭**｜映射顯示將發動強勢主升段，建議順勢控管停損做多。")
    elif pct_change < -5:
        st.error("📉 **亞當趨勢訊號：強烈順勢空頭**｜映射顯示為轉弱主跌段，建議保守避險或尋找順勢空點。")
    else:
        st.info("⚖️ **亞當趨勢訊號：箱型震盪/觀望**｜映射無明顯單向動能，建議等待突破。")

    # 6. Matplotlib 繪圖
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # 繪製真實歷史價格
    ax.plot(
        df.index,
        df["Close"],
        label=f"{stock_id} 歷史價格",
        color="#2c3e50",
        linewidth=1.2,
    )

    # 標示自動採樣的代表性波段區間 (黃橘色醒目線條)
    ax.plot(
        sub_df.index,
        sub_df["Close"],
        color="#ff9800",
        linewidth=2.5,
        label=f"代表性波段區間 ({reflection_days}D)",
    )

    # 繪製未來映射線
    ax.plot(
        df_proj.index,
        df_proj["Adam_Close"],
        label="亞當二次映射預測軌跡",
        color="#9c27b0",
        linestyle="--",
        linewidth=2,
        marker="o",
        markersize=3.5,
    )

    ax.axvline(x=last_date, color="gray", linestyle=":", label="當前時間點")
    ax.set_title(f"Adam Theory Reflection for {stock_id} (Dynamic Swing)")
    ax.set_ylabel("Price (TWD)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    plt.xticks(rotation=45)

    st.pyplot(fig)

    # 7. 明細數據
    with st.expander("📊 查看詳細映射數據列表"):
        st.dataframe(
            df_proj.style.format("{:.2f}"), use_container_width=True
        )
