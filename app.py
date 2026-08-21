from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# 1. 頁面基礎設定
st.set_page_config(
    page_title="台股亞當理論視覺化分析引擎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 台股亞當理論 (Adam Theory) 順勢映射引擎")
st.caption(
    "「不預測市場，只順應趨勢」— 透過雙軸二次映射，繪製未來價格與波動區間的可能軌跡。"
)

# 2. 側邊欄參數設定
st.sidebar.header("⚙️ 參數設定")
stock_id = st.sidebar.text_input(
    "股票代碼 (台股請加 .TW 或 .TWO)", value="2344.TW"
).upper()

# 日期選擇
default_start = pd.Timestamp.now() - pd.DateOffset(years=1)
start_date = st.sidebar.date_input("歷史資料開始日期", default_start)
end_date = st.sidebar.date_input("歷史資料結束日期", pd.Timestamp.now())

# 亞當映射參數
reflection_days = st.sidebar.slider(
    "觀察區間 (反射天數)",
    min_value=10,
    max_value=90,
    value=30,
    step=5,
    help="亞當理論建議使用近期顯著的波段天數進行翻轉映射",
)


# 3. 快取的資料抓取函式
@st.cache_data(ttl=3600)
def load_stock_data(symbol, start, end):
    fetch_end = pd.to_datetime(end) + pd.Timedelta(days=1)
    df = yf.download(symbol, start=start, end=fetch_end, progress=False)

    if df.empty:
        return pd.DataFrame()

    # 處理 yfinance MultiIndex 欄位結構
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 清理缺失值
    df = df.dropna()
    return df


# 載入資料
df = load_stock_data(stock_id, start_date, end_date)

if df.empty or len(df) < reflection_days:
    st.error(
        "❌ 無法取得數據或數據長度不足。請檢查股票代碼（如 2330.TW / 6274.TWO）與日期範圍。"
    )
else:
    # 4. 亞當二次映射邏輯計算
    last_date = df.index[-1]
    last_close = float(df["Close"].iloc[-1])

    # 擷取用於映射的歷史片段
    sub_df = df.tail(reflection_days).copy()

    # 時間與價格雙軸翻轉核心公式：
    # Future_Price(t) = Last_Close + (Last_Close - Historical_Price_Reversed(t))
    hist_closes = sub_df["Close"].to_numpy()[::-1]
    hist_highs = sub_df["High"].to_numpy()[::-1]
    hist_lows = sub_df["Low"].to_numpy()[::-1]

    # 排除當日點 (Index 0)，從次日開始延伸
    future_close = last_close + (last_close - hist_closes[1:])
    # 高低點互換翻轉：原 High 翻轉後變為未來的相對 Low 邊界
    future_high = last_close + (last_close - hist_lows[1:])
    future_low = last_close + (last_close - hist_highs[1:])

    # 產生未來的交易日 (估計避開週末)
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(future_close) * 2,
        freq="B",
    )[: len(future_close)]

    # 組成預測 DataFrame
    df_proj = pd.DataFrame(
        {
            "Adam_Close": future_close,
            "Adam_High": future_high,
            "Adam_Low": future_low,
        },
        index=future_dates,
    )

    # 5. 結果關鍵指標與趨勢評估
    target_price = future_close[-1]
    price_change = target_price - last_close
    pct_change = (price_change / last_close) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("最新收盤價", f"{last_close:.2f} TWD")
    col2.metric(
        f"亞當 {len(future_close)} 日預測目標價",
        f"{target_price:.2f} TWD",
        f"{pct_change:+.2f}%",
    )
    col3.metric(
        "預測波段最高 / 最低",
        f"{df_proj['Adam_High'].max():.2f} / {df_proj['Adam_Low'].min():.2f}",
    )

    # 趨勢建議診斷
    if pct_change > 5:
        st.success(
            "🚀 **亞當趨勢訊號：強烈順勢多頭**｜二次映射呈現強勢噴發形態，建議順勢控管停損做多。"
        )
    elif pct_change < -5:
        st.error(
            "📉 **亞當趨勢訊號：強烈順勢空頭**｜二次映射呈現快速崩跌形態，建議保守避險或順勢尋找放空機會。"
        )
    else:
        st.info(
            "⚖️ **亞當趨勢訊號：箱型震盪/觀望**｜映射無明顯方向性，價格可能維持盤整，建議等待突破。"
        )

    # 6. 繪製專業動態圖表 (Plotly)
    fig = go.Figure()

    # (A) 歷史 K 線圖
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="歷史 K 線",
            increasing_line_color="#ef5350",  # 台股紅漲
            decreasing_line_color="#26a69a",  # 台股綠跌
        )
    )

    # (B) 用於反射的基準價格線
    fig.add_trace(
        go.Scatter(
            x=sub_df.index,
            y=sub_df["Close"],
            mode="lines",
            name=f"反射基準區間 ({reflection_days}D)",
            line=dict(color="#ffa726", width=2.5),
        )
    )

    # (C) 亞當未來映射收盤價軌跡
    fig.add_trace(
        go.Scatter(
            x=df_proj.index,
            y=df_proj["Adam_Close"],
            mode="lines+markers",
            name="亞當二次映射軌跡",
            line=dict(color="#ab47bc", width=2.5, dash="dash"),
            marker=dict(size=4),
        )
    )

    # (D) 未來估算高低點通道包絡線 (Shading)
    fig.add_trace(
        go.Scatter(
            x=list(df_proj.index) + list(df_proj.index[::-1]),
            y=list(df_proj["Adam_High"]) + list(df_proj["Adam_Low"][::-1]),
            fill="toself",
            fillcolor="rgba(171, 71, 188, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="預測波動邊界 (High/Low)",
        )
    )

    # (E) 現狀分界線
    fig.add_vline(
        x=last_date.timestamp() * 1000,
        line_width=1,
        line_dash="dot",
        line_color="gray",
    )

    # Layout 優化
    fig.update_layout(
        title=f"{stock_id} 亞當理論二次映射預測圖",
        yaxis_title="價格 (TWD)",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=550,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # 7. 預測明細數據頁籤
    with st.expander("📊 查看詳細映射數據列表"):
        show_df = df_proj.copy()
        show_df.columns = ["預測收盤價", "預測最高價", "預測最低價"]
        st.dataframe(show_df.style.format("{:.2f}"), use_container_width=True)
