import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# 設定 Streamlit 頁面標題與配置
st.set_page_config(page_title="台股亞當理論分析與映射預測", layout="wide")

st.title("📈 台股亞當理論 (Adam Theory) 順勢分析與二次映射預測")
st.markdown("""
亞當理論的核心是**不預測，只順應趨勢**。
本 App 使用亞當理論著名的**「二次映射法 (Second Reflection)」**，將近期的價格形態沿中心點翻轉，繪製出未來的可能軌跡。
""")

# 1. 側邊欄輸入參數
st.sidebar.header("參數設定")
stock_id = st.sidebar.text_input(
    "股票代碼 (台股請加 .TW 或 .TWO)", value="2344.TW"
)

# 歷史資料預設抓取近 1 年
default_start = pd.Timestamp.now() - pd.DateOffset(years=1)
start_date = st.sidebar.date_input("歷史資料開始日期", default_start)

# 結束日期預設為「今天」
end_date = st.sidebar.date_input("歷史資料結束日期", pd.Timestamp.now())

# 亞當理論參數
reflection_days = st.sidebar.slider(
    "亞當映射天數 (觀察過去幾天來映射未來)",
    min_value=10,
    max_value=60,
    value=20,
)

if st.sidebar.button("開始亞當理論分析"):
    with st.spinner("讀取最新數據與計算亞當映射中..."):

        # 關鍵修復：yfinance 的 end 是不包含當天的，因此 +1 天才能抓到最新的價格
        fetch_end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)

        # 下載歷史資料
        df = yf.download(stock_id, start=start_date, end=fetch_end_date)

        if df.empty or len(df) < reflection_days:
            st.error("資料不足或抓取失敗。請確認股票代碼與日期範圍。")
        else:
            # 解決 yfinance MultiIndex 欄位問題
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 確保欄位為一維 Series
            close_s = df["Close"].squeeze()

            # 顯示當前抓到的最新數據日期，方便核對
            last_traded_date = df.index[-1].strftime("%Y-%m-%d")
            st.caption(
                f"📅 資料來源最新交易日：**{last_traded_date}**（台股盤後數據約有 15-20 分鐘延遲）"
            )

            # --- [Part 1: 亞當理論 - 二次映射計算] ---
            recent_prices = close_s.tail(reflection_days).to_numpy().flatten()
            last_price = float(recent_prices[-1])
            last_date = df.index[-1]

            # 第一次映射 (上下翻轉) + 第二次映射 (時間翻轉)
            reversed_historical = recent_prices[::-1]
            future_adam_prices = last_price + (last_price - reversed_historical)

            # 關鍵修正：確保起始預測日自動跳過週末，精確定位至下一個工作日
            next_start_date = last_date + pd.Timedelta(days=1)
            future_dates = pd.bdate_range(
                start=next_start_date,
                periods=len(future_adam_prices),
                roll="forward",
            )

            # 建立映射 DataFrame
            df_future = pd.DataFrame(
                {"Adam_Projection": future_adam_prices}, index=future_dates
            )

            # --- [Part 2: 手機端版面優化 - 顯示結果與建議] ---
            adam_target_price = future_adam_prices[-1]
            price_change = adam_target_price - last_price
            pct_change = (price_change / last_price) * 100

            st.subheader("🎯 亞當二次映射預測結果")

            # 手機版面採 2 欄配置
            col1, col2 = st.columns(2)
            col1.metric("最新收盤價", f"{last_price:.2f} TWD")
            col2.metric(
                f"未來 {reflection_days} 日目標價",
                f"{adam_target_price:.2f} TWD",
                f"{pct_change:+.2f}%",
            )

            # 全幅提示框呈現趨勢建議
            if pct_change > 3:
                st.success(
                    "💡 **亞當趨勢建議：** 🚀 **強勢多頭 (順勢做多)**\n\n映射顯示價格持續向上突破，建議順勢做多或持股續抱。"
                )
            elif pct_change < -3:
                st.error(
                    "💡 **亞當趨勢建議：** 📉 **強勢空頭 (順勢做空/避險)**\n\n映射顯示價格將轉弱向下，建議順勢減碼或進行避險。"
                )
            else:
                st.info(
                    "💡 **亞當趨勢建議：** ⚖️ **橫盤整理 (觀望待變)**\n\n映射無明顯趨勢方向，建議多看少做，等待突破。"
                )

            # --- [Part 3: 繪製亞當映射圖表] ---
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")

            # 繪製真實歷史價格
            ax.plot(
                df.index,
                close_s,
                label=f"{stock_id} Historical Price",
                color="black",
                linewidth=1.5,
            )

            # 標示用於映射的歷史片段 (黃色線)
            ax.plot(
                df.index[-reflection_days:],
                recent_prices,
                color="gold",
                linewidth=2.5,
                label=f"Reflection Base ({reflection_days}D)",
            )

            # 繪製亞當未來映射線 (紅色點虛線)
            ax.plot(
                df_future.index,
                df_future["Adam_Projection"],
                label="Adam Projection (Future)",
                color="red",
                linestyle="--",
                linewidth=2,
                marker="o",
                markersize=3,
            )

            # 畫出目前時間邊界
            ax.axvline(
                x=last_date, color="gray", linestyle=":", label="Present Day"
            )

            ax.set_title(
                f"Adam Theory - Second Reflection for {stock_id}", fontsize=12
            )
            ax.set_ylabel("Price (TWD)", color="black")
            ax.legend(loc="upper left", fontsize="small")
            ax.grid(True, linestyle="--", alpha=0.3)

            # 時間軸格式化
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
            plt.xticks(rotation=45)

            # 顯示圖表
            st.pyplot(fig)

            # --- [Part 4: 顯示預測數據表] ---
            st.subheader(f"🔮 未來 {reflection_days} 個交易日軌跡數據")
            st.dataframe(
                df_future.style.format("{:.2f}"), use_container_width=True
            )
