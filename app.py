import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 設定 Streamlit 頁面標題與配置
st.set_page_config(page_title="台股亞當理論分析與映射預測", layout="wide")

st.title("📈 台股亞當理論 (Adam Theory) 順勢分析與二次映射預測")
st.markdown("""
亞當理論的核心是**不預測，只順應趨勢**。
本 App 使用亞當理論著名的**「二次映射法 (Second Reflection)」**，將近期的價格形態沿中心點翻轉，繪製出未來的可能軌跡。
""")

# 1. 側邊欄輸入參數
st.sidebar.header("參數設定")
stock_id = st.sidebar.text_input("股票代碼 (台股請加 .TW 或 .TWO)", value="2330.TW")
start_date = st.sidebar.date_input("歷史資料開始日期", pd.to_datetime("2025-01-01"))
end_date = st.sidebar.date_input("歷史資料結束日期", pd.to_datetime("2026-08-01"))

# 亞當理論參數
reflection_days = st.sidebar.slider("亞當映射天數 (觀察過去幾天來映射未來)", min_value=10, max_value=60, value=20)

if st.sidebar.button("開始亞當理論分析"):
    with st.spinner("讀取數據與計算亞當映射中..."):
        # 下載歷史資料
        df = yf.download(stock_id, start=start_date, end=end_date)

        if df.empty or len(df) < reflection_days:
            st.error(f"資料不足，無法進行分析。請確保日期範圍至少包含 {reflection_days} 個交易日。")
        else:
            # 解決 yfinance MultiIndex 欄位問題
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 確保欄位為一維 Series
            close_s = df['Close'].squeeze()
            
            # --- [Part 1: 亞當理論 2.0 - 二次映射計算 (已修正維度)] ---
            # 取出最近 N 天的歷史價格並轉為一維陣列
            recent_prices = close_s.tail(reflection_days).to_numpy().flatten()
            last_price = float(recent_prices[-1])
            last_date = df.index[-1]

            # 第一次映射 (上下翻轉) + 第二次映射 (時間翻轉)
            reversed_historical = recent_prices[::-1]
            future_adam_prices = last_price + (last_price - reversed_historical)

            # 產生未來的交易日 (精準對應長度)
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(future_adam_prices), freq='B')

            # 建立映射 DataFrame
            df_future = pd.DataFrame({
                'Adam_Projection': future_adam_prices
            }, index=future_dates)

            # --- [Part 2: 亞當趨勢強度與操作面版] ---
            adam_target_price = future_adam_prices[-1]
            price_change = adam_target_price - last_price
            pct_change = (price_change / last_price) * 100

            st.subheader("🎯 亞當二次映射預測結果")
            cols = st.columns(3)
            cols[0].metric("最新收盤價", f"{last_price:.2f} TWD")
            cols[1].metric(f"未來 {reflection_days} 日亞當目標價", f"{adam_target_price:.2f} TWD", f"{pct_change:+.2f}%")
            
            trend_signal = "🚀 強勢多頭 (順勢做多)" if pct_change > 3 else ("📉 強勢空頭 (順勢做空)" if pct_change < -3 else "橫盤整理 (觀望)")
            cols[2].metric("亞當趨勢建議", trend_signal)

            # --- [Part 3: 繪製亞當映射圖表] ---
            fig, ax = plt.subplots(figsize=(10, 6))
            
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            # 繪製真實歷史價格
            ax.plot(df.index, close_s, label=f'{stock_id} Historical Price', color='black', linewidth=1.5)
            
            # 標示用於映射的歷史片段 (黃色線)
            ax.plot(df.index[-reflection_days:], recent_prices, color='gold', linewidth=2.5, label=f'Reflection Base ({reflection_days}D)')

            # 繪製亞當未來映射線 (紅色點虛線)
            ax.plot(df_future.index, df_future['Adam_Projection'], label='Adam Projection (Future)', color='red', linestyle='--', linewidth=2, marker='o', markersize=3)

            # 畫出目前時間邊界
            ax.axvline(x=last_date, color='gray', linestyle=':', label='Present Day')

            ax.set_title(f'Adam Theory - Second Reflection for {stock_id}')
            ax.set_ylabel('Price (TWD)', color='black')
            ax.legend(loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.3)

            # 時間軸格式化
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
            plt.xticks(rotation=45)

            # 顯示圖表
            st.pyplot(fig)

            # --- [Part 4: 顯示預測數據表] ---
            st.subheader(f"🔮 未來 {reflection_days} 個交易日亞當映射預測軌跡")
            st.dataframe(df_future.style.format("{:.2f}"))
