# ĐỒ ÁN: GÓI PHẦN MỀM ỨNG DỤNG CHO TÀI CHÍNH 1
#**Chủ đề:** Xây dựng Hệ thống Phân tích & Quản trị Danh mục Đầu tư VN30

## 1. GIỚI THIỆU DỰ ÁN
#Đây là ứng dụng Web App tương tác (Interactive Dashboard) được xây dựng bằng Python/Streamlit, nhằm mục đích:
#- Tự động hóa quy trình thu thập dữ liệu chứng khoán theo thời gian thực.
#- Thực hiện các kiểm định thống kê và mô hình hóa tài chính chuẩn mực (ARIMA, CAPM).
#- Đề xuất và kiểm thử chiến lược đầu tư dựa trên dữ liệu định lượng.

## 2. CẤU TRÚC DỰ ÁN & TÍNH NĂNG
#Dự án được thiết kế bám sát theo Rubric chấm điểm, bao gồm 6 Module chính:

### 🟢 Module 1: Dữ liệu & EDA
#- **Nguồn dữ liệu:** Kết nối API TCBS lấy dữ liệu Real-time (01/01/2020 - Nay).
#- **Xử lý:** + Làm sạch ngày nghỉ lễ/cuối tuần (Market Calendar Sync).
#  + Xử lý Timezone và dữ liệu khuyết (Forward Fill).
#- **Trực quan hóa:** Biểu đồ Nến không đứt quãng & Heatmap tương quan.

### 🟢 Module 2: Dự báo ARIMA
#- **Quy trình Box-Jenkins:**
# + Kiểm định tính dừng (ADF Test).
# + Ước lượng tham số (p, d, q) bằng phương pháp MLE.
# + **Kiểm định Ljung-Box:** Đánh giá phần dư  để đảm bảo mô hình tin cậy.

### 🟢 Module 3: Mô hình CAPM
#- **Benchmark:** Sử dụng chỉ số **VNINDEX thực tế**  để tính Beta chuẩn xác.
#- **Kết quả:** Bảng xếp hạng rủi ro hệ thống (Beta) và đường đặc tính chứng khoán (SML).

### 🟢 Module 4: Quản trị Danh mục
#- Phân loại danh mục: **An toàn ** vs **Mạo hiểm **.
#- Đánh giá hiệu quả: Sharpe Ratio & Max Drawdown.

### 🌟 Module 5: Phần Mở rộng - Bonus
#- **Backtest Engine:** Kiểm thử chiến lược giao dịch kỹ thuật (MA Crossover) trên dữ liệu quá khứ.
#- **Momentum Strategy:** Xếp hạng sức mạnh giá cổ phiếu.

### 🟢 Module 6: Báo cáo
#- Xuất toàn bộ kết quả phân tích ra file Excel chuẩn định dạng (.xlsx).

## 3. HƯỚNG DẪN CÀI ĐẶT & CHẠY
#**Yêu cầu môi trường:** Python 3.8+

#**Bước 1: Cài đặt thư viện**
#Mở Terminal và chạy lệnh sau:
#```bash
#pip install streamlit pandas numpy requests plotly statsmodels scipy xlsxwriter

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import statsmodels.api as sm
import io
import os
import warnings

# 1. CẤU HÌNH & KHỞI TẠO
st.set_page_config(
    page_title="Đồ Án: VN30 Index Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings("ignore")

# CSS
st.markdown("""
<style>
    .main-header {font-size: 24px; font-weight: bold; color: #1565C0;}
    .sub-header {font-size: 18px; font-weight: bold; color: #424242; margin-top: 20px;}
    .comment-box {background-color: #E3F2FD; border-left: 5px solid #2196F3; padding: 15px; border-radius: 5px; margin-top: 10px; font-style: italic; color: #000;}
    .bonus-intro {background-color: #FFF3E0; border-left: 5px solid #FF9800; padding: 15px; border-radius: 5px; margin-bottom: 15px; color: #000;}
    div.stButton > button:first-child {background-color: #1976D2; color: white; font-weight: bold; width: 100%; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# --- THAM SỐ DỰ ÁN ---
CURRENT_DATE = datetime.now()
START_DATE = '2020-01-01'
RISK_FREE_RATE = 0.025

# DANH SÁCH: VNINDEX LÀM BENCHMARK
TICKERS = [
    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
    'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'SAB', 'SHB', 'SSB', 'SSI', 'STB',
    'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE',
    'VNINDEX'
]

# 2. MODULE XỬ LÝ DỮ LIỆU
def sanitize_data(df):
    """Làm sạch dữ liệu thô"""
    if df.empty: return df
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.dropna(subset=['Close'], inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    return df.sort_index()

@st.cache_data(ttl=300)
def load_and_process_data():
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://tcinvest.tcbs.com.vn/'}
    combined_data = []

    if not os.path.exists('data_project'):
        os.makedirs('data_project')

    bar = st.progress(0, text="Đang đồng bộ dữ liệu thị trường...")
    start_ts = int(datetime.strptime(START_DATE, '%Y-%m-%d').timestamp())
    end_ts = int(CURRENT_DATE.timestamp())

    for i, ticker in enumerate(TICKERS):
        try:
            data_type = 'index' if ticker == 'VNINDEX' else 'stock'
            url = f"https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={ticker}&type={data_type}&resolution=D&from={start_ts}&to={end_ts}"
            res = requests.get(url, headers=headers, timeout=5)

            if res.status_code == 200:
                data = res.json()
                if 'data' in data and len(data['data']) > 0:
                    df = pd.DataFrame(data['data'])
                    df['Date'] = pd.to_datetime(df['tradingDate'])
                    df.set_index('Date', inplace=True)

                    df.rename(
                        columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'},
                        inplace=True)
                    df['Ticker'] = ticker

                    if data_type == 'stock':
                        cols_price = ['Open', 'High', 'Low', 'Close']
                        df[cols_price] = df[cols_price] / 1000

                    df_clean = sanitize_data(df)
                    df_clean.to_csv(f"data_project/{ticker}.csv")
                    combined_data.append(df_clean[['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']])
        except:
            pass
        bar.progress((i + 1) / len(TICKERS))

    bar.empty()
    if not combined_data: return None, None, None

    df_total = pd.concat(combined_data)
    df_prices = df_total.pivot_table(index=df_total.index, columns='Ticker', values='Close')

    # --- MARKET SYNC ---
    if 'VNINDEX' in df_prices.columns:
        valid_days = df_prices['VNINDEX'].dropna().index
        df_prices = df_prices.loc[valid_days]
        df_total = df_total[df_total.index.isin(valid_days)]

    df_prices.fillna(method='ffill', inplace=True)
    df_returns = np.log(df_prices / df_prices.shift(1)).dropna()

    return df_total, df_prices, df_returns

def get_rangebreaks(df):
    dt_all = pd.date_range(start=df.index[0], end=df.index[-1])
    dt_obs = [d.strftime("%Y-%m-%d") for d in df.index]
    return [d.strftime("%Y-%m-%d") for d in dt_all if d.strftime("%Y-%m-%d") not in dt_obs]

# 3. TÍNH TOÁN
df_raw, df_prices, df_returns = load_and_process_data()

if df_prices is None:
    st.error("Lỗi kết nối dữ liệu. Vui lòng kiểm tra lại.")
    st.stop()

# --- TÍNH CAPM ---
mkt_col = 'VNINDEX'
market_ret = df_returns[mkt_col]
res_capm = []

for t in df_returns.columns:
    if t != mkt_col:
        s, i, r, p_val, _ = stats.linregress(market_ret, df_returns[t])
        res_capm.append({'Mã': t, 'Beta': s, 'Alpha': i, 'R2': r ** 2})

df_capm = pd.DataFrame(res_capm).set_index('Mã')

# --- TÍNH PORTFOLIO & METRICS  ---
safe_stocks = df_capm.sort_values('Beta').head(3).index.tolist()
risky_stocks = df_capm.sort_values('Beta', ascending=False).head(3).index.tolist()

nav = pd.DataFrame()
nav['VNINDEX'] = (1 + df_returns['VNINDEX']).cumprod()
nav['An Toàn'] = (1 + df_returns[safe_stocks].mean(axis=1)).cumprod()
nav['Mạo Hiểm'] = (1 + df_returns[risky_stocks].mean(axis=1)).cumprod()

metrics = []
for col in nav.columns:
    # Tính toán cho từng danh mục
    r = df_returns['VNINDEX'] if col == 'VNINDEX' else df_returns[safe_stocks].mean(axis=1) if col == 'An Toàn' else \
    df_returns[risky_stocks].mean(axis=1)

    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol
    max_dd = ((nav[col] - nav[col].cummax()) / nav[col].cummax()).min()

    metrics.append({
        'Danh mục': col,
        'Lợi nhuận Năm': f"{ann_ret:.2%}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_dd:.2%}"
    })

# --- C. MOMENTUM ---
mom = df_prices.pct_change(252).iloc[-1].sort_values(ascending=False)

# 4. GIAO DIỆN & TABS
with st.sidebar:
    st.title("⚙️ Project Control")
    st.info(f"**Data Range:** {START_DATE} ➔ {df_prices.index.max().strftime('%Y-%m-%d')}")
    st.success(f"**Market Benchmark:** VNINDEX")
    with st.expander("✅ Rubric Checklist"):
        st.markdown("Dữ liệu & EDA")
        st.markdown("ARIMA Dự báo")
        st.markdown("CAPM")
        st.markdown("Danh mục Hiệu quả")
        st.markdown("Bonus: Backtest & Momentum")

st.title("📊 BÁO CÁO PHÂN TÍCH ĐẦU TƯ CHỨNG KHOÁN VN30")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dữ liệu & EDA",
    "Dự báo ARIMA",
    "Định giá CAPM",
    "Danh mục Đầu tư",
    "Phần Mở rộng",
    "Xuất Báo cáo"
])

# --- TAB 1: EDA ---
with tab1:
    st.markdown("<h3 class='sub-header'>Phân tích Dữ liệu & Thống kê Mô tả</h3>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])
    with c2:
        st.markdown("**Thống kê Lợi suất**")
        ticker = st.selectbox("Chọn mã:", df_prices.columns, index=0)
        desc = df_returns[ticker].describe()
        st.dataframe(desc)

        mean_ret = desc['mean'] * 100
        std_dev = desc['std'] * 100
        st.markdown(f"""
        <div class='comment-box'>
        <b>Nhận xét sơ bộ:</b><br>
        Biến động trung bình ngày của <b>{ticker}</b> là <b>{mean_ret:.3f}%</b>. 
        Độ lệch chuẩn (rủi ro) là <b>{std_dev:.2f}%</b>.
        </div>
        """, unsafe_allow_html=True)

    with c1:
        df_one = df_raw[df_raw['Ticker'] == ticker]
        df_one = df_one[df_one.index.isin(df_prices.index)]
        fig = go.Figure(data=[
            go.Candlestick(x=df_one.index, open=df_one['Open'], high=df_one['High'], low=df_one['Low'],
                           close=df_one['Close'])])
        fig.update_xaxes(rangebreaks=[dict(values=get_rangebreaks(df_one))])
        fig.update_layout(height=500, title=f"Biểu đồ giá: {ticker}", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        last_p = df_one['Close'].iloc[-1]
        st.markdown(f"<div class='comment-box'>Giá đóng cửa phiên gần nhất: <b>{last_p:,.2f}</b>.</div>",
                    unsafe_allow_html=True)

    # --- HEATMAP ---
    st.markdown("---")
    st.markdown("**Biểu đồ nhiệt Tương quan**")
    fig_corr = px.imshow(df_returns.corr(), text_auto=False, aspect="auto", color_continuous_scale='RdBu_r', zmin=-1,
                         zmax=1)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown(
        "<div class='comment-box'><b>Nhận xét:</b> Các ô màu đỏ thể hiện tương quan thuận cao (cùng tăng/giảm), màu xanh đậm là tương quan nghịch. VNINDEX thường có tương quan dương mạnh với các mã ngân hàng (ACB, TCB, MBB).</div>",
        unsafe_allow_html=True)

# --- TAB 2: ARIMA ---
with tab2:
    st.markdown("<h3 class='sub-header'>Dự báo Chuỗi thời gian (ARIMA)</h3>", unsafe_allow_html=True)

    c_ctrl, c_plot = st.columns([1, 3])
    with c_ctrl:
        target = st.selectbox("Mã dự báo:", df_prices.columns,
                              index=df_prices.columns.get_loc('HPG') if 'HPG' in df_prices.columns else 0)
        monthly = df_prices[target].resample('ME').last()
        m_ret = np.log(monthly / monthly.shift(1)).dropna()

        adf = adfuller(m_ret)
        is_stat = adf[1] < 0.05
        st.info(f"ADF P-value: {adf[1]:.4f}\n({'✅ Dừng' if is_stat else '⚠️ Cần d=1'})")
        p = st.number_input("p", 0, 5, 1);
        q = st.number_input("q", 0, 5, 1);
        d = 0 if is_stat else 1
    with c_plot:
        try:
            model = ARIMA(m_ret, order=(p, d, q)).fit()
            fc = model.get_forecast(1)
            pred = (np.exp(fc.predicted_mean.iloc[0]) - 1) * 100
            lb_p = sm.stats.acorr_ljungbox(model.resid, lags=[10], return_df=True)['lb_pvalue'].values[0]

            st.markdown(
                f"<div class='comment-box'>Dự báo tháng sau: <b>{pred:.2f}%</b> | Ljung-Box P={lb_p:.4f} ({'Tốt' if lb_p > 0.05 else 'Chưa tốt'})</div>",
                unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=m_ret.index, y=m_ret, name='Thực tế'))
            fig.add_trace(
                go.Scatter(x=m_ret.index, y=model.fittedvalues, name='Mô hình', line=dict(color='red', dash='dot')))
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.error("Lỗi tham số.")

# --- TAB 3: CAPM ---
with tab3:
    st.markdown("<h3 class='sub-header'>Mô hình Định giá Tài sản Vốn (CAPM)</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("**Bảng Hệ số Beta**")
        st.dataframe(
            df_capm.sort_values('Beta', ascending=False).style.background_gradient(cmap='Reds', subset=['Beta']))
    with c2:
        st.write("**Đường SML (Scatter Plot)**")
        pick = st.selectbox("Chi tiết:", df_returns.columns, index=0)
        if pick != mkt_col:
            fig = px.scatter(x=market_ret, y=df_returns[pick], trendline="ols", labels={'x': mkt_col, 'y': pick})
            st.plotly_chart(fig, use_container_width=True)
            beta_v = df_capm.loc[pick, 'Beta']
            st.markdown(
                f"<div class='comment-box'>Beta = {beta_v:.2f}. Rủi ro hệ thống: {'Cao' if beta_v > 1 else 'Thấp'}.</div>",
                unsafe_allow_html=True)

# --- TAB 4: PORTFOLIO ---
with tab4:
    st.markdown("<h3 class='sub-header'>Danh mục Đầu tư</h3>", unsafe_allow_html=True)
    st.write(f"🛡️ **An toàn:** {', '.join(safe_stocks)} | 🚀 **Mạo hiểm:** {', '.join(risky_stocks)}")
    st.line_chart(nav)
    st.table(pd.DataFrame(metrics).set_index('Danh mục'))
    st.markdown(f"<div class='comment-box'><b>Đánh giá:</b> Danh mục có Sharpe Ratio cao nhất là hiệu quả nhất.</div>",
                unsafe_allow_html=True)

# --- TAB 5: BONUS ---
with tab5:
    st.markdown("<h3 class='sub-header'>Bonus</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='bonus-intro'><b>Chiến lược Backtest (MA Crossover):</b> Mua khi đường MA ngắn cắt lên đường MA dài.</div>",
        unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        bt = st.selectbox("Mã Backtest:", df_prices.columns,
                          index=df_prices.columns.get_loc('HPG') if 'HPG' in df_prices.columns else 0)
    if st.button("Chạy Backtest"):
        d = df_prices[[bt]].copy()
        d['S'] = d[bt].rolling(20).mean();
        d['L'] = d[bt].rolling(50).mean()
        d['Sig'] = np.where(d['S'] > d['L'], 1.0, 0.0)
        d['Ret'] = d[bt].pct_change()
        d['Strat'] = d['Ret'] * pd.Series(d['Sig']).shift(1).values
        cum = (1 + d[['Ret', 'Strat']].fillna(0)).cumprod()
        with c2:
            st.line_chart(cum)
            profit = cum['Strat'].iloc[-1] - 1
            st.success(f"Lợi nhuận Chiến lược: {profit:.2%}")

    st.markdown("---")
    st.markdown("#### Momentum Ranking")
    st.dataframe(mom.head(5).to_frame("Tăng trưởng 1Y"))

# --- TAB 6: EXPORT ---
with tab6:
    st.markdown("<h3 class='sub-header'>Xuất Báo Cáo</h3>", unsafe_allow_html=True)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as w:
        df_prices.to_excel(w, sheet_name='Prices')
        df_returns.describe().to_excel(w, sheet_name='Stats')
        df_capm.to_excel(w, sheet_name='CAPM')
        pd.DataFrame(metrics).to_excel(w, sheet_name='Portfolio')
        mom.to_frame('Momentum').to_excel(w, sheet_name='Bonus')
    buf.seek(0)
    st.download_button("📥 Tải Excel Nộp Bài", buf, "Bao_Cao_VN30.xlsx")