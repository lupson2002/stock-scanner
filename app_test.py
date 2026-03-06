import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone
from supabase import create_client, Client
from scipy.signal import argrelextrema
import time
import re
import random
import concurrent.futures

# [NEW] ETF 및 시총 스크래핑용 추가 라이브러리
import FinanceDataReader as fdr
import requests
from bs4 import BeautifulSoup
from finvizfinance.screener.overview import Overview

# =========================================================
# [설정] Supabase 연결 정보 (보안 적용)
# =========================================================
try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
except Exception as e:
    st.error(f"⚠️ Secrets 설정이 필요합니다. (에러: {e})")
    st.stop()

# ==========================================
# 1. 페이지 설정 및 DB 연결
# ==========================================
st.set_page_config(page_title="Pro 주식 검색기", layout="wide")
st.title("📈 Pro 주식 검색기: 섹터/국가/기술적/시총상위/퀀티와이즈 통합")

@st.cache_resource
def init_supabase():
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        return None

supabase = init_supabase()

# ==========================================
# 2. 구글 시트 연결 설정
# ==========================================
SHEET_ID = '1NVThO1z2HHF0TVXVRGmbVsSU_Svyjg8fxd7E90z2o8A'
STOCK_GID = '0' 
STOCK_CSV_URL = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={STOCK_GID}'
ETF_GID = '2023286696'
ETF_CSV_URL = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={ETF_GID}'
COUNTRY_GID = '1247750129'
COUNTRY_CSV_URL = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={COUNTRY_GID}'

# ==========================================
# 3. 공통 함수 정의
# ==========================================
def get_tickers_from_sheet():
    try:
        df = pd.read_csv(STOCK_CSV_URL, header=None)
        tickers = sorted(list(set([str(x).strip() for x in df[0] if str(x).strip()])))
        return tickers
    except Exception as e:
        st.error(f"주식 시트 읽기 실패: {e}")
        return []

def get_etfs_from_sheet():
    try:
        df = pd.read_csv(ETF_CSV_URL, header=None)
        etf_list = []
        for index, row in df.iterrows():
            raw_ticker = str(row[0]).strip()
            if not raw_ticker or raw_ticker.lower() in ['ticker', 'symbol', '종목코드', '티커', 'nan']:
                continue
            ticker = raw_ticker.split(':')[-1].strip() if ':' in raw_ticker else raw_ticker
            name = str(row[1]).strip() if len(row) > 1 else ticker
            if ticker: etf_list.append((ticker, name))
        return etf_list
    except Exception as e: return []

def get_country_etfs_from_sheet():
    try:
        df = pd.read_csv(COUNTRY_CSV_URL, header=None)
        etf_list = []
        for index, row in df.iterrows():
            raw_ticker = str(row[0]).strip()
            if not raw_ticker or raw_ticker.lower() in ['ticker', 'symbol', '종목코드', '티커', 'nan']: continue
            ticker = raw_ticker.split(':')[-1].strip() if ':' in raw_ticker else raw_ticker
            name = str(row[1]).strip() if len(row) > 1 else ticker
            if ticker: etf_list.append((ticker, name))
        return etf_list
    except Exception as e: return []

def get_unique_tickers_from_db():
    if not supabase: return []
    try:
        response = supabase.table("history").select("ticker").execute()
        if response.data: return list(set([row['ticker'] for row in response.data]))
        return []
    except Exception: return []

def remove_duplicates_from_db():
    if not supabase: return
    try:
        response = supabase.table("history").select("id, ticker, created_at").order("created_at", desc=True).execute()
        data = response.data
        if not data: return
        seen_tickers = set()
        ids_to_remove = []
        for row in data:
            if row['ticker'] in seen_tickers: ids_to_remove.append(row['id'])
            else: seen_tickers.add(row['ticker'])
        if ids_to_remove:
            for pid in ids_to_remove: supabase.table("history").delete().eq("id", pid).execute()
            st.success(f"🧹 중복 데이터 {len(ids_to_remove)}개 삭제 완료.")
    except Exception as e: st.error(f"중복 제거 실패: {e}")

# [수정됨] 병렬 처리 시 데이터 꼬임 완벽 차단을 위한 독립 세션 적용 및 내부 스레드 강제 종료
def smart_download(ticker, interval="1d", period="2y"):
    if ':' in ticker: ticker = ticker.split(':')[-1]
    ticker = ticker.replace('/', '-')
    ticker_yf = ticker.replace(' ', '-')
    
    candidates = [ticker_yf]
    if ticker_yf.isdigit() and len(ticker_yf) == 6:
        candidates = [f"{ticker_yf}.KS", f"{ticker_yf}.KQ", ticker_yf]
    
    # 각 스레드마다 고유한 통신 세션 발급
    session = requests.Session()
    
    for t in candidates:
        try:
            for _ in range(3):
                # threads=False를 주어 yfinance 자체의 버그성 멀티스레딩을 끄고, session으로 통신 분리
                df = yf.download(
                    t, period=period, interval=interval, progress=False, 
                    auto_adjust=False, threads=False, session=session
                )
                
                if len(df) > 0:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.loc[:, ~df.columns.duplicated()].copy()
                    session.close() # 메모리 해제
                    return t, df
                time.sleep(0.3)
        except: 
            continue
            
    session.close()
    return ticker, pd.DataFrame()

@st.cache_data(ttl=3600*24) 
def get_ticker_info_safe(ticker):
    try:
        tick = yf.Ticker(ticker)
        for _ in range(3):
            try:
                meta = tick.info
                if meta: return meta
            except: time.sleep(0.5)
        return None
    except: return None

def get_stock_sector(ticker):
    meta = get_ticker_info_safe(ticker)
    if not meta: return "Unknown"
    quote_type = meta.get('quoteType', '').upper()
    if 'ETF' in quote_type or 'FUND' in quote_type:
        name = meta.get('shortName', meta.get('longName', 'ETF'))
        return f"[ETF] {name}"
    sector = meta.get('sector', meta.get('industry', meta.get('shortName', '')))
    translations = {
        'Technology': '기술', 'Healthcare': '헬스케어', 'Financial Services': '금융',
        'Consumer Cyclical': '임의소비재', 'Industrials': '산업재', 'Basic Materials': '소재',
        'Energy': '에너지', 'Utilities': '유틸리티', 'Real Estate': '부동산',
        'Communication Services': '통신', 'Consumer Defensive': '필수소비재',
        'Semiconductors': '반도체'
    }
    return translations.get(sector, sector)

def save_to_supabase(data_list, strategy_name):
    if not supabase or not data_list: return
    rows_to_insert = []
    for item in data_list:
        rows_to_insert.append({
            "ticker": str(item.get('종목코드', '')),
            "sector": str(item.get('섹터', '-')),
            "price": str(item.get('현재가', '')).replace(',', ''),
            "strategy": strategy_name,
            "high_date": str(item.get('현52주신고가일', '')), 
            "bw": str(item.get('BW_Value', '')), 
            "macd_v": str(item.get('MACD_V_Value', ''))
        })
    try:
        supabase.table("history").insert(rows_to_insert).execute()
        st.toast(f"✅ {len(rows_to_insert)}개 종목 DB 저장 완료!", icon="💾")
    except Exception as e: pass

def normalize_ticker_for_db_storage(t):
    if not t: return ""
    t_str = str(t).upper().strip()
    if t_str.endswith("-US"): return t_str[:-3].replace('.', '-')
    if t_str.endswith("-HK"): return t_str[:-3] + ".HK"
    if t_str.endswith("-JP"): return t_str[:-3] + ".T"
    if t_str.endswith("-KS"): return t_str[:-3]
    if t_str.endswith("-KQ"): return t_str[:-3]
    if '-' in t_str and not any(x in t_str for x in ['-US', '-HK', '-JP', '-KS', '-KQ']): return t_str.split('-')[0]
    return t_str

def normalize_ticker_for_app_lookup(t):
    if not t: return ""
    t_str = str(t).upper().strip()
    if t_str.endswith(".KS"): return t_str[:-3]
    if t_str.endswith(".KQ"): return t_str[:-3]
    if '.' in t_str and not any(x in t_str for x in ['.HK', '.T', '.KS', '.KQ']): return t_str.replace('.', '-')
    return t_str

@st.cache_data(ttl=600) 
def fetch_latest_quant_data_from_db():
    if not supabase: return {}
    try:
        response = supabase.table("quant_data").select("*").order("created_at", desc=True).execute()
        if not response.data: return {}
        df = pd.DataFrame(response.data)
        df_latest = df.drop_duplicates(subset='ticker', keep='first')
        return {row['ticker']: {'1w': str(row.get('change_1w') or "-"), '1m': str(row.get('change_1m') or "-"), '3m': str(row.get('change_3m') or "-")} for _, row in df_latest.iterrows()}
    except Exception: return {}

GLOBAL_QUANT_DATA = fetch_latest_quant_data_from_db()

def get_eps_changes_from_db(ticker):
    norm_ticker = normalize_ticker_for_app_lookup(ticker)
    if norm_ticker in GLOBAL_QUANT_DATA:
        d = GLOBAL_QUANT_DATA[norm_ticker]
        return d['1w'], d['1m'], d['3m']
    return "-", "-", "-"

# ==========================================
# [NEW] 시총 상위 티커 캐싱 (클라우드 차단 예외처리 포함)
# ==========================================
@st.cache_data(ttl=86400)
def get_top_marketcap_tickers():
    tickers = []
    errors = []
    
    try:
        df_krx = fdr.StockListing('KRX')
        kr_tickers = df_krx.sort_values(by='Marcap', ascending=False).head(400)['Code'].tolist()
        tickers.extend(kr_tickers)
    except Exception as e:
        errors.append(f"KRX 한국주식 수집 실패: {e}")

    try:
        stock_screener = Overview()
        stock_screener.set_filter(filters_dict={'Industry': 'Stocks only (ex-Funds)'})
        df_us_stocks = stock_screener.screener_view(order='Market Cap', ascend=False)
        tickers.extend(df_us_stocks.head(1000)['Ticker'].tolist())
    except Exception as e:
        errors.append("Finviz 미국주식 스크리닝 서버 접속 차단(Cloudflare). 한국 주식만 분석합니다.")

    try:
        etf_screener = Overview()
        etf_screener.set_filter(filters_dict={'Industry': 'Exchange Traded Fund'})
        df_us_etfs = etf_screener.screener_view(order='Market Cap', ascend=False)
        tickers.extend(df_us_etfs.head(400)['Ticker'].tolist())
    except Exception as e:
        pass 

    return sorted(list(set(tickers))), errors

# ==========================================
# 4. 분석 알고리즘 (원본 100% 유지)
# ==========================================
def calculate_macdv(df, short=12, long=26, signal=9):
    ema_fast = df['Close'].ewm(span=short, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=long, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(span=long, adjust=False).mean()
    macd_v = (macd_line / (atr + 1e-9)) * 100
    macd_v_signal = macd_v.ewm(span=signal, adjust=False).mean()
    return macd_v, macd_v_signal

def calculate_common_indicators(df, is_weekly=False):
    if len(df) < 60: return None
    df = df.copy()
    period = 20 if is_weekly else 60
    df[f'EMA{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    df[f'STD{period}'] = df['Close'].rolling(window=period).std()
    df['BB_UP'] = df[f'EMA{period}'] + (2 * df[f'STD{period}'])
    df['BB_LO'] = df[f'EMA{period}'] - (2 * df[f'STD{period}'])
    df['BandWidth'] = (df['BB_UP'] - df['BB_LO']) / df[f'EMA{period}']
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_V'], df['MACD_V_Signal'] = calculate_macdv(df, 12, 26, 9)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = tr.ewm(span=14, adjust=False).mean()
    return df

def calculate_daily_indicators(df):
    if len(df) < 260: return None
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['STD50'] = df['Close'].rolling(window=50).std()
    df['BB50_UP'] = df['SMA50'] + (2.0 * df['STD50'])
    df['BB50_LO'] = df['SMA50'] - (2.0 * df['STD50'])
    df['BW50'] = (df['BB50_UP'] - df['BB50_LO']) / df['SMA50']
    df['Donchian_High_50'] = df['High'].rolling(window=50).max().shift(1)
    df['Change'] = df['Close'].diff()
    df['Vol_Up'] = np.where(df['Change'] > 0, df['Volume'], 0)
    df['Vol_Down'] = np.where(df['Change'] < 0, df['Volume'], 0)
    df['Vol_Flat'] = np.where(df['Change'] == 0, df['Volume'], 0)
    roll_up = df['Vol_Up'].rolling(window=50).sum()
    roll_down = df['Vol_Down'].rolling(window=50).sum()
    roll_flat = df['Vol_Flat'].rolling(window=50).sum()
    df['VR50'] = ((roll_up + roll_flat/2) / (roll_down + roll_flat/2 + 1e-9)) * 100
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB20_UP'] = df['SMA20'] + (2.0 * df['STD20'])
    df['BB20_LO'] = df['SMA20'] - (2.0 * df['STD20'])
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR20'] = df['TR'].rolling(window=20).mean()
    kc_mult = 1.5 
    df['KC20_UP'] = df['SMA20'] + (kc_mult * df['ATR20'])
    df['KC20_LO'] = df['SMA20'] - (kc_mult * df['ATR20'])
    df['TTM_Squeeze'] = (df['BB20_UP'] < df['KC20_UP']) & (df['BB20_LO'] > df['KC20_LO'])
    ema_fast = df['Close'].ewm(span=20, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=200, adjust=False).mean()
    df['MACD_Line_C'] = ema_fast - ema_slow
    df['MACD_Signal_C'] = df['MACD_Line_C'].ewm(span=20, adjust=False).mean()
    df['MACD_OSC_C'] = df['MACD_Line_C'] - df['MACD_Signal_C']
    df['ATR14'] = df['TR'].ewm(span=14, adjust=False).mean()
    df['MACD_V'], _ = calculate_macdv(df, 12, 26, 9)
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    return df

def check_vcp_pattern(df):
    if len(df) < 250: return False, None
    df = calculate_daily_indicators(df) 
    if df is None: return False, None
    
    curr = df.iloc[-1]
    sma50 = df['Close'].rolling(50).mean().iloc[-1]
    sma150 = df['Close'].rolling(150).mean().iloc[-1]
    sma200 = df['Close'].rolling(200).mean().iloc[-1]
    
    # 오리지널 원본 로직 복구
    cond1 = curr['Close'] > sma150 and curr['Close'] > sma200
    cond2 = sma150 > sma200
    cond3 = df['SMA50'].iloc[-1] > df['SMA50'].iloc[-20] 
    cond4 = sma50 > sma150
    low_52 = df['Low'].iloc[-252:].min()
    cond5 = curr['Close'] > low_52 * 1.25
    high_52 = df['High'].iloc[-252:].max()
    cond6 = curr['Close'] > high_52 * 0.75
    
    stage_1_pass = cond1 and cond2 and cond3 and cond4 and cond5 and cond6
    if not stage_1_pass: return False, None 

    window = 60
    subset = df.iloc[-window:]
    p1 = subset.iloc[:20]    
    p2 = subset.iloc[20:40]  
    p3 = subset.iloc[40:]    
    
    range1 = (p1['High'].max() - p1['Low'].min()) / p1['High'].max()
    range2 = (p2['High'].max() - p2['Low'].min()) / p2['High'].max()
    range3 = (p3['High'].max() - p3['Low'].min()) / p3['High'].max()
    
    contraction = (range3 < range2) or (range2 < range1) or (range3 < 0.12)
    if not contraction: return False, None

    last_vol_avg = p3['Volume'].mean()
    prev_vol_avg = p1['Volume'].mean()
    vol_dry_up = last_vol_avg < prev_vol_avg * 1.2 
    tight_area = range3 < 0.15 
    
    stage_3_pass = vol_dry_up and tight_area
    
    stop_loss = p3['Low'].min()
    risk = curr['Close'] - stop_loss
    target_price = curr['Close'] + (risk * 3) if risk > 0 else 0
    
    prior_days = p3.iloc[:-1] 
    if len(prior_days) > 0:
        pivot_point = prior_days['High'].max() 
    else:
        pivot_point = p3['High'].max() 

    vol_ma50 = df['Volume'].iloc[-51:-1].mean()
    breakout = (curr['Close'] > pivot_point) and (curr['Volume'] > vol_ma50 * 1.2)
    
    status = ""
    if stage_3_pass and not breakout:
        status = "3단계 (수렴중)"
    elif stage_3_pass and breakout:
        status = "4단계 (돌파!🚀)"
    else:
        if breakout and tight_area:
             status = "4단계 (돌파!🚀)"
        else:
             return False, None

    return True, {
        'status': status, 'stop_loss': stop_loss, 'target_price': target_price,
        'squeeze': "🔥" if df['TTM_Squeeze'].iloc[-1] else "-", 'price': curr['Close'], 'pivot': pivot_point 
    }

def get_weekly_macd_status(daily_df):
    try:
        df_w = daily_df.resample('W-FRI').agg({
            'Close': 'last', 'High': 'max', 'Low': 'min', 'Volume': 'sum'
        }).dropna()
        if len(df_w) < 26: return "-"
        ema12 = df_w['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_w['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        curr_macd = macd_line.iloc[-1]
        curr_sig = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_sig = signal_line.iloc[-2]
        if curr_macd > curr_sig:
            if prev_macd <= prev_sig: return "⚡GC (매수신호)"
            else: return "🔵 Buy (유지)"
        else: return "🔻 Sell (매도)"
    except: return "-"

def plot_vcp_chart(df, ticker, info):
    df_plot = df.iloc[-252:].copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'], name='Price'
    ))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'].rolling(50).mean(), line=dict(color='green', width=1), name='SMA 50'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'].rolling(150).mean(), line=dict(color='blue', width=1), name='SMA 150'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'].rolling(200).mean(), line=dict(color='red', width=1), name='SMA 200'))
    fig.add_hline(y=info['pivot'], line_dash="dot", line_color="red", annotation_text="Pivot (Breakout)")
    fig.add_hline(y=info['stop_loss'], line_dash="dot", line_color="blue", annotation_text="Stop Loss")
    fig.update_layout(title=f"{ticker} - VCP Analysis Chart", xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
    return fig

def check_daily_condition(df):
    if len(df) < 260: return False, None
    df = calculate_daily_indicators(df)
    if df is None: return False, None
    curr = df.iloc[-1]
    
    dc_cond = (df['Close'] > df['Donchian_High_50']).iloc[-3:].any()
    bb_cond = (df['Close'] > df['BB50_UP']).iloc[-3:].any()
    mandatory = dc_cond or bb_cond
    
    vr_cond = (df['VR50'].iloc[-3:] > 110).any()
    bw_cond = (df['BW50'].iloc[-51] > curr['BW50']) if len(df)>55 else False
    macd_cond = curr['MACD_OSC_C'] > 0
    optional_count = sum([vr_cond, bw_cond, macd_cond])
    
    if mandatory and (optional_count >= 2):
        squeeze_on = df['TTM_Squeeze'].iloc[-5:].any()
        win_52 = df.iloc[-252:]
        high_52_date = win_52['Close'].idxmax().strftime('%Y-%m-%d')
        prev_win = win_52[win_52.index < win_52['Close'].idxmax()]
        prev_date = prev_win['Close'].idxmax().strftime('%Y-%m-%d') if len(prev_win)>0 else "-"
        diff_days = (win_52['Close'].idxmax() - prev_win['Close'].idxmax()).days if len(prev_win)>0 else 0
        return True, {
            'price': curr['Close'], 'atr': curr['ATR14'], 'high_date': high_52_date, 
            'prev_date': prev_date, 'diff_days': diff_days, 'bw_curr': curr['BW50'], 
            'macdv': curr['MACD_V'], 'squeeze': "🔥TTM Squeeze" if squeeze_on else "-" 
        }
    return False, None

def check_weekly_condition(df):
    if len(df) < 40: return False, None
    df = df.copy()
    df['SMA30'] = df['Close'].rolling(window=30).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI14'] = 100 - (100 / (1 + rs))

    e12 = df['Close'].ewm(span=12, adjust=False).mean()
    e26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd - sig

    sma12 = df['Close'].rolling(12).mean()
    std12 = df['Close'].rolling(12).std()
    bb_up_12 = sma12 + (2 * std12)
    
    e12_c = df['Close'].ewm(span=12, adjust=False).mean()
    e36_c = df['Close'].ewm(span=36, adjust=False).mean()
    macd_c = e12_c - e36_c
    sig_c = macd_c.ewm(span=9, adjust=False).mean()
    
    df['MACD_V'], _ = calculate_macdv(df, 12, 26, 9)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = tr.ewm(span=14, adjust=False).mean()

    curr = df.iloc[-1]
    cond_basic_1 = curr['Close'] > curr['SMA30']
    cond_basic_2 = curr['RSI14'] > 50
    if len(df) < 2: return False, None
    cond_basic_3 = (df['MACD_Hist'].iloc[-1] > df['MACD_Hist'].iloc[-2]) or (df['MACD_Hist'].iloc[-1] > 0)

    if not (cond_basic_1 and cond_basic_2 and cond_basic_3):
        return False, None

    is_strat_1 = False
    past_12w = df.iloc[-13:-1]
    if len(past_12w) > 0:
        past_breakout = (past_12w['Close'] > bb_up_12.loc[past_12w.index]).any()
        current_rest = curr['Close'] <= (bb_up_12.iloc[-1] * 1.02)
        if past_breakout and current_rest:
            recent_high = past_12w['High'].max()
            price_support = curr['Close'] >= (recent_high * 0.85)
            ema_support = curr['Close'] > curr['EMA20']
            if price_support and ema_support:
                is_strat_1 = True

    is_strat_2 = False
    prev_macd_c = macd_c.iloc[-2]
    prev_sig_c = sig_c.iloc[-2]
    curr_macd_c = macd_c.iloc[-1]
    curr_sig_c = sig_c.iloc[-1]
    if (prev_macd_c <= prev_sig_c) and (curr_macd_c > curr_sig_c):
        is_strat_2 = True

    status_list = []
    if is_strat_1: status_list.append("돌파수렴(눌림)")
    if is_strat_2: status_list.append("MACD매수")
    
    if status_list:
        final_status = " / ".join(status_list)
        return True, {
            'price': curr['Close'], 'atr': curr['ATR14'], 'bw_curr': 0, 'bw_change': final_status, 'macdv': curr['MACD_V']
        }
    return False, None

def check_monthly_condition(df):
    if len(df) < 12: return False, None
    df = df.copy()
    ath_price = df['High'].max()
    curr_price = df['Close'].iloc[-1]
    if curr_price >= ath_price * 0.90:
        ath_idx = df['High'].idxmax()
        month_count = (df['Close'] >= ath_price * 0.90).sum()
        return True, {'price': curr_price, 'ath_price': ath_price, 'ath_date': ath_idx.strftime('%Y-%m'), 'month_count': month_count}
    return False, None

def check_dual_ma_breakout(df):
    if len(df) < 250: return False, None
    df = df.copy()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['DC_High'] = df['High'].rolling(window=20).max().shift(1)

    df['Gap_Pct'] = (df['EMA20'] - df['EMA200']).abs()
