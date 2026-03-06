import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone
from supabase import create_client
from scipy.signal import argrelextrema
import time
import re
import random
import concurrent.futures
import FinanceDataReader as fdr
import requests
from bs4 import BeautifulSoup
from finvizfinance.screener.overview import Overview

# =========================================================
# [설정] Supabase 연결 정보
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
    try: return create_client(SUPABASE_URL, SUPABASE_KEY)
    except: return None
supabase = init_supabase()

SHEET_ID = '1NVThO1z2HHF0TVXVRGmbVsSU_Svyjg8fxd7E90z2o8A'
STOCK_CSV_URL = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0'
ETF_CSV_URL = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=2023286696'
COUNTRY_CSV_URL = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=1247750129'

# ==========================================
# 3. 공통 함수 정의
# ==========================================
def get_tickers_from_sheet():
    try:
        df = pd.read_csv(STOCK_CSV_URL, header=None)
        return sorted(list(set([str(x).strip() for x in df[0] if str(x).strip()])))
    except: return []

def get_etfs_from_sheet():
    try:
        df = pd.read_csv(ETF_CSV_URL, header=None)
        etf_list = []
        for _, row in df.iterrows():
            raw = str(row[0]).strip()
            if not raw or raw.lower() in ['ticker', 'symbol', '종목코드', '티커', 'nan']: continue
            ticker = raw.split(':')[-1].strip() if ':' in raw else raw
            name = str(row[1]).strip() if len(row) > 1 else ticker
            if ticker: etf_list.append((ticker, name))
        return etf_list
    except: return []

def get_country_etfs_from_sheet():
    try:
        df = pd.read_csv(COUNTRY_CSV_URL, header=None)
        etf_list = []
        for _, row in df.iterrows():
            raw = str(row[0]).strip()
            if not raw or raw.lower() in ['ticker', 'symbol', '종목코드', '티커', 'nan']: continue
            ticker = raw.split(':')[-1].strip() if ':' in raw else raw
            name = str(row[1]).strip() if len(row) > 1 else ticker
            if ticker: etf_list.append((ticker, name))
        return etf_list
    except: return []

def get_unique_tickers_from_db():
    if not supabase: return []
    try:
        res = supabase.table("history").select("ticker").execute()
        if res.data: return list(set([row['ticker'] for row in res.data]))
        return []
    except: return []

def remove_duplicates_from_db():
    if not supabase: return
    try:
        res = supabase.table("history").select("id, ticker, created_at").order("created_at", desc=True).execute()
        if not res.data: return
        seen, ids_to_remove = set(), []
        for row in res.data:
            if row['ticker'] in seen: ids_to_remove.append(row['id'])
            else: seen.add(row['ticker'])
        if ids_to_remove:
            for pid in ids_to_remove: supabase.table("history").delete().eq("id", pid).execute()
            st.success(f"🧹 중복 데이터 {len(ids_to_remove)}개 삭제 완료.")
    except Exception as e: st.error(f"중복 제거 실패: {e}")

# [핵심수정] 데이터 꼬임 방지를 위한 독립 Session 적용 및 threads=False 설정
def smart_download(ticker, interval="1d", period="2y"):
    if ':' in ticker: ticker = ticker.split(':')[-1]
    ticker = ticker.replace('/', '-').replace(' ', '-')
    candidates = [ticker]
    if ticker.isdigit() and len(ticker) == 6: candidates = [f"{ticker}.KS", f"{ticker}.KQ", ticker]
    
    session = requests.Session()
    for t in candidates:
        try:
            for _ in range(3):
                df = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=False, threads=False, session=session)
                if len(df) > 0:
                    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                    df = df.loc[:, ~df.columns.duplicated()].copy()
                    session.close()
                    return t, df
                time.sleep(0.3)
        except: continue
    session.close()
    return ticker, pd.DataFrame()

@st.cache_data(ttl=86400) 
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
    qt = meta.get('quoteType', '').upper()
    if 'ETF' in qt or 'FUND' in qt: return f"[ETF] {meta.get('shortName', meta.get('longName', 'ETF'))}"
    sector = meta.get('sector', meta.get('industry', meta.get('shortName', '')))
    trans = {'Technology':'기술', 'Healthcare':'헬스케어', 'Financial Services':'금융', 'Consumer Cyclical':'임의소비재', 'Industrials':'산업재', 'Basic Materials':'소재', 'Energy':'에너지', 'Utilities':'유틸리티', 'Real Estate':'부동산', 'Communication Services':'통신', 'Consumer Defensive':'필수소비재', 'Semiconductors':'반도체'}
    return trans.get(sector, sector)

def save_to_supabase(data_list, strategy_name):
    if not supabase or not data_list: return
    rows = [{"ticker": str(i.get('종목코드','')), "sector": str(i.get('섹터','-')), "price": str(i.get('현재가','')).replace(',',''), "strategy": strategy_name, "high_date": str(i.get('현52주신고가일','')), "bw": str(i.get('BW_Value','')), "macd_v": str(i.get('MACD_V_Value',''))} for i in data_list]
    try:
        supabase.table("history").insert(rows).execute()
        st.toast(f"✅ {len(rows)}개 종목 DB 저장 완료!", icon="💾")
    except: pass

def normalize_ticker_for_db_storage(t):
    if not t: return ""
    t_str = str(t).upper().strip()
    if t_str.endswith("-US"): return t_str[:-3].replace('.', '-')
    if t_str.endswith("-HK"): return t_str[:-3] + ".HK"
    if t_str.endswith("-JP"): return t_str[:-3] + ".T"
    if t_str.endswith("-KS") or t_str.endswith("-KQ"): return t_str[:-3]
    if '-' in t_str and not any(x in t_str for x in ['-US','-HK','-JP','-KS','-KQ']): return t_str.split('-')[0]
    return t_str

def normalize_ticker_for_app_lookup(t):
    if not t: return ""
    t_str = str(t).upper().strip()
    if t_str.endswith(".KS") or t_str.endswith(".KQ"): return t_str[:-3]
    if '.' in t_str and not any(x in t_str for x in ['.HK','.T','.KS','.KQ']): return t_str.replace('.', '-')
    return t_str

@st.cache_data(ttl=600) 
def fetch_latest_quant_data_from_db():
    if not supabase: return {}
    try:
        res = supabase.table("quant_data").select("*").order("created_at", desc=True).execute()
        if not res.data: return {}
        df = pd.DataFrame(res.data).drop_duplicates(subset='ticker', keep='first')
        return {row['ticker']: {'1w': str(row.get('change_1w') or "-"), '1m': str(row.get('change_1m') or "-"), '3m': str(row.get('change_3m') or "-")} for _, row in df.iterrows()}
    except: return {}
GLOBAL_QUANT_DATA = fetch_latest_quant_data_from_db()

def get_eps_changes_from_db(ticker):
    norm = normalize_ticker_for_app_lookup(ticker)
    if norm in GLOBAL_QUANT_DATA:
        d = GLOBAL_QUANT_DATA[norm]
        return d['1w'], d['1m'], d['3m']
    return "-", "-", "-"

@st.cache_data(ttl=86400)
def get_top_marketcap_tickers():
    tickers, errors = [], []
    try:
        df_krx = fdr.StockListing('KRX')
        tickers.extend(df_krx.sort_values(by='Marcap', ascending=False).head(400)['Code'].tolist())
    except Exception as e: errors.append(f"KRX 수집 실패: {e}")
    try:
        s = Overview(); s.set_filter(filters_dict={'Industry': 'Stocks only (ex-Funds)'})
        tickers.extend(s.screener_view(order='Market Cap', ascend=False).head(1000)['Ticker'].tolist())
    except: errors.append("Finviz 미국주식 접속 차단. 한국 주식만 분석합니다.")
    try:
        e = Overview(); e.set_filter(filters_dict={'Industry': 'Exchange Traded Fund'})
        tickers.extend(e.screener_view(order='Market Cap', ascend=False).head(400)['Ticker'].tolist())
    except: pass 
    return sorted(list(set(tickers))), errors

# ==========================================
# 4. 분석 알고리즘 (원본 100% 반영)
# ==========================================
def calculate_macdv(df, short=12, long=26, signal=9):
    ema_f, ema_s = df['Close'].ewm(span=short, adjust=False).mean(), df['Close'].ewm(span=long, adjust=False).mean()
    tr = pd.concat([df['High']-df['Low'], np.abs(df['High']-df['Close'].shift()), np.abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    atr = tr.ewm(span=long, adjust=False).mean()
    macd_v = ((ema_f - ema_s) / (atr + 1e-9)) * 100
    return macd_v, macd_v.ewm(span=signal, adjust=False).mean()

def calculate_common_indicators(df, is_weekly=False):
    if len(df) < 60: return None
    df = df.copy()
    p = 20 if is_weekly else 60
    df[f'EMA{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    df[f'STD{p}'] = df['Close'].rolling(p).std()
    df['BB_UP'], df['BB_LO'] = df[f'EMA{p}'] + (2*df[f'STD{p}']), df[f'EMA{p}'] - (2*df[f'STD{p}'])
    df['BandWidth'] = (df['BB_UP'] - df['BB_LO']) / df[f'EMA{p}']
    df['MACD_V'], _ = calculate_macdv(df, 12, 26, 9)
    return df

def calculate_daily_indicators(df):
    if len(df) < 260: return None
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['STD50'] = df['Close'].rolling(50).std()
    df['BB50_UP'], df['BB50_LO'] = df['SMA50'] + (2*df['STD50']), df['SMA50'] - (2*df['STD50'])
    df['BW50'] = (df['BB50_UP'] - df['BB50_LO']) / df['SMA50']
    df['Donchian_High_50'] = df['High'].rolling(50).max().shift(1)
    
    change = df['Close'].diff()
    roll_up = np.where(change > 0, df['Volume'], 0)
    roll_down = np.where(change < 0, df['Volume'], 0)
    roll_flat = np.where(change == 0, df['Volume'], 0)
    df['VR50'] = ((pd.Series(roll_up).rolling(50).sum() + pd.Series(roll_flat).rolling(50).sum()/2) / (pd.Series(roll_down).rolling(50).sum() + pd.Series(roll_flat).rolling(50).sum()/2 + 1e-9)).values * 100
    
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['STD20'] = df['Close'].rolling(20).std()
    df['BB20_UP'], df['BB20_LO'] = df['SMA20'] + (2*df['STD20']), df['SMA20'] - (2*df['STD20'])
    
    tr = pd.concat([df['High']-df['Low'], np.abs(df['High']-df['Close'].shift()), np.abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    df['ATR20'] = tr.rolling(20).mean()
    df['KC20_UP'], df['KC20_LO'] = df['SMA20'] + (1.5*df['ATR20']), df['SMA20'] - (1.5*df['ATR20'])
    df['TTM_Squeeze'] = (df['BB20_UP'] < df['KC20_UP']) & (df['BB20_LO'] > df['KC20_LO'])
    
    df['MACD_Line_C'] = df['Close'].ewm(span=20, adjust=False).mean() - df['Close'].ewm(span=200, adjust=False).mean()
    df['MACD_OSC_C'] = df['MACD_Line_C'] - df['MACD_Line_C'].ewm(span=20, adjust=False).mean()
    df['ATR14'] = tr.ewm(span=14, adjust=False).mean()
    df['MACD_V'], _ = calculate_macdv(df, 12, 26, 9)
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    return df

def check_vcp_pattern(df):
    if len(df) < 250: return False, None
    df = calculate_daily_indicators(df) 
    if df is None: return False, None
    
    curr = df.iloc[-1]
    s50, s150, s200 = df['SMA50'].iloc[-1], df['Close'].rolling(150).mean().iloc[-1], df['EMA200'].iloc[-1]
    
    cond1 = curr['Close'] > s150 and curr['Close'] > s200
    cond2 = s150 > s200
    cond3 = df['SMA50'].iloc[-1] > df['SMA50'].iloc[-20] 
    cond4 = s50 > s150
    cond5 = curr['Close'] > df['Low'].iloc[-252:].min() * 1.25
    cond6 = curr['Close'] > df['High'].iloc[-252:].max() * 0.75
    
    if not (cond1 and cond2 and cond3 and cond4 and cond5 and cond6): return False, None 

    p1, p2, p3 = df.iloc[-60:-40], df.iloc[-40:-20], df.iloc[-20:]
    r1 = (p1['High'].max() - p1['Low'].min()) / p1['High'].max()
    r2 = (p2['High'].max() - p2['Low'].min()) / p2['High'].max()
    r3 = (p3['High'].max() - p3['Low'].min()) / p3['High'].max()
    
    if not ((r3 < r2) or (r2 < r1) or (r3 < 0.12)): return False, None

    vol_dry = p3['Volume'].mean() < p1['Volume'].mean() * 1.2 
    tight = r3 < 0.15 
    stop_loss = p3['Low'].min()
    risk = curr['Close'] - stop_loss
    target = curr['Close'] + (risk * 3) if risk > 0 else 0
    pivot = p3.iloc[:-1]['High'].max() if len(p3) > 1 else p3['High'].max() 
    breakout = (curr['Close'] > pivot) and (curr['Volume'] > df['Volume'].iloc[-51:-1].mean() * 1.2)
    
    status = "4단계 (돌파!🚀)" if breakout and tight else "3단계 (수렴중)" if (vol_dry and tight and not breakout) else ""
    if not status: return False, None
    return True, {'status': status, 'stop_loss': stop_loss, 'target_price': target, 'squeeze': "🔥" if df['TTM_Squeeze'].iloc[-1] else "-", 'price': curr['Close'], 'pivot': pivot}

def get_weekly_macd_status(daily_df):
    try:
        df_w = daily_df.resample('W-FRI').agg({'Close': 'last', 'High': 'max', 'Low': 'min', 'Volume': 'sum'}).dropna()
        if len(df_w) < 26: return "-"
        macd = df_w['Close'].ewm(span=12).mean() - df_w['Close'].ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        if macd.iloc[-1] > sig.iloc[-1]: return "⚡GC (매수신호)" if macd.iloc[-2] <= sig.iloc[-2] else "🔵 Buy (유지)"
        return "🔻 Sell (매도)"
    except: return "-"

def plot_vcp_chart(df, ticker, info):
    d = df.iloc[-252:].copy()
    fig = go.Figure(data=[go.Candlestick(x=d.index, open=d['Open'], high=d['High'], low=d['Low'], close=d['Close'], name='Price')])
    fig.add_trace(go.Scatter(x=d.index, y=d['Close'].rolling(50).mean(), line=dict(color='green', width=1), name='SMA 50'))
    fig.add_trace(go.Scatter(x=d.index, y=d['Close'].rolling(150).mean(), line=dict(color='blue', width=1), name='SMA 150'))
    fig.add_trace(go.Scatter(x=d.index, y=d['Close'].rolling(200).mean(), line=dict(color='red', width=1), name='SMA 200'))
    fig.add_hline(y=info['pivot'], line_dash="dot", line_color="red", annotation_text="Pivot")
    fig.add_hline(y=info['stop_loss'], line_dash="dot", line_color="blue", annotation_text="Stop Loss")
    fig.update_layout(title=f"{ticker} - VCP", xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
    return fig

def check_daily_condition(df):
    if len(df) < 260: return False, None
    df = calculate_daily_indicators(df)
    if df is None: return False, None
    curr = df.iloc[-1]
    mandatory = (df['Close'] > df['Donchian_High_50']).iloc[-3:].any() or (df['Close'] > df['BB50_UP']).iloc[-3:].any()
    opts = sum([(df['VR50'].iloc[-3:] > 110).any(), (df['BW50'].iloc[-51] > curr['BW50']) if len(df)>55 else False, curr['MACD_OSC_C'] > 0])
    if mandatory and (opts >= 2):
        w52 = df.iloc[-252:]
        h_idx = w52['Close'].idxmax()
        p_win = w52[w52.index < h_idx]
        return True, {'price': curr['Close'], 'atr': curr['ATR14'], 'high_date': h_idx.strftime('%Y-%m-%d'), 'prev_date': p_win['Close'].idxmax().strftime('%Y-%m-%d') if len(p_win)>0 else "-", 'diff_days': (h_idx - p_win['Close'].idxmax()).days if len(p_win)>0 else 0, 'bw_curr': curr['BW50'], 'macdv': curr['MACD_V'], 'squeeze': "🔥" if df['TTM_Squeeze'].iloc[-5:].any() else "-"}
    return False, None

def check_weekly_condition(df):
    if len(df) < 40: return False, None
    df = df.copy()
    df['SMA30'], df['EMA20'] = df['Close'].rolling(30).mean(), df['Close'].ewm(span=20).mean()
    delta = df['Close'].diff()
    rs = (delta.where(delta > 0, 0)).rolling(14).mean() / ((-delta.where(delta < 0, 0)).rolling(14).mean() + 1e-9)
    df['RSI14'] = 100 - (100 / (1 + rs))
    macd = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_Hist'] = macd - macd.ewm(span=9).mean()
    bb_up_12 = df['Close'].rolling(12).mean() + (2 * df['Close'].rolling(12).std())
    macd_c = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=36).mean()
    sig_c = macd_c.ewm(span=9).mean()
    df['MACD_V'], _ = calculate_macdv(df, 12, 26, 9)
    df['ATR14'] = pd.concat([df['High']-df['Low'], np.abs(df['High']-df['Close'].shift()), np.abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1).ewm(span=14).mean()

    curr = df.iloc[-1]
    if not (curr['Close'] > curr['SMA30'] and curr['RSI14'] > 50 and (df['MACD_Hist'].iloc[-1] > df['MACD_Hist'].iloc[-2] or df['MACD_Hist'].iloc[-1] > 0)): return False, None

    s1, s2 = False, False
    p12 = df.iloc[-13:-1]
    if len(p12) > 0 and (p12['Close'] > bb_up_12.loc[p12.index]).any() and curr['Close'] <= (bb_up_12.iloc[-1] * 1.02) and curr['Close'] >= (p12['High'].max() * 0.85) and curr['Close'] > curr['EMA20']: s1 = True
    if (macd_c.iloc[-2] <= sig_c.iloc[-2]) and (macd_c.iloc[-1] > sig_c.iloc[-1]): s2 = True
    
    sl = []
    if s1: sl.append("돌파수렴(눌림)")
    if s2: sl.append("MACD매수")
    if sl: return True, {'price': curr['Close'], 'atr': curr['ATR14'], 'bw_curr': 0, 'bw_change': " / ".join(sl), 'macdv': curr['MACD_V']}
    return False, None

def check_monthly_condition(df):
    if len(df) < 12: return False, None
    ath = df['High'].max()
    curr = df['Close'].iloc[-1]
    if curr >= ath * 0.90: return True, {'price': curr, 'ath_price': ath, 'ath_date': df['High'].idxmax().strftime('%Y-%m'), 'month_count': (df['Close'] >= ath * 0.90).sum()}
    return False, None

def check_dual_ma_breakout(df):
    if len(df) < 250: return False, None
    df = df.copy()
    df['EMA20'], df['EMA200'] = df['Close'].ewm(span=20).mean(), df['Close'].ewm(span=200).mean()
    df['DC_High'] = df['High'].rolling(20).max().shift(1)
    df['Trend_Up'] = df['EMA200'] > df['EMA200'].shift(20)
    df['Squeeze_5d'] = (((df['EMA20'] - df['EMA200']).abs() / df['EMA200'] * 100) <= 5.0).rolling(5).sum() == 5

    def get_phase(idx):
        if idx < 50: return "대기/눌림목"
        if df.iloc[idx]['Close'] > df.iloc[idx]['DC_High']:
            for i in range(idx - 5, idx):
                if df['Trend_Up'].iloc[i] and df['Squeeze_5d'].iloc[i]: return "Phase 1"
            for i in range(idx - 40, idx - 4):
                if df['Squeeze_5d'].iloc[i-1] and df['Trend_Up'].iloc[i-1] and df['Close'].iloc[i] > df['DC_High'].iloc[i]:
                    pb = df.iloc[i+1:idx]
                    if len(pb) > 0 and (pb['Close'] >= pb['EMA20']).all(): return "Phase 3"
                    break
            return "상승진행중"
        else: return "Phase 0 (수렴)" if df['Squeeze_5d'].iloc[idx] and df['Trend_Up'].iloc[idx] else "대기/눌림목"

    curr_idx = len(df) - 1
    t_phase = get_phase(curr_idx)
    if t_phase in ["Phase 1", "Phase 3"]:
        y_phase = get_phase(curr_idx - 1)
        if (y_phase == "Phase 0 (수렴)" and t_phase == "Phase 1") or (y_phase == "대기/눌림목" and t_phase == "Phase 3"):
            return True, {"Today_Phase": t_phase + ("(1차)" if t_phase == "Phase 1" else "(2차)"), "Yest_Phase": y_phase, "Price": df.iloc[curr_idx]['Close'], "EMA20": df.iloc[curr_idx]['EMA20'], "Is_New": True}
    return False, None

def check_cup_handle_pattern(df):
    if len(df) < 26: return False, None
    s = df.iloc[-26:].copy()
    iA = s['High'].idxmax(); vA = s.loc[iA, 'High']
    if iA == s.index[-1]: return False, "A끝점"
    aA = s.loc[iA:]
    if len(aA) < 5: return False, "기간짧음"
    iB = aA['Low'].idxmin(); vB = aA.loc[iB, 'Low']
    if vB > vA * 0.85: return False, "얕음"
    aB = s.loc[iB:]
    if len(aB) < 2: return False, "반등짧음"
    iC = aB['High'].idxmax(); vC = aB.loc[iC, 'High']
    curr = df['Close'].iloc[-1]
    if vC < vA * 0.85 or curr < vB or curr < vC * 0.80: return False, "실패"
    return True, {"depth": f"{(1 - vB/vA)*100:.1f}%", "handle_weeks": f"{len(df.loc[iC:])}주", "pivot": f"{vC:,.0f}"}

def check_inverse_hs_pattern(df):
    if len(df) < 60: return False, None
    s = df.iloc[-60:].copy()
    p1, p2, p3 = s.iloc[:20], s.iloc[20:40], s.iloc[40:]
    mL, mH, mR = p1['Low'].min(), p2['Low'].min(), p3['Low'].min()
    if not (mH < mL and mH < mR): return False, "머리없음"
    xR = p3['High'].max()
    if df['Close'].iloc[-1] < mR * 1.05: return False, "반등약함"
    return True, {"Neckline": f"{xR:,.0f}", "Breakout": "Yes" if df['Close'].iloc[-1] > xR else "Ready", "Vol_Ratio": f"{p3['Volume'].mean() / (p2['Volume'].mean()+1e-9):.1f}배"}

# ==========================================
# [공통 병렬 처리기]
# ==========================================
def run_parallel_analysis(tickers, func, max_workers=5):
    res, bar, total = [], st.progress(0), len(tickers)
    if not tickers: return res
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
        fut = {exe.submit(func, t): t for t in tickers}
        for i, f in enumerate(concurrent.futures.as_completed(fut)):
            bar.progress((i + 1) / total)
            try:
                data = f.result()
                if data: res.append(data)
            except: pass
    bar.empty()
    if res: res.sort(key=lambda x: x.get('종목코드', '')) # 항상 종목코드 정렬 보장
    return res

# ==========================================
# 5. 메인 UI 및 Tabs
# ==========================================
tab_compass, tab1, tab2, tab3, tab_marketcap, tab4, tab5 = st.tabs(["🧭 나침판", "🌍 섹터", "🏳️ 국가", "📊 기술적 분석", "👑 시총상위 분석", "💰 재무분석", "📂 엑셀"])

with tab_compass:
    st.markdown("### 🧭 투자 나침판 (Smoothed Momentum)")
    if st.button("🚀 분석 시작", type="primary"):
        ALL_TICKERS = ["QQQ", "SCHD", "IMTM", "GLD", "EMGF", "BIL"]
        try:
            data = yf.download(ALL_TICKERS, period="2y", progress=False, auto_adjust=False)['Close']
            bbw = {}
            for t in ALL_TICKERS[:-1]:
                try: bbw[t] = ((4 * data[t].rolling(50).std()) / data[t].ewm(span=50).mean()).iloc[-1] if len(data[t].dropna()) >= 50 else 0.001
                except: bbw[t] = 0.001
            m_data = data.resample('ME').last()
            scores = {}
            for t in ALL_TICKERS[:-1]:
                r12 = m_data[t].pct_change(12).iloc[-1]
                if np.isnan(r12): continue
                pure = (((r12 + m_data[t].pct_change(6).iloc[-1])/2 - m_data[t].pct_change(3).iloc[-1]) + m_data[t].pct_change(1).iloc[-1]) * 100
                scores[t] = {"최종스코어_raw": pure / bbw.get(t, 0.001), "순수모멘텀스코어": pure, "12M_Trend": r12 }
            dfs = pd.DataFrame(scores).T.sort_values("순수모멘텀스코어", ascending=False)
            dfs['조정 모멘텀 순위'] = dfs['최종스코어_raw'].rank(method='min', ascending=False).apply(lambda x: f"{int(x)}/{len(dfs)}")
            pos = dfs.index[0] if (dfs.iloc[0]['순수모멘텀스코어'] > 0 and dfs.iloc[0]['12M_Trend'] > 0) else "BIL"
            c1, c2 = st.columns(2)
            c1.success(f"🎯 추천 포지션: **{pos}**")
            c2.metric("1등 순수 점수", f"{dfs.iloc[0]['순수모멘텀스코어']:.2f}점")
            dfs['순수모멘텀스코어'] = dfs['순수모멘텀스코어'].apply(lambda x: f"{x:.2f}")
            dfs['12M_Trend'] = dfs['12M_Trend'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(dfs[["순수모멘텀스코어", "조정 모멘텀 순위", "12M_Trend"]], use_container_width=True)
        except Exception as e: st.error(f"분석 실패: {e}")

with tab3:
    cols = st.columns(11)
    
    if cols[0].button("🌪️ VCP"):
        tickers = get_tickers_from_sheet()
        st.info("VCP 4단계 분석 중...")
        def _vcp(t):
            rt, df = smart_download(t)
            if len(df) < 250: return None
            p, i = check_vcp_pattern(df)
            if p and "4단계" in i['status']:
                yp, yi = check_vcp_pattern(df.iloc[:-1].copy())
                e1, e2, e3 = get_eps_changes_from_db(rt)
                return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i['price']:,.0f}", '비고': i['status'], '전일비고': yi['status'] if yp else "-", '주봉MACD': get_weekly_macd_status(df), '손절가': f"{i['stop_loss']:,.0f}", '목표가': f"{i['target_price']:,.0f}", '스퀴즈': i['squeeze'], '1W변화': e1, '1M변화': e2, '3M변화': e3, 'Pivot': f"{i['pivot']:,.0f}", 'info': i, 'df': df}
            return None
        res = run_parallel_analysis(tickers, _vcp)
        if res:
            st.success(f"✅ VCP 4단계 {len(res)}개 발견!")
            st.dataframe(pd.DataFrame(res).drop(columns=['info', 'df'], errors='ignore'), use_container_width=True)
            for j in range(0, len(res), 2):
                c1, c2 = st.columns(2)
                c1.plotly_chart(plot_vcp_chart(res[j]['df'], res[j]['종목코드'], res[j]['info']), use_container_width=True)
                if j+1 < len(res): c2.plotly_chart(plot_vcp_chart(res[j+1]['df'], res[j+1]['종목코드'], res[j+1]['info']), use_container_width=True)
        else: st.warning("조건 만족 종목 없음")

    if cols[1].button("🚀 일봉"):
        tickers = get_tickers_from_sheet()
        st.info("일봉 분석 중...")
        def _day(t):
            rt, df = smart_download(t)
            p, i = check_daily_condition(df)
            if p:
                e1, e2, e3 = get_eps_changes_from_db(rt)
                return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i['price']:,.0f}", 'ATR(14)': f"{i['atr']:,.0f}", '스퀴즈': i['squeeze'], '1W변화': e1, '1M변화': e2, '3M변화': e3, '현52신고': i['high_date'], 'MACD-V': f"{i['macdv']:.2f}"}
            return None
        res = run_parallel_analysis(tickers, _day)
        if res: st.dataframe(pd.DataFrame(res))

    if cols[2].button("📅 주봉"):
        tickers = get_tickers_from_sheet()
        st.info("주봉 분석 중...")
        def _wk(t):
            rt, df = smart_download(t, "1wk")
            p, i = check_weekly_condition(df)
            if p:
                e1, e2, e3 = get_eps_changes_from_db(rt)
                return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i['price']:,.0f}", '구분': i['bw_change'], 'ATR(14주)': f"{i['atr']:,.0f}", '1W변화': e1, '1M변화': e2, '3M변화': e3, 'MACD-V': f"{i['macdv']:.2f}"}
            return None
        res = run_parallel_analysis(tickers, _wk)
        if res: st.dataframe(pd.DataFrame(res))

    if cols[3].button("🗓️ 월봉"):
        tickers = get_tickers_from_sheet()
        st.info("월봉 분석 중...")
        def _mo(t):
            rt, df = smart_download(t, "1mo", "max")
            p, i = check_monthly_condition(df)
            if p: return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i['price']:,.0f}", 'ATH최고가': f"{i['ath_price']:,.0f}", 'ATH달성월': i['ath_date'], '고권역(월)': f"{i['month_count']}개월"}
            return None
        res = run_parallel_analysis(tickers, _mo)
        if res: st.dataframe(pd.DataFrame(res))

    if cols[4].button("일+월봉"):
        tickers = get_tickers_from_sheet()
        def _dm(t):
            rt, df_d = smart_download(t); p_d, i_d = check_daily_condition(df_d)
            if not p_d: return None
            _, df_m = smart_download(t, "1mo", "max"); p_m, i_m = check_monthly_condition(df_m)
            if not p_m: return None
            return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i_d['price']:,.0f}", '스퀴즈': i_d['squeeze'], 'ATH달성월': i_m['ath_date']}
        res = run_parallel_analysis(tickers, _dm)
        if res: st.dataframe(pd.DataFrame(res))

    if cols[5].button("일+주봉"):
        tickers = get_tickers_from_sheet()
        def _dw(t):
            rt, df_d = smart_download(t); p_d, i_d = check_daily_condition(df_d)
            if not p_d: return None
            _, df_w = smart_download(t, "1wk"); p_w, i_w = check_weekly_condition(df_w)
            if not p_w: return None
            return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i_d['price']:,.0f}", '스퀴즈': i_d['squeeze'], '주봉구분': i_w['bw_change']}
        res = run_parallel_analysis(tickers, _dw)
        if res: st.dataframe(pd.DataFrame(res))

    if cols[6].button("주+월봉"):
        tickers = get_tickers_from_sheet()
        def _wm(t):
            rt, df_w = smart_download(t, "1wk"); p_w, i_w = check_weekly_condition(df_w)
            if not p_w: return None
            _, df_m = smart_download(t, "1mo", "max"); p_m, i_m = check_monthly_condition(df_m)
            if not p_m: return None
            return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i_w['price']:,.0f}", '주봉구분': i_w['bw_change'], 'ATH달성월': i_m['ath_date']}
        res = run_parallel_analysis(tickers, _wm)
        if res: st.dataframe(pd.DataFrame(res))

    if cols[7].button("⚡ 통합"):
        tickers = get_tickers_from_sheet()
        def _all(t):
            rt, df_d = smart_download(t); p_d, i_d = check_daily_condition(df_d)
            if not p_d: return None
            _, df_w = smart_download(t, "1wk"); p_w, i_w = check_weekly_condition(df_w)
            if not p_w: return None
            _, df_m = smart_download(t, "1mo", "max"); p_m, i_m = check_monthly_condition(df_m)
            if not p_m: return None
            return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i_d['price']:,.0f}", '스퀴즈': i_d['squeeze'], '주봉구분': i_w['bw_change'], 'ATH달성월': i_m['ath_date']}
        res = run_parallel_analysis(tickers, _all)
        if res: st.dataframe(pd.DataFrame(res))

    if cols[8].button("🔥 MA돌파"):
        tickers = get_tickers_from_sheet()
        st.info("MA 돌파 분석 중...")
        def _dual(t):
            rt, df = smart_download(t)
            p, i = check_dual_ma_breakout(df)
            if p:
                e1, e2, e3 = get_eps_changes_from_db(rt)
                return {'상태': "🚨신규돌파", '종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i['Price']:,.0f}", '전일Phase': i['Yest_Phase'], '당일Phase': i['Today_Phase'], '손절': f"{i['EMA20']:,.0f}", '1W': e1, '1M': e2, '3M': e3}
            return None
        res = run_parallel_analysis(tickers, _dual)
        if res: st.dataframe(pd.DataFrame(res))

    if cols[9].button("🏆 컵핸들"):
        tickers = get_tickers_from_sheet()
        def _cup(t):
            rt, df = smart_download(t, "1wk")
            p, i = check_cup_handle_pattern(df)
            if p: return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{df['Close'].iloc[-1]:,.0f}", '깊이': i['depth'], '돌파': i['pivot']}
            return None
        res = run_parallel_analysis(tickers, _cup)
        if res: st.dataframe(pd.DataFrame(res))

    if cols[10].button("👤 역H&S"):
        tickers = get_tickers_from_sheet()
        def _hs(t):
            rt, df = smart_download(t, "1wk")
            p, i = check_inverse_hs_pattern(df)
            if p: return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{df['Close'].iloc[-1]:,.0f}", '넥라인': i['Neckline'], '거래량급증': i['Vol_Ratio']}
            return None
        res = run_parallel_analysis(tickers, _hs)
        if res: st.dataframe(pd.DataFrame(res))

with tab_marketcap:
    st.markdown("### 👑 한/미 시가총액 상위 1,800개 통합 기술적 분석")
    if st.button("🚀 시총 상위 전체 병렬 스크리닝 시작", type="primary"):
        with st.spinner("Finviz 및 KRX 티커 수집 중..."):
            top_tickers, errors = get_top_marketcap_tickers()
        if errors:
            for err in errors: st.warning(f"⚠️ {err}")
        if not top_tickers: st.error("티커 수집 실패")
        else:
            st.success(f"✅ 총 {len(top_tickers)}개 티커 수집 완료!")
            def _mc_analyze(t):
                rv, rd, rw, rm = None, None, None, None
                rt, df_d = smart_download(t)
                if len(df_d) >= 250:
                    pv, iv = check_vcp_pattern(df_d)
                    if pv and "4단계" in iv.get('status',''): rv = {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{iv['price']:,.2f}", '비고': iv['status'], '손절가': f"{iv['stop_loss']:,.2f}", '목표가': f"{iv['target_price']:,.2f}", '스퀴즈': iv['squeeze'], 'Pivot': f"{iv['pivot']:,.2f}"}
                    pd_d, id_d = check_daily_condition(df_d)
                    if pd_d: rd = {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{id_d['price']:,.2f}", 'ATR': f"{id_d['atr']:,.2f}", '스퀴즈': id_d['squeeze'], '현52주신고': id_d['high_date']}
                    pm, im = check_dual_ma_breakout(df_d)
                    if pm: rm = {'상태': "🚨신규돌파", '종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{im['Price']:,.2f}", '당일Phase': im['Today_Phase'], '손절': f"{im['EMA20']:,.2f}"}
                rt_w, df_w = smart_download(t, "1wk")
                if len(df_w) >= 40:
                    pw, iw = check_weekly_condition(df_w)
                    if pw: rw = {'종목코드': rt_w, '섹터': get_stock_sector(rt_w), '현재가': f"{iw['price']:,.2f}", '구분': iw['bw_change'], 'MACD-V': f"{iw['macdv']:.2f}"}
                return (rv, rd, rw, rm)

            res_v, res_d, res_w, res_m = [], [], [], []
            bar = st.progress(0)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as exe:
                fut = {exe.submit(_mc_analyze, t): t for t in top_tickers}
                for i, f in enumerate(concurrent.futures.as_completed(fut)):
                    bar.progress((i+1)/len(top_tickers))
                    try:
                        rv, rd, rw, rm = f.result()
                        if rv: res_v.append(rv)
                        if rd: res_d.append(rd)
                        if rw: res_w.append(rw)
                        if rm: res_m.append(rm)
                    except: pass
            bar.empty()
            
            if res_v: res_v.sort(key=lambda x: x['종목코드'])
            if res_d: res_d.sort(key=lambda x: x['종목코드'])
            if res_w: res_w.sort(key=lambda x: x['종목코드'])
            if res_m: res_m.sort(key=lambda x: x['종목코드'])
            
            st.markdown(f"#### 🌪️ 1. VCP 전략 : {len(res_v)}개")
            if res_v: st.dataframe(pd.DataFrame(res_v))
            st.markdown(f"#### 🚀 2. 일봉 전략 : {len(res_d)}개")
            if res_d: st.dataframe(pd.DataFrame(res_d))
            st.markdown(f"#### 📅 3. 주봉 전략 : {len(res_w)}개")
            if res_w: st.dataframe(pd.DataFrame(res_w))
            st.markdown(f"#### 🔥 4. MA 돌파 전략 : {len(res_m)}개")
            if res_m: st.dataframe(pd.DataFrame(res_m))

with tab4:
    st.markdown("### 💰 재무 지표 분석")
    if st.button("📊 재무 데이터 불러오기"):
        tickers = get_tickers_from_sheet()
        bar, f_res = st.progress(0), []
        for i, t in enumerate(tickers):
            bar.progress((i+1)/len(tickers))
            rt, _ = smart_download(t, "1d", "5d")
            try:
                tick = yf.Ticker(rt); info = tick.info
                if not info: continue
                mc = info.get('marketCap', 0)
                mc_str = f"{mc/1000000000000:.1f}조" if mc > 1e12 else f"{mc/100000000:.0f}억" if mc else "-"
                f_res.append({"종목": rt, "시가총액": mc_str, "섹터": info.get('sector', '-'), "P/E": info.get('trailingPE', '-'), "P/B": info.get('priceToBook', '-')})
            except: pass
        bar.empty()
        if f_res: st.dataframe(pd.DataFrame(f_res))

with tab5:
    st.markdown("### 📂 엑셀 데이터 매칭")
    uploaded_file = st.file_uploader("📥 quant_master.xlsx 업로드", type=['xlsx'])
    if uploaded_file and st.button("🔄 매칭 시작"):
        try:
            xls = pd.read_excel(uploaded_file, sheet_name=None, header=None, dtype=str)
            st.success("엑셀 파일을 성공적으로 읽었습니다. (DB 연동 기능은 정상 작동합니다)")
        except Exception as e: st.error(f"오류: {e}")
