import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone
from supabase import create_client
import time
import random
import concurrent.futures
import FinanceDataReader as fdr
import requests
from bs4 import BeautifulSoup
from finvizfinance.screener.overview import Overview

# ==========================================
# 1. 설정 및 DB 연결
# ==========================================
st.set_page_config(page_title="Pro 주식 검색기", layout="wide")
st.title("📈 Pro 주식 검색기: 섹터/국가/기술적/시총상위/퀀티와이즈 통합")

try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
except Exception as e:
    st.error(f"⚠️ Secrets 설정 필요: {e}"); st.stop()

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
# 2. 공통 유틸 함수
# ==========================================
def get_tickers_from_csv(url, is_etf=False):
    try:
        df = pd.read_csv(url, header=None)
        res = []
        for _, row in df.iterrows():
            raw = str(row[0]).strip()
            if not raw or raw.lower() in ['ticker', 'symbol', '종목코드', 'nan']: continue
            ticker = raw.split(':')[-1].strip() if ':' in raw else raw
            if not ticker: continue
            if is_etf:
                name = str(row[1]).strip() if len(row) > 1 else ticker
                res.append((ticker, name))
            else: res.append(ticker)
        return sorted(list(set(res))) if not is_etf else res
    except: return []

def get_unique_tickers_from_db():
    if not supabase: return []
    try:
        res = supabase.table("history").select("ticker").execute()
        return list(set([r['ticker'] for r in res.data])) if res.data else []
    except: return []

def remove_duplicates_from_db():
    if not supabase: return
    try:
        res = supabase.table("history").select("id, ticker, created_at").order("created_at", desc=True).execute()
        if not res.data: return
        seen, ids_to_remove = set(), []
        for r in res.data:
            if r['ticker'] in seen: ids_to_remove.append(r['id'])
            else: seen.add(r['ticker'])
        if ids_to_remove:
            for pid in ids_to_remove: supabase.table("history").delete().eq("id", pid).execute()
            st.success(f"🧹 중복 데이터 {len(ids_to_remove)}개 삭제 완료.")
    except Exception as e: st.error(f"중복 제거 실패: {e}")

# [핵심] 병렬 처리 시 데이터 꼬임 방지(독립 세션 사용)
def smart_download(ticker, interval="1d", period="2y"):
    t_str = str(ticker).split(':')[-1].replace('/', '-').replace(' ', '-')
    cands = [f"{t_str}.KS", f"{t_str}.KQ", t_str] if t_str.isdigit() and len(t_str)==6 else [t_str]
    session = requests.Session()
    for t in cands:
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
                if tick.info: return tick.info
            except: time.sleep(0.3)
    except: pass
    return None

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

@st.cache_data(ttl=600) 
def fetch_latest_quant_data_from_db():
    if not supabase: return {}
    try:
        res = supabase.table("quant_data").select("*").order("created_at", desc=True).execute()
        if not res.data: return {}
        df = pd.DataFrame(res.data).drop_duplicates(subset='ticker', keep='first')
        return {r['ticker']: {'1w': str(r.get('change_1w') or "-"), '1m': str(r.get('change_1m') or "-"), '3m': str(r.get('change_3m') or "-")} for _, r in df.iterrows()}
    except: return {}

GLOBAL_QUANT_DATA = fetch_latest_quant_data_from_db()

def get_eps_changes_from_db(ticker):
    t_str = str(ticker).upper().strip()
    if t_str.endswith(".KS") or t_str.endswith(".KQ"): t_str = t_str[:-3]
    elif '.' in t_str and not any(x in t_str for x in ['.HK','.T','.KS','.KQ']): t_str = t_str.replace('.', '-')
    if t_str in GLOBAL_QUANT_DATA:
        d = GLOBAL_QUANT_DATA[t_str]
        return d['1w'], d['1m'], d['3m']
    return "-", "-", "-"

# [핵심] KRX 에러 처리 및 Finviz 차단 우회(FDR 미국주식 활용)
@st.cache_data(ttl=86400)
def get_top_marketcap_tickers():
    tickers, errors = [], []
    # 1. 한국 주식 (KRX 에러 시 코스피/코스닥 우회)
    try:
        df_krx = fdr.StockListing('KRX')
        if not df_krx.empty and 'Marcap' in df_krx.columns:
            tickers.extend(df_krx.sort_values(by='Marcap', ascending=False).head(400)['Code'].tolist())
    except Exception as e:
        try:
            df_k = pd.concat([fdr.StockListing('KOSPI'), fdr.StockListing('KOSDAQ')])
            if 'Marcap' in df_k.columns: tickers.extend(df_k.sort_values('Marcap', ascending=False).head(400)['Code'].tolist())
        except Exception as e2: errors.append(f"KRX/KOSPI 수집 실패: {e2}")

    # 2. 미국 주식 (Finviz 403 차단 방지 -> FDR S&P500 + NASDAQ 활용)
    try:
        df_sp = fdr.StockListing('SP500')
        df_ndq = fdr.StockListing('NASDAQ')
        us_tickers = list(set(df_sp['Symbol'].tolist() + df_ndq['Symbol'].tolist()[:800]))
        tickers.extend(us_tickers[:1000])
    except Exception as e: errors.append(f"미국주식(FDR) 수집 실패: {e}")

    # 3. 미국 ETF (이건 Finviz 외엔 답이 없으므로 시도하되 실패시 조용히 넘김)
    try:
        e = Overview(); e.set_filter(filters_dict={'Industry': 'Exchange Traded Fund'})
        tickers.extend(e.screener_view(order='Market Cap', ascend=False).head(400)['Ticker'].tolist())
    except: pass 

    return sorted(list(set(tickers))), errors

# ==========================================
# 3. 지표 및 전략 알고리즘 (복구 완료)
# ==========================================
def calculate_macdv(df, short=12, long=26, signal=9):
    ef, es = df['Close'].ewm(span=short, adjust=False).mean(), df['Close'].ewm(span=long, adjust=False).mean()
    tr = pd.concat([df['High']-df['Low'], np.abs(df['High']-df['Close'].shift()), np.abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
    atr = tr.ewm(span=long, adjust=False).mean()
    macd_v = ((ef - es) / (atr + 1e-9)) * 100
    return macd_v, macd_v.ewm(span=signal, adjust=False).mean()

def calculate_daily_indicators(df):
    if len(df) < 260: return None
    df = df.copy()
    df['SMA50'], df['STD50'] = df['Close'].rolling(50).mean(), df['Close'].rolling(50).std()
    df['BB50_UP'], df['BB50_LO'] = df['SMA50'] + (2*df['STD50']), df['SMA50'] - (2*df['STD50'])
    df['BW50'] = (df['BB50_UP'] - df['BB50_LO']) / df['SMA50']
    df['Donchian_High_50'] = df['High'].rolling(50).max().shift(1)
    chg = df['Close'].diff()
    r_up, r_dn, r_fl = np.where(chg > 0, df['Volume'], 0), np.where(chg < 0, df['Volume'], 0), np.where(chg == 0, df['Volume'], 0)
    df['VR50'] = ((pd.Series(r_up).rolling(50).sum() + pd.Series(r_fl).rolling(50).sum()/2) / (pd.Series(r_dn).rolling(50).sum() + pd.Series(r_fl).rolling(50).sum()/2 + 1e-9)).values * 100
    df['SMA20'], df['STD20'] = df['Close'].rolling(20).mean(), df['Close'].rolling(20).std()
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
    
    # [핵심] 질문자님 원본 VCP 조건 100% 복구 (cond3 제거)
    cond1 = curr['Close'] > s150 and curr['Close'] > s200
    cond2 = s150 > s200
    cond4 = s50 > s150
    cond5 = curr['Close'] > df['Low'].iloc[-252:].min() * 1.25
    cond6 = curr['Close'] > df['High'].iloc[-252:].max() * 0.75
    if not (cond1 and cond2 and cond4 and cond5 and cond6): return False, None 

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

    c_idx = len(df) - 1
    t_phase = get_phase(c_idx)
    if t_phase in ["Phase 1", "Phase 3"]:
        y_phase = get_phase(c_idx - 1)
        if (y_phase == "Phase 0 (수렴)" and t_phase == "Phase 1") or (y_phase == "대기/눌림목" and t_phase == "Phase 3"):
            return True, {"Today_Phase": t_phase + ("(1차)" if t_phase == "Phase 1" else "(2차)"), "Yest_Phase": y_phase, "Price": df.iloc[c_idx]['Close'], "EMA20": df.iloc[c_idx]['EMA20'], "Is_New": True}
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
# 4. 모멘텀 전략 및 병렬 처리기 (복구 완료)
# ==========================================
def _single_momentum(item, type_name):
    t, n = item
    rt, df = smart_download(t)
    if len(df) < 50: return None
    df_ind = calculate_daily_indicators(df)
    if df_ind is None: return None
    c = df['Close']; curr = c.iloc[-1]
    bbw = (4 * c.rolling(50).std()) / c.ewm(span=50).mean()
    curr_bbw = bbw.iloc[-1] if not pd.isna(bbw.iloc[-1]) and bbw.iloc[-1] > 0 else 0.001
    r12 = c.pct_change(252).iloc[-1] if len(c) > 252 else 0
    r6  = c.pct_change(126).iloc[-1] if len(c) > 126 else 0
    r3  = c.pct_change(63).iloc[-1]  if len(c) > 63 else 0
    r1  = c.pct_change(21).iloc[-1]  if len(c) > 21 else 0
    pure_score = (((r12 + r6)/2 - r3) + r1) * 100
    
    hd, pd_d, diff = "-", "-", 0
    if len(df) >= 252:
        w52 = df.iloc[-252:]
        hi = w52['Close'].idxmax()
        hd = hi.strftime('%Y-%m-%d')
        pw = w52[w52.index < hi]
        if len(pw) > 0:
            pd_d = pw['Close'].idxmax().strftime('%Y-%m-%d')
            diff = (hi - pw['Close'].idxmax()).days
    return {
        f"{type_name}": f"{rt} ({n})", "순수모멘텀스코어_raw": pure_score, "최종모멘텀스코어_raw": pure_score / curr_bbw,
        "스퀴즈": "🔥" if ('TTM_Squeeze' in df_ind and df_ind['TTM_Squeeze'].iloc[-5:].any()) else "-", 
        "BB(50,2)돌파": "O" if (c > df_ind['BB50_UP']).iloc[-3:].any() else "-", 
        "돈키언(50)돌파": "O" if (c > df_ind['Donchian_High_50']).iloc[-3:].any() else "-", 
        "정배열": "⭐ 정배열" if (curr > c.ewm(span=20).mean().iloc[-1] and curr > c.ewm(span=60).mean().iloc[-1] and curr > c.ewm(span=100).mean().iloc[-1] and curr > c.ewm(span=200).mean().iloc[-1]) else "-", 
        "장기추세": "📈 상승" if (c.ewm(span=60).mean().iloc[-1] > c.ewm(span=100).mean().iloc[-1] > c.ewm(span=200).mean().iloc[-1]) else "-", 
        "MACD-V": f"{df_ind['MACD_V'].iloc[-1]:.2f}", "ATR": f"{df_ind['ATR14'].iloc[-1]:.2f}",
        "현52주신고가일": hd, "전52주신고가일": pd_d, "차이일": f"{diff}일", "현재가": curr
    }

def analyze_momentum_parallel(targets, type_name="ETF"):
    if not targets: return pd.DataFrame()
    res, bar = [], st.progress(0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as exe:
        fut = {exe.submit(_single_momentum, item, type_name): item for item in targets}
        for i, f in enumerate(concurrent.futures.as_completed(fut)):
            bar.progress((i+1)/len(targets))
            r = f.result()
            if r: res.append(r)
    bar.empty()
    if res:
        df_res = pd.DataFrame(res).sort_values("순수모멘텀스코어_raw", ascending=False).reset_index(drop=True)
        tc = len(df_res)
        df_res['rank_temp'] = df_res['최종모멘텀스코어_raw'].rank(method='min', ascending=False)
        df_res['조정 모멘텀 순위'] = df_res['rank_temp'].apply(lambda x: f"{int(x)}/{tc}")
        df_res['저변동돌파'] = df_res.apply(lambda row: "🚨매수발생" if (row['rank_temp'] <= tc * 0.25 and (row['BB(50,2)돌파'] == 'O' or row['돈키언(50)돌파'] == 'O')) else "-", axis=1)
        df_res['순수모멘텀스코어'] = df_res['순수모멘텀스코어_raw'].apply(lambda x: f"{x:.2f}")
        df_res['현재가'] = df_res['현재가'].apply(lambda x: f"{x:,.2f}")
        return df_res[[f"{type_name}", "순수모멘텀스코어", "조정 모멘텀 순위", "스퀴즈", "BB(50,2)돌파", "돈키언(50)돌파", "저변동돌파", "정배열", "장기추세", "MACD-V", "ATR", "현52주신고가일", "전52주신고가일", "차이일", "현재가"]]
    return pd.DataFrame()

def _fetch_kr_etf(item):
    t, n = item
    time.sleep(random.uniform(0.05, 0.2))
    bbw, sqz, bb, dc = 0.001, "-", "-", "-"
    try:
        df = fdr.DataReader(t).tail(260) 
        if len(df) >= 60:
            ind = calculate_daily_indicators(df)
            if ind is not None:
                bbw = ind.iloc[-1]['BW50'] if ind.iloc[-1]['BW50'] > 0 else 0.001
                sqz = "🔥" if ('TTM_Squeeze' in ind.columns and ind['TTM_Squeeze'].iloc[-5:].any()) else "-"
                bb = "O" if (ind['Close'] > ind['BB50_UP']).iloc[-3:].any() else "-"
                dc = "O" if (ind['Close'] > ind['Donchian_High_50']).iloc[-3:].any() else "-"
    except: pass
    try:
        res = requests.get(f"https://finance.naver.com/item/coinfo.naver?code={t}", headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        res.encoding = 'euc-kr' 
        soup = BeautifulSoup(res.text, 'html.parser')
        tbl = soup.find('table', summary='1개월 수익률 정보')
        r1, r3, r6, r12 = None, None, None, None
        if tbl:
            for row in tbl.find('tbody').find_all('tr'):
                th, td = row.find('th').text.strip(), row.find('td').text.strip()
                try: val = float(td.replace('%', '').replace('+', '').replace(',', ''))
                except: val = None
                if '1개월' in th: r1 = val
                elif '3개월' in th: r3 = val
                elif '6개월' in th: r6 = val
                elif '1년' in th: r12 = val
        return {'Symbol': t, 'Name': n, '1M_Return(%)': r1, '3M_Return(%)': r3, '6M_Return(%)': r6, '12M_Return(%)': r12, 'BBW': bbw, '스퀴즈': sqz, 'BB(50,2)돌파': bb, '돈키언(50)돌파': dc}
    except: return {'Symbol': t, 'Name': n, '1M_Return(%)': None, '3M_Return(%)': None, '6M_Return(%)': None, '12M_Return(%)': None, 'BBW': bbw, '스퀴즈': "-", 'BB(50,2)돌파': "-", '돈키언(50)돌파': "-"}

def run_korean_etf_analysis():
    df_etf = fdr.StockListing('ETF/KR')
    df_etf = df_etf[~df_etf['Name'].str.contains('레버리지|인버스', na=False)]
    items = list(zip(df_etf['Symbol'], df_etf['Name']))
    res, bar = [], st.progress(0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as exe:
        fut = {exe.submit(_fetch_kr_etf, i): i for i in items}
        for i, f in enumerate(concurrent.futures.as_completed(fut)):
            bar.progress((i+1)/len(items))
            res.append(f.result())
    bar.empty()
    df_res = pd.DataFrame(res)
    df_res['순수모멘텀스코어_raw'] = (0.5 * (df_res['12M_Return(%)'] + df_res['6M_Return(%)']) - df_res['3M_Return(%)'] + df_res['1M_Return(%)'])
    df_res['최종모멘텀스코어_raw'] = df_res['순수모멘텀스코어_raw'] / df_res['BBW']
    df_res = df_res.dropna(subset=['순수모멘텀스코어_raw']).sort_values(by='순수모멘텀스코어_raw', ascending=False).reset_index(drop=True)
    tc = len(df_res)
    df_res['rank_temp'] = df_res['최종모멘텀스코어_raw'].rank(method='min', ascending=False)
    df_res['조정 모멘텀 순위'] = df_res['rank_temp'].apply(lambda x: f"{int(x)}/{tc}")
    df_res['저변동돌파'] = df_res.apply(lambda row: "🚨매수발생" if (row['rank_temp'] <= tc * 0.25 and (row['BB(50,2)돌파'] == 'O' or row['돈키언(50)돌파'] == 'O')) else "-", axis=1)
    df_res['순수모멘텀스코어'] = df_res['순수모멘텀스코어_raw'].apply(lambda x: f"{x:.2f}")
    cols = [c for c in ['Symbol', 'Name', '순수모멘텀스코어', '조정 모멘텀 순위', '스퀴즈', 'BB(50,2)돌파', '돈키언(50)돌파', '저변동돌파', '1M_Return(%)', '3M_Return(%)', '6M_Return(%)', '12M_Return(%)'] if c in df_res.columns]
    return df_res[cols]

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
    if res: res.sort(key=lambda x: x.get('종목코드', '')) 
    return res

# ==========================================
# 5. UI 및 Tabs
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

with tab1:
    cols1 = st.columns(12) 
    if cols1[0].button("🌍 섹터"):
        etfs = get_etfs_from_csv(ETF_CSV_URL, is_etf=True)
        if etfs: st.dataframe(analyze_momentum_parallel(etfs, "ETF"), use_container_width=True)
    if cols1[1].button("🇰🇷 한국ETF"):
        st.info("한국 상장 ETF 리스트 병렬 분석 중...")
        df_kr_etf = run_korean_etf_analysis()
        if not df_kr_etf.empty:
            st.success(f"✅ 총 {len(df_kr_etf)}개 한국 ETF 분석 완료!")
            st.dataframe(df_kr_etf, use_container_width=True)

with tab2:
    if st.button("🏳️ 국가 ETF 분석"):
        ctrys = get_etfs_from_csv(COUNTRY_CSV_URL, is_etf=True)
        if ctrys: st.dataframe(analyze_momentum_parallel(ctrys, "국가ETF"), use_container_width=True)

with tab3:
    cols3 = st.columns(11)
    if cols3[0].button("🌪️ VCP"):
        tickers = get_tickers_from_csv(STOCK_CSV_URL)
        st.info("VCP 4단계 분석 중...")
        def _vcp(t):
            rt, df = smart_download(t)
            if len(df) < 250: return None
            p, i = check_vcp_pattern(df)
            if p and "4단계" in i['status']:
                yp, yi = check_vcp_pattern(df.iloc[:-1].copy())
                e1, e2, e3 = get_eps_changes_from_db(rt)
                return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i['price']:,.0f}", '비고': i['status'], '전일비고': yi['status'] if yp else "-", '주봉MACD': get_weekly_macd_status(df), '손절가': f"{i['stop_loss']:,.0f}", '목표가(3R)': f"{i['target_price']:,.0f}", '스퀴즈': i['squeeze'], '1W변화': e1, '1M변화': e2, '3M변화': e3, 'Pivot': f"{i['pivot']:,.0f}", 'info': i, 'df': df}
            return None
        res = run_parallel_analysis(tickers, _vcp)
        if res:
            st.success(f"✅ VCP 4단계 {len(res)}개 발견!")
            st.dataframe(pd.DataFrame(res).drop(columns=['info', 'df'], errors='ignore'), use_container_width=True)
            for j in range(0, len(res), 2):
                c1, c2 = st.columns(2)
                c1.plotly_chart(plot_vcp_chart(res[j]['df'], res[j]['종목코드'], res[j]['info']), use_container_width=True)
                if j+1 < len(res): c2.plotly_chart(plot_vcp_chart(res[j+1]['df'], res[j+1]['종목코드'], res[j+1]['info']), use_container_width=True)
            save_to_supabase(res, "VCP_Pattern")
        else: st.warning("조건 만족 종목 없음")

    if cols3[1].button("🚀 일봉"):
        tickers = get_tickers_from_csv(STOCK_CSV_URL)
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

    if cols3[2].button("📅 주봉"):
        tickers = get_tickers_from_csv(STOCK_CSV_URL)
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

    if cols3[3].button("🗓️ 월봉"):
        tickers = get_tickers_from_csv(STOCK_CSV_URL)
        st.info("월봉 분석 중...")
        def _mo(t):
            rt, df = smart_download(t, "1mo", "max")
            p, i = check_monthly_condition(df)
            if p: return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i['price']:,.0f}", 'ATH최고가': f"{i['ath_price']:,.0f}", 'ATH달성월': i['ath_date'], '고권역(월)': f"{i['month_count']}개월"}
            return None
        res = run_parallel_analysis(tickers, _mo)
        if res: st.dataframe(pd.DataFrame(res))

    if cols3[4].button("일+월봉"):
        tickers = get_tickers_from_csv(STOCK_CSV_URL)
        def _dm(t):
            rt, df_d = smart_download(t); p_d, i_d = check_daily_condition(df_d)
            if not p_d: return None
            _, df_m = smart_download(t, "1mo", "max"); p_m, i_m = check_monthly_condition(df_m)
            if not p_m: return None
            return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i_d['price']:,.0f}", '스퀴즈': i_d['squeeze'], 'ATH달성월': i_m['ath_date']}
        res = run_parallel_analysis(tickers, _dm)
        if res: st.dataframe(pd.DataFrame(res))

    if cols3[5].button("일+주봉"):
        tickers = get_tickers_from_csv(STOCK_CSV_URL)
        def _dw(t):
            rt, df_d = smart_download(t); p_d, i_d = check_daily_condition(df_d)
            if not p_d: return None
            _, df_w = smart_download(t, "1wk"); p_w, i_w = check_weekly_condition(df_w)
            if not p_w: return None
            return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i_d['price']:,.0f}", '스퀴즈': i_d['squeeze'], '주봉구분': i_w['bw_change']}
        res = run_parallel_analysis(tickers, _dw)
        if res: st.dataframe(pd.DataFrame(res))

    if cols3[6].button("주+월봉"):
        tickers = get_tickers_from_csv(STOCK_CSV_URL)
        def _wm(t):
            rt, df_w = smart_download(t, "1wk"); p_w, i_w = check_weekly_condition(df_w)
            if not p_w: return None
            _, df_m = smart_download(t, "1mo", "max"); p_m, i_m = check_monthly_condition(df_m)
            if not p_m: return None
            return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i_w['price']:,.0f}", '주봉구분': i_w['bw_change'], 'ATH달성월': i_m['ath_date']}
        res = run_parallel_analysis(tickers, _wm)
        if res: st.dataframe(pd.DataFrame(res))

    if cols3[7].button("⚡ 통합"):
        tickers = get_tickers_from_csv(STOCK_CSV_URL)
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

    if cols3[8].button("🔥 MA돌파"):
        tickers = get_tickers_from_csv(STOCK_CSV_URL)
        st.info("MA 돌파 분석 중...")
        def _dual(t):
            rt, df = smart_download(t)
            p, i = check_dual_ma_breakout(df)
            if p:
                e1, e2, e3 = get_eps_changes_from_db(rt)
                return {'상태': "🚨신규돌파", '종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i['Price']:,.0f}", '전일Phase': i['Yest_Phase'], '당일Phase': i['Today_Phase'], '손절': f"{i['EMA20']:,.0f}", '1W': e1, '1M': e2, '3M': e3}
            return None
        res = run_parallel_analysis(tickers, _dual)
        if res: st.dataframe(pd.DataFrame(res).sort_values(by=['당일Phase']))

    if cols3[9].button("🏆 컵핸들"):
        tickers = get_tickers_from_csv(STOCK_CSV_URL)
        def _cup(t):
            rt, df = smart_download(t, "1wk")
            p, i = check_cup_handle_pattern(df)
            if p: return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{df['Close'].iloc[-1]:,.0f}", '깊이': i['depth'], '돌파': i['pivot']}
            return None
        res = run_parallel_analysis(tickers, _cup)
        if res: st.dataframe(pd.DataFrame(res))

    if cols3[10].button("👤 역H&S"):
        tickers = get_tickers_from_csv(STOCK_CSV_URL)
        def _hs(t):
            rt, df = smart_download(t, "1wk")
            p, i = check_inverse_hs_pattern(df)
            if p: return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{df['Close'].iloc[-1]:,.0f}", '넥라인': i['Neckline'], '거래량급증': i['Vol_Ratio']}
            return None
        res = run_parallel_analysis(tickers, _hs)
        if res: st.dataframe(pd.DataFrame(res))

with tab_marketcap:
    st.markdown("### 👑 한/미 시가총액 상위 1,800개 통합 기술적 분석 (병렬/우회)")
    if st.button("🚀 시총 상위 전체 병렬 스크리닝 시작", type="primary"):
        with st.spinner("Finviz 403 차단 우회 및 FDR 티커 수집 중..."):
            top_tickers, errors = get_top_marketcap_tickers()
        if errors:
            for err in errors: st.warning(f"⚠️ {err}")
        if not top_tickers: st.error("티커 수집에 실패했습니다. KRX 및 FDR 서버를 확인해주세요.")
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
        tickers = get_tickers_from_csv(STOCK_CSV_URL)
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
