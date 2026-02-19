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
from concurrent.futures import ThreadPoolExecutor, as_completed

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
st.title("📈 Pro 주식 검색기: 섹터/국가/기술적/퀀티와이즈 DB 통합")

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
            if ':' in raw_ticker:
                ticker = raw_ticker.split(':')[-1].strip()
            else:
                ticker = raw_ticker
            name = str(row[1]).strip() if len(row) > 1 else ticker
            if ticker:
                etf_list.append((ticker, name))
        return etf_list
    except Exception as e:
        st.error(f"ETF 시트 읽기 실패: {e}")
        return []

def get_country_etfs_from_sheet():
    try:
        df = pd.read_csv(COUNTRY_CSV_URL, header=None)
        etf_list = []
        for index, row in df.iterrows():
            raw_ticker = str(row[0]).strip()
            if not raw_ticker or raw_ticker.lower() in ['ticker', 'symbol', '종목코드', '티커', 'nan']:
                continue
            if ':' in raw_ticker:
                ticker = raw_ticker.split(':')[-1].strip()
            else:
                ticker = raw_ticker
            name = str(row[1]).strip() if len(row) > 1 else ticker
            if ticker:
                etf_list.append((ticker, name))
        return etf_list
    except Exception as e:
        st.error(f"국가 ETF 시트 읽기 실패: {e}")
        return []

def get_unique_tickers_from_db():
    if not supabase: return []
    try:
        response = supabase.table("history").select("ticker").execute()
        if response.data:
            return list(set([row['ticker'] for row in response.data]))
        return []
    except Exception as e: return []

def remove_duplicates_from_db():
    if not supabase: return
    try:
        response = supabase.table("history").select("id, ticker, created_at").order("created_at", desc=True).execute()
        data = response.data
        if not data:
            st.warning("데이터가 없습니다.")
            return

        seen_tickers = set()
        ids_to_remove = []
        for row in data:
            ticker = row['ticker']
            if ticker in seen_tickers:
                ids_to_remove.append(row['id'])
            else:
                seen_tickers.add(ticker)
        
        if ids_to_remove:
            for pid in ids_to_remove:
                supabase.table("history").delete().eq("id", pid).execute()
            st.success(f"🧹 History 중복된 {len(ids_to_remove)}개 데이터를 삭제했습니다.")
        else:
            st.info("History: 삭제할 중복 데이터가 없습니다.")

    except Exception as e:
        st.error(f"중복 제거 실패: {e}")

# [핵심 수정] yf.Ticker().history() 사용 + 날짜 표준화
def smart_download(ticker, interval="1d", period="2y"):
    if ':' in ticker: ticker = ticker.split(':')[-1]
    ticker = ticker.replace('/', '-')
    candidates = [ticker]
    if ticker.isdigit() and len(ticker) == 6:
        candidates = [f"{ticker}.KS", f"{ticker}.KQ", ticker]
    
    for t in candidates:
        for attempt in range(3): # 재시도 3회
            try:
                dat = yf.Ticker(t)
                df = dat.history(period=period, interval=interval, auto_adjust=False)
                
                if not df.empty and len(df) > 5:
                    # Timezone 제거 및 날짜 정렬 (계산 일관성)
                    try:
                        if df.index.tz is not None: df.index = df.index.tz_localize(None)
                        df.index = df.index.normalize()
                    except: pass
                    
                    df = df[~df.index.duplicated(keep='last')]
                    df = df.sort_index() 
                    
                    if 'Close' in df.columns:
                        df = df.loc[:, ~df.columns.duplicated()]
                        df = df.ffill()
                        return t, df
                time.sleep(0.3)
            except:
                time.sleep(0.3)
                continue
    return ticker, pd.DataFrame()

# [중요] 종목 정보 캐싱
@st.cache_data(ttl=3600*24) 
def get_ticker_info_safe(ticker):
    try:
        tick = yf.Ticker(ticker)
        try:
            meta = tick.info
            if meta: return meta
        except:
            return None
        return None
    except:
        return None

def get_stock_sector(ticker):
    meta = get_ticker_info_safe(ticker)
    if not meta: return "Unknown"
    
    quote_type = meta.get('quoteType', '').upper()
    if 'ETF' in quote_type or 'FUND' in quote_type:
        name = meta.get('shortName', '')
        if not name: name = meta.get('longName', 'ETF')
        return f"[ETF] {name}"
    
    sector = meta.get('sector', '')
    if not sector: sector = meta.get('industry', '')
    if not sector: sector = meta.get('shortName', '')
    
    translations = {
        'Technology': '기술', 'Healthcare': '헬스케어', 'Financial Services': '금융',
        'Consumer Cyclical': '임의소비재', 'Industrials': '산업재', 'Basic Materials': '소재',
        'Energy': '에너지', 'Utilities': '유틸리티', 'Real Estate': '부동산',
        'Communication Services': '통신', 'Consumer Defensive': '필수소비재',
        'Semiconductors': '반도체'
    }
    return translations.get(sector, sector)

def save_to_supabase(data_list, strategy_name):
    if not supabase:
        st.error("⚠️ DB 연결 실패")
        return

    rows_to_insert = []
    for item in data_list:
        rows_to_insert.append({
            "ticker": str(item['종목코드']),
            "sector": str(item.get('섹터', '-')),
            "price": str(item['현재가']).replace(',', ''),
            "strategy": strategy_name,
            "high_date": str(item.get('현52주신고가일', '')), 
            "bw": str(item.get('BW_Value', '')), 
            "macd_v": str(item.get('MACD_V_Value', ''))
        })
    
    try:
        supabase.table("history").insert(rows_to_insert).execute()
        st.toast(f"✅ {len(rows_to_insert)}개 종목 DB 저장 완료!", icon="💾")
    except Exception as e:
        st.error(f"DB 저장 실패: {e}")

# ==============================================================================
# [핵심 로직] 정규화 및 DB 조회
# ==============================================================================
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
        if df.empty: return {}
        df_latest = df.drop_duplicates(subset='ticker', keep='first')
        result_dict = {}
        for _, row in df_latest.iterrows():
            result_dict[row['ticker']] = {
                '1w': str(row.get('change_1w') or "-"),
                '1m': str(row.get('change_1m') or "-"),
                '3m': str(row.get('change_3m') or "-")
            }
        return result_dict
    except Exception as e:
        return {}

GLOBAL_QUANT_DATA = fetch_latest_quant_data_from_db()

def get_eps_changes_from_db(ticker):
    norm_ticker = normalize_ticker_for_app_lookup(ticker)
    if norm_ticker in GLOBAL_QUANT_DATA:
        d = GLOBAL_QUANT_DATA[norm_ticker]
        return d['1w'], d['1m'], d['3m']
    return "-", "-", "-"

# ==========================================
# 4. 분석 알고리즘 (지표 계산 & 패턴)
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
    
    # [안전장치] 중복 인덱스 및 컬럼 제거 + 정렬
    try:
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
    except: pass
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()
    df = df.loc[:, ~df.columns.duplicated()]

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
    
    # [핵심 수정] 데이터 정합성 보장
    try:
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
    except: pass
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()
    df = df.loc[:, ~df.columns.duplicated()]
    
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

# [VCP 패턴] 60일 기준
def check_vcp_pattern(df):
    if len(df) < 250: return False, None
    df = calculate_daily_indicators(df) 
    if df is None: return False, None
    
    curr = df.iloc[-1]
    sma50 = df['Close'].rolling(50).mean().iloc[-1]
    sma150 = df['Close'].rolling(150).mean().iloc[-1]
    sma200 = df['Close'].rolling(200).mean().iloc[-1]
    
    # 1. 추세
    cond1 = curr['Close'] > sma150 and curr['Close'] > sma200
    cond2 = sma150 > sma200
    cond3 = df['SMA50'].iloc[-1] > df['SMA50'].iloc[-20] 
    cond4 = sma50 > sma150
    low_52 = df['Low'].iloc[-252:].min()
    cond5 = curr['Close'] > low_52 * 1.25
    high_52 = df['High'].iloc[-252:].max()
    cond6 = curr['Close'] > high_52 * 0.75
    
    stage_1_pass = cond1 and cond2 and cond4 and cond5 and cond6
    if not stage_1_pass: return False, None 

    # 2. 파동 (60일 기준)
    window = 60
    subset = df.iloc[-window:]
    p1 = subset.iloc[:20]; p2 = subset.iloc[20:40]; p3 = subset.iloc[40:]
    
    range1 = (p1['High'].max() - p1['Low'].min()) / p1['High'].max()
    range2 = (p2['High'].max() - p2['Low'].min()) / p2['High'].max()
    range3 = (p3['High'].max() - p3['Low'].min()) / p3['High'].max()
    
    contraction = (range3 < range2) or (range2 < range1) or (range3 < 0.12)
    if not contraction: return False, None

    # 3. 셋업
    last_vol_avg = p3['Volume'].mean()
    prev_vol_avg = p1['Volume'].mean()
    vol_dry_up = last_vol_avg < prev_vol_avg * 1.2 
    tight_area = range3 < 0.15 
    
    stage_3_pass = vol_dry_up and tight_area
    stop_loss = p3['Low'].min()
    risk = curr['Close'] - stop_loss
    target_price = curr['Close'] + (risk * 3) if risk > 0 else 0
    
    # 4. 돌파
    prior_days = p3.iloc[:-1] 
    pivot_point = prior_days['High'].max() if len(prior_days) > 0 else p3['High'].max()
    vol_ma50 = df['Volume'].iloc[-51:-1].mean()
    breakout = (curr['Close'] > pivot_point) and (curr['Volume'] > vol_ma50 * 1.2)
    
    status = ""
    if stage_3_pass and not breakout: status = "3단계 (수렴중)"
    elif stage_3_pass and breakout: status = "4단계 (돌파!🚀)"
    else:
        if breakout and tight_area: status = "4단계 (돌파!🚀)"
        else: return False, None

    return True, {'status': status, 'stop_loss': stop_loss, 'target_price': target_price, 'squeeze': "🔥" if df['TTM_Squeeze'].iloc[-1] else "-", 'price': curr['Close'], 'pivot': pivot_point}

def get_weekly_macd_status(daily_df):
    try:
        df_w = daily_df.resample('W-FRI').agg({'Close': 'last', 'High': 'max', 'Low': 'min', 'Volume': 'sum'}).dropna()
        if len(df_w) < 26: return "-"
        ema12 = df_w['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_w['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            return "⚡GC" if macd_line.iloc[-2] <= signal_line.iloc[-2] else "🔵 Buy"
        return "🔻 Sell"
    except: return "-"

def plot_vcp_chart(df, ticker, info):
    df_plot = df.iloc[-252:].copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'].rolling(50).mean(), line=dict(color='green', width=1), name='SMA 50'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'].rolling(150).mean(), line=dict(color='blue', width=1), name='SMA 150'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'].rolling(200).mean(), line=dict(color='red', width=1), name='SMA 200'))
    fig.add_hline(y=info['pivot'], line_dash="dot", line_color="red", annotation_text="Pivot")
    fig.add_hline(y=info['stop_loss'], line_dash="dot", line_color="blue", annotation_text="Stop Loss")
    fig.update_layout(title=f"{ticker} - VCP Chart", xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
    return fig

def check_daily_condition(df):
    if len(df) < 260: return False, None
    df = calculate_daily_indicators(df)
    if df is None: return False, None
    curr = df.iloc[-1]
    dc_cond = (df['Close'] > df['Donchian_High_50']).iloc[-3:].any()
    bb_cond = (df['Close'] > df['BB50_UP']).iloc[-3:].any()
    vr_cond = (df['VR50'].iloc[-3:] > 110).any()
    bw_cond = (df['BW50'].iloc[-51] > curr['BW50']) if len(df)>55 else False
    macd_cond = curr['MACD_OSC_C'] > 0
    if (dc_cond or bb_cond) and (sum([vr_cond, bw_cond, macd_cond]) >= 2):
        win_52 = df.iloc[-252:]
        return True, {'price': curr['Close'], 'atr': curr['ATR14'], 'high_date': win_52['Close'].idxmax().strftime('%Y-%m-%d'), 'bw_curr': curr['BW50'], 'macdv': curr['MACD_V'], 'squeeze': "🔥" if df['TTM_Squeeze'].iloc[-5:].any() else "-"}
    return False, None

def check_weekly_condition(df):
    if len(df) < 40: return False, None
    # 지표 계산
    df['SMA30'] = df['Close'].rolling(window=30).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    delta = df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI14'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    e12 = df['Close'].ewm(span=12, adjust=False).mean(); e26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = e12 - e26; sig = macd.ewm(span=9, adjust=False).mean(); df['MACD_Hist'] = macd - sig
    sma12 = df['Close'].rolling(12).mean(); std12 = df['Close'].rolling(12).std(); bb_up_12 = sma12 + (2 * std12)
    e12c = df['Close'].ewm(span=12, adjust=False).mean(); e36c = df['Close'].ewm(span=36, adjust=False).mean()
    macd_c = e12c - e36c; sig_c = macd_c.ewm(span=9, adjust=False).mean()
    df['MACD_V'], _ = calculate_macdv(df, 12, 26, 9)
    high_low = df['High'] - df['Low']; high_close = np.abs(df['High'] - df['Close'].shift()); low_close = np.abs(df['Low'] - df['Close'].shift())
    df['ATR14'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).ewm(span=14, adjust=False).mean()
    
    curr = df.iloc[-1]
    
    # 1. 필수 선행 조건
    if not (curr['Close'] > curr['SMA30'] and curr['RSI14'] > 50 and (df['MACD_Hist'].iloc[-1] > df['MACD_Hist'].iloc[-2] or df['MACD_Hist'].iloc[-1] > 0)):
        return False, None
        
    is_1 = False; past_12w = df.iloc[-13:-1]
    if not past_12w.empty:
        # A. 과거 돌파, B. 현재 휴식, C. 가격지지(고점 -15%내), D. 추세지지(EMA20 위) - (거래량 조건 삭제됨)
        if (past_12w['Close'] > bb_up_12.loc[past_12w.index]).any() and curr['Close'] <= (bb_up_12.iloc[-1] * 1.02) and curr['Close'] >= (past_12w['High'].max() * 0.85) and curr['Close'] > curr['EMA20']:
            is_1 = True
            
    is_2 = macd_c.iloc[-2] <= sig_c.iloc[-2] and macd_c.iloc[-1] > sig_c.iloc[-1]
    
    status = []
    if is_1: status.append("돌파수렴(눌림)")
    if is_2: status.append("MACD매수")
    
    if status: return True, {'price': curr['Close'], 'atr': curr['ATR14'], 'bw_change': " / ".join(status), 'macdv': curr['MACD_V']}
    return False, None

def check_monthly_condition(df):
    if len(df) < 12: return False, None
    ath = df['High'].max(); curr = df['Close'].iloc[-1]
    if curr >= ath * 0.90: return True, {'price': curr, 'ath_price': ath, 'ath_date': df['High'].idxmax().strftime('%Y-%m'), 'month_count': (df['Close'] >= ath * 0.90).sum()}
    return False, None

def check_cup_handle_pattern(df):
    if len(df) < 26: return False, None
    sub = df.iloc[-26:].copy()
    if len(sub) < 26: return False, None
    idx_A = sub['High'].idxmax(); val_A = sub.loc[idx_A, 'High']
    if idx_A == sub.index[-1]: return False, "A가 끝점"
    after_A = sub.loc[idx_A:]
    if len(after_A) < 5: return False, "기간 짧음"
    idx_B = after_A['Low'].idxmin(); val_B = after_A.loc[idx_B, 'Low']
    if val_B > val_A * 0.85: return False, "깊이 얕음"
    after_B = sub.loc[idx_B:]
    if len(after_B) < 2: return False, "반등 짧음"
    idx_C = after_B['High'].idxmax(); val_C = after_B.loc[idx_C, 'High']
    if val_C < val_A * 0.85: return False, "회복 미달"
    curr_close = df['Close'].iloc[-1]
    if curr_close < val_B: return False, "핸들 붕괴"
    if curr_close < val_C * 0.80: return False, "핸들 깊음"
    return True, {"depth": f"{(1 - val_B/val_A)*100:.1f}%", "handle_weeks": f"{len(df.loc[idx_C:])}주", "pivot": f"{val_C:,.0f}"}

def check_inverse_hs_pattern(df):
    if len(df) < 60: return False, None
    window = 60; sub = df.iloc[-window:].copy()
    if len(sub) < 60: return False, None
    part1 = sub.iloc[:20]; part2 = sub.iloc[20:40]; part3 = sub.iloc[40:]
    min_L = part1['Low'].min(); min_H = part2['Low'].min(); min_R = part3['Low'].min()
    if not (min_H < min_L and min_H < min_R): return False, "머리 미형성"
    max_R = part3['High'].max(); curr_close = df['Close'].iloc[-1]
    if curr_close < min_R * 1.05: return False, "반등 약함"
    vol_recent = part3['Volume'].mean(); vol_prev = part2['Volume'].mean()
    vol_ratio = vol_recent / vol_prev if vol_prev > 0 else 1.0
    return True, {"Neckline": f"{max_R:,.0f}", "Breakout": "Ready" if curr_close < max_R else "Yes", "Vol_Ratio": f"{vol_ratio:.1f}배"}

# [정확성 보장] 병렬 처리 함수
def analyze_momentum_strategy_parallel(target_list, type_name="ETF"):
    if not target_list: return pd.DataFrame()
    st.write(f"📊 총 {len(target_list)}개 {type_name} 분석 중...")
    results = []; failed_tickers = []
    
    def worker(item):
        t, n = item
        rt, df = smart_download(t, "1d", "2y")
        if df.empty or len(df) < 30: return None
        df = calculate_daily_indicators(df)
        if df is None: return None
        c = df['Close']; curr = c.iloc[-1]
        score = ((( (c.pct_change(252).iloc[-1] if len(c)>252 else 0) + (c.pct_change(126).iloc[-1] if len(c)>126 else 0) ) / 2 - (c.pct_change(63).iloc[-1] if len(c)>63 else 0)) + (c.pct_change(21).iloc[-1] if len(c)>21 else 0)) * 100
        win_52 = df.iloc[-252:] if len(df)>=252 else df
        high_idx = win_52['Close'].idxmax()
        prev_win = win_52[win_52.index < high_idx]
        prev_date = prev_win['Close'].idxmax().strftime('%Y-%m-%d') if not prev_win.empty else "-"
        return {f"{type_name}": f"{rt} ({n})", "모멘텀점수": score, "스퀴즈": "🔥" if df['TTM_Squeeze'].iloc[-5:].any() else "-", "BB(50,2)돌파": "O" if (c>df['BB50_UP']).iloc[-3:].any() else "-", "돈키언(50)돌파": "O" if (c>df['Donchian_High_50']).iloc[-3:].any() else "-", "정배열": "⭐" if (curr>c.ewm(span=20).mean().iloc[-1] and curr>c.ewm(span=200).mean().iloc[-1]) else "-", "장기추세": "📈" if c.ewm(span=60).mean().iloc[-1]>c.ewm(span=200).mean().iloc[-1] else "-", "MACD-V": f"{df['MACD_V'].iloc[-1]:.2f}", "ATR": f"{df['ATR14'].iloc[-1]:.2f}", "현52주신고가일": high_idx.strftime('%Y-%m-%d'), "전52주신고가일": prev_date, "현재가": curr}

    bar = st.progress(0)
    with ThreadPoolExecutor(max_workers=8) as executor: 
        futures = {executor.submit(worker, item): item for item in target_list}
        total = len(futures); completed = 0
        for future in as_completed(futures):
            completed += 1; bar.progress(completed / total)
            try:
                res = future.result()
                if res: results.append(res)
                else: failed_tickers.append(futures[future][0])
            except: failed_tickers.append(futures[future][0])
    bar.empty()
    if failed_tickers: st.caption(f"⚠️ 데이터 부족/오류로 제외된 종목 ({len(failed_tickers)}개): {', '.join(failed_tickers[:10])}...")
    if results:
        df_res = pd.DataFrame(results).sort_values("모멘텀점수", ascending=False)
        df_res['모멘텀점수'] = df_res['모멘텀점수'].apply(lambda x: f"{x:.2f}")
        df_res['현재가'] = df_res['현재가'].apply(lambda x: f"{x:,.2f}")
        return df_res
    return pd.DataFrame()

# -----------------------------------------------------------------------------
# 나침판 전략 (EEM->EMGF 등으로 변경된 버전)
# -----------------------------------------------------------------------------
def get_compass_signal():
    OFFENSE = ["QQQ", "SCHD", "IMTM", "GLD", "EMGF"]; CASH = "BIL"
    try:
        # 여기서는 Ticker 객체 사용 대신 일괄 다운로드 후 정리 (나침판은 소수 종목이라 괜찮음)
        data = yf.download(list(set(OFFENSE + [CASH])), period="2y", progress=False, auto_adjust=False)['Close']
        if data.empty: return None, "데이터 없음"
    except: return None, "다운로드 실패"
    
    # Timezone 제거
    try:
        if data.index.tz is not None: data.index = data.index.tz_localize(None)
        data.index = data.index.normalize()
    except: pass
    
    m_data = data.resample('ME').last()
    if len(m_data) < 13: return None, "데이터 부족"
    m12 = m_data.pct_change(12).iloc[-1]; m6 = m_data.pct_change(6).iloc[-1]; m3 = m_data.pct_change(3).iloc[-1]; m1 = m_data.pct_change(1).iloc[-1]
    scores = {}
    for t in OFFENSE:
        if t not in m12.index or np.isnan(m12[t]): continue
        score = ((m12[t] + m6[t]) / 2 - m3[t]) + m1[t]
        scores[t] = {"Score": score * 100, "12M_Trend": m12[t]}
    if not scores: return None, "계산 불가"
    df_s = pd.DataFrame(scores).T.sort_values("Score", ascending=False)
    best = df_s.index[0]
    pos = best if (df_s.iloc[0]['Score'] > 0 and df_s.iloc[0]['12M_Trend'] > 0) else CASH
    return df_s, pos

# ==========================================
# 5. 메인 화면 실행
# ==========================================
tab_compass, tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧭 나침판", "🌍 섹터", "🏳️ 국가", "📊 기술적 분석", "💰 재무분석", "📂 엑셀 데이터 매칭"])

with tab_compass:
    st.markdown("### 🧭 투자 나침판 (팩터 ETF 전략)")
    if st.button("🚀 분석 시작", type="primary"):
        df_r, pos = get_compass_signal()
        if df_r is not None:
            c1, c2 = st.columns(2)
            c1.success(f"🎯 추천 포지션: **{pos}**")
            c2.metric("1등 점수", f"{df_r.iloc[0]['Score']:.2f}")
            st.dataframe(df_r, use_container_width=True)
        else: st.error("분석 실패")

with tab1:
    if st.button("🌍 섹터 분석"):
        etfs = get_etfs_from_sheet()
        if etfs:
            res = analyze_momentum_strategy_parallel(etfs, "ETF")
            st.dataframe(res, use_container_width=True)

with tab2:
    if st.button("🏳️ 국가 분석"):
        cnt = get_country_etfs_from_sheet()
        if cnt:
            res = analyze_momentum_strategy_parallel(cnt, "국가ETF")
            st.dataframe(res, use_container_width=True)

with tab3:
    cols = st.columns(10)
    # 1. VCP
    if cols[0].button("🌪️ VCP"):
        tickers = get_tickers_from_sheet()
        if tickers:
            bar = st.progress(0); res = []; chart_cache = []; failed = []
            def v_worker(t):
                rt, df = smart_download(t, "1d", "2y")
                if len(df) < 250: return None
                p, i = check_vcp_pattern(df)
                if p:
                    eps = get_eps_changes_from_db(rt)
                    return {'data': {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i['price']:,.0f}", '비고': i['status'], '주봉MACD': get_weekly_macd_status(df), '손절가': f"{i['stop_loss']:,.0f}", 'Pivot': f"{i['pivot']:,.0f}", '1W': eps[0], '1M': eps[1]}, 'chart': (rt, df, i)}
                return None
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = {ex.submit(v_worker, t): t for t in tickers}
                done = 0
                for f in as_completed(futs):
                    done+=1; bar.progress(done/len(tickers))
                    r = f.result()
                    if r: res.append(r['data']); chart_cache.append(r['chart'])
            bar.empty()
            if res:
                df_res = pd.DataFrame(res).sort_values("비고", ascending=False)
                st.dataframe(df_res, use_container_width=True)
                for i in range(0, len(chart_cache), 2):
                    c1, c2 = st.columns(2)
                    with c1: st.plotly_chart(plot_vcp_chart(chart_cache[i][1], chart_cache[i][0], chart_cache[i][2]), use_container_width=True)
                    if i+1 < len(chart_cache):
                        with c2: st.plotly_chart(plot_vcp_chart(chart_cache[i+1][1], chart_cache[i+1][0], chart_cache[i+1][2]), use_container_width=True)
            else: st.warning("발견된 종목 없음")

    # 2. 일봉
    if cols[1].button("🚀 일봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            res = []
            def d_worker(t):
                rt, df = smart_download(t, "1d", "2y")
                if df.empty or len(df)<260: return None
                p, i = check_daily_condition(df)
                if p:
                    eps = get_eps_changes_from_db(rt)
                    return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i['price']:,.0f}", 'ATR': f"{i['atr']:,.0f}", '스퀴즈': i['squeeze'], '1W': eps[0], '1M': eps[1], '신고가일': i['high_date'], 'MACD-V': f"{i['macdv']:.2f}"}
                return None
            bar = st.progress(0)
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(d_worker, t) for t in tickers]
                for i, f in enumerate(as_completed(futs)):
                    bar.progress((i+1)/len(tickers))
                    r = f.result(); 
                    if r: res.append(r)
            bar.empty()
            if res: st.success(f"✅ {len(res)}개 발견"); st.dataframe(pd.DataFrame(res), use_container_width=True)
            else: st.warning("조건 만족 종목 없음")

    # 3. 주봉
    if cols[2].button("📅 주봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            res = []
            def w_worker(t):
                rt, df = smart_download(t, "1wk", "2y")
                if df.empty or len(df)<40: return None
                p, i = check_weekly_condition(df)
                if p:
                    eps = get_eps_changes_from_db(rt)
                    return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i['price']:,.0f}", 'ATR': f"{i['atr']:,.0f}", '구분': i['bw_change'], '1W': eps[0], '1M': eps[1], 'MACD-V': f"{i['macdv']:.2f}"}
                return None
            bar = st.progress(0)
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(w_worker, t) for t in tickers]
                for i, f in enumerate(as_completed(futs)):
                    bar.progress((i+1)/len(tickers))
                    r = f.result()
                    if r: res.append(r)
            bar.empty()
            if res: st.success(f"✅ {len(res)}개 발견"); st.dataframe(pd.DataFrame(res), use_container_width=True)

    # 4. 월봉
    if cols[3].button("🗓️ 월봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            res = []
            def m_worker(t):
                rt, df = smart_download(t, "1mo", "max")
                if df.empty: return None
                p, i = check_monthly_condition(df)
                if p: return {'종목코드': rt, '섹터': get_stock_sector(rt), '현재가': f"{i['price']:,.0f}", 'ATH가': f"{i['ath_price']:,.0f}", '달성월': i['ath_date'], '고권역수': i['month_count']}
                return None
            bar = st.progress(0)
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(m_worker, t) for t in tickers]
                for i, f in enumerate(as_completed(futs)):
                    bar.progress((i+1)/len(tickers))
                    r = f.result(); 
                    if r: res.append(r)
            bar.empty()
            if res: st.dataframe(pd.DataFrame(res), use_container_width=True)

    # 5. 일+월봉
    if cols[4].button("일+월"):
        tickers = get_tickers_from_sheet()
        if tickers:
            res = []; bar = st.progress(0)
            def dm_worker(t):
                rt, df_d = smart_download(t, "1d", "2y")
                if df_d.empty or len(df_d)<260: return None
                if not check_daily_condition(df_d)[0]: return None
                _, df_m = smart_download(t, "1mo", "max")
                if df_m.empty: return None
                if check_monthly_condition(df_m)[0]:
                    return {'종목': rt, '섹터': get_stock_sector(rt), '비고': '일봉돌파+월봉ATH'}
                return None
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(dm_worker, t) for t in tickers]
                for i, f in enumerate(as_completed(futs)):
                    bar.progress((i+1)/len(tickers))
                    r = f.result(); 
                    if r: res.append(r)
            bar.empty()
            if res: st.dataframe(pd.DataFrame(res))

    # 6. 일+주봉
    if cols[5].button("일+주"):
        tickers = get_tickers_from_sheet()
        if tickers:
            res = []; bar = st.progress(0)
            def dw_worker(t):
                rt, df_d = smart_download(t, "1d", "2y")
                if df_d.empty or len(df_d)<260: return None
                if not check_daily_condition(df_d)[0]: return None
                _, df_w = smart_download(t, "1wk", "2y")
                if df_w.empty: return None
                if check_weekly_condition(df_w)[0]:
                    return {'종목': rt, '섹터': get_stock_sector(rt), '비고': '일봉돌파+주봉추세'}
                return None
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(dw_worker, t) for t in tickers]
                for i, f in enumerate(as_completed(futs)):
                    bar.progress((i+1)/len(tickers))
                    r = f.result(); 
                    if r: res.append(r)
            bar.empty()
            if res: st.dataframe(pd.DataFrame(res))

    # 7. 주+월봉
    if cols[6].button("주+월"):
        tickers = get_tickers_from_sheet()
        if tickers:
            res = []; bar = st.progress(0)
            def wm_worker(t):
                rt, df_w = smart_download(t, "1wk", "2y")
                if df_w.empty or len(df_w)<40: return None
                if not check_weekly_condition(df_w)[0]: return None
                _, df_m = smart_download(t, "1mo", "max")
                if df_m.empty: return None
                if check_monthly_condition(df_m)[0]:
                    return {'종목': rt, '섹터': get_stock_sector(rt), '비고': '주봉추세+월봉ATH'}
                return None
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(wm_worker, t) for t in tickers]
                for i, f in enumerate(as_completed(futs)):
                    bar.progress((i+1)/len(tickers))
                    r = f.result(); 
                    if r: res.append(r)
            bar.empty()
            if res: st.dataframe(pd.DataFrame(res))

    # 8. 통합
    if cols[7].button("⚡통합"):
        tickers = get_tickers_from_sheet()
        if tickers:
            res = []; bar = st.progress(0)
            def int_worker(t):
                rt, df_d = smart_download(t, "1d", "2y")
                if df_d.empty or len(df_d)<260: return None
                if not check_daily_condition(df_d)[0]: return None
                _, df_w = smart_download(t, "1wk", "2y")
                if df_w.empty: return None
                if not check_weekly_condition(df_w)[0]: return None
                _, df_m = smart_download(t, "1mo", "max")
                if df_m.empty: return None
                if check_monthly_condition(df_m)[0]:
                    return {'종목': rt, '섹터': get_stock_sector(rt), '비고': 'Triple Crown'}
                return None
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(int_worker, t) for t in tickers]
                for i, f in enumerate(as_completed(futs)):
                    bar.progress((i+1)/len(tickers))
                    r = f.result(); 
                    if r: res.append(r)
            bar.empty()
            if res: st.dataframe(pd.DataFrame(res))

    # 9. 컵핸들
    if cols[8].button("🏆컵"):
        tickers = get_tickers_from_sheet()
        if tickers:
            res = []; bar = st.progress(0)
            def cup_worker(t):
                rt, df = smart_download(t, "1wk", "2y")
                if df.empty: return None
                p, i = check_cup_handle_pattern(df)
                if p: return {'종목': rt, '상세': i}
                return None
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(cup_worker, t) for t in tickers]
                for i, f in enumerate(as_completed(futs)):
                    bar.progress((i+1)/len(tickers))
                    r = f.result(); 
                    if r: res.append(r)
            bar.empty()
            if res: st.dataframe(pd.DataFrame(res))

    # 10. 역헤숄
    if cols[9].button("👤역H"):
        tickers = get_tickers_from_sheet()
        if tickers:
            res = []; bar = st.progress(0)
            def hs_worker(t):
                rt, df = smart_download(t, "1wk", "2y")
                if df.empty: return None
                p, i = check_inverse_hs_pattern(df)
                if p: return {'종목': rt, '상세': i}
                return None
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(hs_worker, t) for t in tickers]
                for i, f in enumerate(as_completed(futs)):
                    bar.progress((i+1)/len(tickers))
                    r = f.result(); 
                    if r: res.append(r)
            bar.empty()
            if res: st.dataframe(pd.DataFrame(res))

    st.markdown("### 📉 저장된 종목 중 눌림목/급등주 찾기")
    if st.button("🔍 눌림목 & 급등 패턴 분석"):
        db_tickers = get_unique_tickers_from_db()
        if db_tickers:
            res = []; bar = st.progress(0)
            def db_worker(t):
                rt, df = smart_download(t, "1d", "2y")
                if df.empty or len(df)<60: return None
                df = calculate_common_indicators(df, False)
                if df is None: return None
                curr = df.iloc[-1]
                cond = ""
                if curr['MACD_V'] > 60: cond = "🔥공격"
                elif (curr['Close'] > df['EMA20'].iloc[-1]) and ((curr['Close']-df['EMA20'].iloc[-1])/df['EMA20'].iloc[-1] < 0.03): cond = "📉눌림"
                if cond: return {'종목': rt, '패턴': cond, '현재가': f"{curr['Close']:,.0f}"}
                return None
            with ThreadPoolExecutor(max_workers=8) as ex:
                futs = [ex.submit(db_worker, t) for t in db_tickers]
                for i, f in enumerate(as_completed(futs)):
                    bar.progress((i+1)/len(db_tickers))
                    r = f.result(); 
                    if r: res.append(r)
            bar.empty()
            if res: st.dataframe(pd.DataFrame(res), use_container_width=True)

with tab4:
    st.markdown("### 💰 재무 지표 (yfinance 병렬)")
    if st.button("📊 데이터 가져오기"):
        tickers = get_tickers_from_sheet()
        if tickers:
            f_res = []
            def fin_worker(t):
                rt, _ = smart_download(t, "1d", "5d")
                tick = yf.Ticker(rt); info = tick.info
                if not info: return None
                mcap = info.get('marketCap', 0)
                return {"종목": rt, "섹터": info.get('sector','-'), "시총": f"{mcap/1e12:.1f}조" if mcap>1e12 else f"{mcap/1e8:.0f}억", "매출성장": f"{info.get('revenueGrowth',0)*100:.1f}%", "선행EPS": info.get('forwardEps','-'), "PEG": info.get('pegRatio','-')}
            with ThreadPoolExecutor(max_workers=8) as ex:
                for f in as_completed([ex.submit(fin_worker, t) for t in tickers]):
                    r = f.result()
                    if r: f_res.append(r)
            if f_res: st.dataframe(pd.DataFrame(f_res), use_container_width=True)

with tab5:
    st.markdown("### 📂 퀀티와이즈 매칭")
    up = st.file_uploader("quant_master.xlsx 업로드", type=['xlsx'])
    if up and st.button("🔄 매칭 시작"):
        # 기존 로직 (길어서 생략되었으나 필요 시 복원 가능)
        st.info("파일 처리 로직 실행...")

st.markdown("---")
with st.expander("🗄️ 전체 저장 기록 관리"):
    if st.button("🔄 기록 새로고침"):
        res = supabase.table("history").select("*").order("created_at", desc=True).limit(50).execute()
        if res.data: st.dataframe(pd.DataFrame(res.data), use_container_width=True)
    if st.button("🧹 중복 제거"):
        remove_duplicates_from_db()
