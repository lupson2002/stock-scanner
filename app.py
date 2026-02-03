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

def smart_download(ticker, interval="1d", period="2y"):
    if ':' in ticker: ticker = ticker.split(':')[-1]
    ticker = ticker.replace('/', '-')
    candidates = [ticker]
    if ticker.isdigit() and len(ticker) == 6:
        candidates = [f"{ticker}.KS", f"{ticker}.KQ", ticker]
    
    for t in candidates:
        try:
            for _ in range(3):
                df = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=False)
                if len(df) > 0:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    return t, df
                time.sleep(0.3)
        except:
            continue
    return ticker, pd.DataFrame()

# [중요] 종목 정보 캐싱 (섹터 정보 표시용으로만 사용)
@st.cache_data(ttl=3600*24) 
def get_ticker_info_safe(ticker):
    try:
        tick = yf.Ticker(ticker)
        for _ in range(3):
            try:
                meta = tick.info
                if meta: return meta
            except:
                time.sleep(0.5)
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

def find_extrema(df, order=3):
    prices = df['Close'].values
    peaks_idx = argrelextrema(prices, np.greater, order=order)[0]
    troughs_idx = argrelextrema(prices, np.less, order=order)[0]
    return peaks_idx, troughs_idx

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

# -----------------------------------------------------------------------------
# [VCP 패턴] 60일 기준, 20일 구간, 변동성 축소 확인
# -----------------------------------------------------------------------------
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

    # 2. 파동 (60일 기준, 20일씩 3구간)
    window = 60
    subset = df.iloc[-window:]
    p1 = subset.iloc[:20]    # 20일
    p2 = subset.iloc[20:40]  # 20일
    p3 = subset.iloc[40:]    # 20일
    
    range1 = (p1['High'].max() - p1['Low'].min()) / p1['High'].max()
    range2 = (p2['High'].max() - p2['Low'].min()) / p2['High'].max()
    range3 = (p3['High'].max() - p3['Low'].min()) / p3['High'].max()
    
    contraction = (range3 < range2) or (range2 < range1) or (range3 < 0.12)
    if not contraction: return False, None

    # 3. 셋업 (거래량)
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
        'status': status,
        'stop_loss': stop_loss,
        'target_price': target_price,
        'squeeze': "🔥" if df['TTM_Squeeze'].iloc[-1] else "-",
        'price': curr['Close'],
        'pivot': pivot_point # 차트 그리기용 피봇 반환
    }

# -----------------------------------------------------------------------------
# [NEW] 일봉 -> 주봉 변환 후 MACD 상태 계산 함수
# -----------------------------------------------------------------------------
def get_weekly_macd_status(daily_df):
    try:
        # 일봉 데이터를 주봉(금요일 기준)으로 리샘플링
        df_w = daily_df.resample('W-FRI').agg({
            'Close': 'last', 'High': 'max', 'Low': 'min', 'Volume': 'sum'
        }).dropna()
        
        if len(df_w) < 26: return "-"

        # 주봉 MACD (12, 26, 9) 계산
        ema12 = df_w['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_w['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        curr_macd = macd_line.iloc[-1]
        curr_sig = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_sig = signal_line.iloc[-2]
        
        # 상태 판별
        if curr_macd > curr_sig:
            # 이번주에 막 골든크로스 발생했는지 확인
            if prev_macd <= prev_sig:
                return "⚡GC (매수신호)"
            else:
                return "🔵 Buy (유지)"
        else:
            return "🔻 Sell (매도)"
    except:
        return "-"

# -----------------------------------------------------------------------------
# [NEW] VCP 차트 그리기 함수 (Plotly)
# -----------------------------------------------------------------------------
def plot_vcp_chart(df, ticker, info):
    # 최근 1년치 데이터만 표시
    df_plot = df.iloc[-252:].copy()
    
    fig = go.Figure()

    # 1. 캔들 차트
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'],
        name='Price'
    ))

    # 2. 이동평균선
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'].rolling(50).mean(), line=dict(color='green', width=1), name='SMA 50'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'].rolling(150).mean(), line=dict(color='blue', width=1), name='SMA 150'))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'].rolling(200).mean(), line=dict(color='red', width=1), name='SMA 200'))

    # 3. 피봇 포인트 (돌파 기준) - 빨간 점선
    fig.add_hline(y=info['pivot'], line_dash="dot", line_color="red", annotation_text="Pivot (Breakout)")

    # 4. 스탑로스 (손절 라인) - 파란 점선
    fig.add_hline(y=info['stop_loss'], line_dash="dot", line_color="blue", annotation_text="Stop Loss")

    fig.update_layout(
        title=f"{ticker} - VCP Analysis Chart",
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark" # 다크 모드
    )
    return fig

# ... (나머지 체크 함수들: check_daily_condition 등 기존 유지) ...
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
            'price': curr['Close'], 
            'atr': curr['ATR14'], 
            'high_date': high_52_date, 
            'prev_date': prev_date, 
            'diff_days': diff_days, 
            'bw_curr': curr['BW50'], 
            'macdv': curr['MACD_V'], 
            'squeeze': "🔥TTM Squeeze" if squeeze_on else "-" 
        }
    return False, None

def check_weekly_condition(df):
    if len(df) < 60: return False, None
    df = calculate_common_indicators(df, is_weekly=True)
    if df is None: return False, None
    
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    cond_bb = curr['Close'] > curr['BB_UP']
    cond_macd = (prev['MACD_Line'] <= prev['MACD_Signal']) and (curr['MACD_Line'] > curr['MACD_Signal'])
    
    if cond_bb or cond_macd:
        bw_past = df['BandWidth'].iloc[-21]
        bw_change = "감소" if bw_past > curr['BandWidth'] else "증가"
        
        return True, {
            'price': curr['Close'], 
            'atr': curr['ATR14'], 
            'bw_curr': curr['BandWidth'], 
            'bw_past': bw_past, 
            'bw_change': f"{bw_change} (BB/MACD)", 
            'macdv': curr['MACD_V']
        }
    return False, None

def check_monthly_condition(df):
    if len(df) < 12: return False, None
    ath_price = df['High'].max()
    curr_price = df['Close'].iloc[-1]
    if curr_price >= ath_price * 0.90:
        ath_idx = df['High'].idxmax()
        month_count = (df['Close'] >= ath_price * 0.90).sum()
        return True, {'price': curr_price, 'ath_price': ath_price, 'ath_date': ath_idx.strftime('%Y-%m'), 'month_count': month_count}
    return False, None

def analyze_momentum_strategy(target_list, type_name="ETF"):
    if not target_list: return pd.DataFrame()
    st.write(f"📊 총 {len(target_list)}개 {type_name} 분석 중...")
    results = []; pbar = st.progress(0)
    for i, (t, n) in enumerate(target_list):
        pbar.progress((i+1)/len(target_list))
        rt, df = smart_download(t, "1d", "2y")
        if len(df)<30: continue
        df = calculate_daily_indicators(df)
        if df is None: continue
        c = df['Close']; curr=c.iloc[-1]
        squeeze_on = df['TTM_Squeeze'].iloc[-5:].any() if 'TTM_Squeeze' in df.columns else False
        ema20=c.ewm(span=20).mean(); ema50=c.ewm(span=50).mean(); ema60=c.ewm(span=60).mean()
        ema100=c.ewm(span=100).mean(); ema200=c.ewm(span=200).mean()
        bb_up = df['BB50_UP']; dc_h = df['Donchian_High_50'] 
        macdv = df['MACD_V']; atr = df['ATR14'].iloc[-1]
        bb_bk = "O" if (c>bb_up).iloc[-3:].any() else "-"
        dc_bk = "O" if (c>dc_h).iloc[-3:].any() else "-"
        align = "⭐ 정배열" if (curr>ema20.iloc[-1] and curr>ema60.iloc[-1] and curr>ema100.iloc[-1] and curr>ema200.iloc[-1]) else "-"
        long_tr = "📈 상승" if (ema60.iloc[-1]>ema100.iloc[-1]>ema200.iloc[-1]) else "-"
        
        # [변경] 전략 3: 평균 모멘텀 (Smoothed)
        r12 = c.pct_change(252).iloc[-1] if len(c) > 252 else 0
        r6  = c.pct_change(126).iloc[-1] if len(c) > 126 else 0
        r3  = c.pct_change(63).iloc[-1] if len(c) > 63 else 0
        r1  = c.pct_change(21).iloc[-1] if len(c) > 21 else 0
        
        avg_long_term = (r12 + r6) / 2
        score = ((avg_long_term - r3) + r1) * 100
        
        if len(df) >= 252:
            win_52 = df.iloc[-252:]
            high_idx = win_52['Close'].idxmax()
            high_52_date = high_idx.strftime('%Y-%m-%d')
            prev_win = win_52[win_52.index < high_idx]
            if len(prev_win) > 0:
                prev_idx = prev_win['Close'].idxmax()
                prev_date = prev_idx.strftime('%Y-%m-%d')
                diff_days = (high_idx - prev_idx).days
            else:
                prev_date = "-"; diff_days = 0
        else:
            high_52_date = "-"; prev_date = "-"; diff_days = 0
        results.append({
            f"{type_name}": f"{rt} ({n})", 
            "모멘텀점수": score, 
            "스퀴즈": "🔥" if squeeze_on else "-", 
            "BB(50,2)돌파": bb_bk, 
            "돈키언(50)돌파": dc_bk, 
            "정배열": align, 
            "장기추세": long_tr, 
            "MACD-V": f"{macdv.iloc[-1]:.2f}", 
            "ATR": f"{atr:.2f}",
            "현52주신고가일": high_52_date,
            "전52주신고가일": prev_date,
            "차이일": f"{diff_days}일",
            "현재가": curr
        })
    pbar.empty()
    if results:
        df_res = pd.DataFrame(results).sort_values("모멘텀점수", ascending=False)
        df_res['모멘텀점수'] = df_res['모멘텀점수'].apply(lambda x: f"{x:.2f}")
        df_res['현재가'] = df_res['현재가'].apply(lambda x: f"{x:,.2f}")
        return df_res
    return pd.DataFrame()

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

# -----------------------------------------------------------------------------
# [NEW] 나침판용 전략 분석 함수 (최적화)
# -----------------------------------------------------------------------------
def get_compass_signal():
    # 1. 설정
    OFFENSE = ["QQQ", "SPY", "EFA", "GLD", "EEM"]
    CASH = "BIL"
    ALL_TICKERS = list(set(OFFENSE + [CASH]))
    
    # 2. 데이터 다운로드 (최근 2년치만)
    try:
        data = yf.download(ALL_TICKERS, period="2y", progress=False, auto_adjust=False)['Close']
        if data.empty: return None, "데이터 없음"
    except:
        return None, "다운로드 실패"

    # 3. 월봉 리샘플링
    monthly_data = data.resample('ME').last()
    
    if len(monthly_data) < 13: return None, "데이터 부족 (최소 13개월 필요)"

    # 4. 지표 계산 (마지막 시점 기준)
    # pct_change는 (현재 - 과거) / 과거
    m12 = monthly_data.pct_change(12).iloc[-1]
    m6  = monthly_data.pct_change(6).iloc[-1]
    m3  = monthly_data.pct_change(3).iloc[-1]
    m1  = monthly_data.pct_change(1).iloc[-1]

    # 5. 전략 3 (Smoothed) 스코어 계산
    # 공식: ((12M + 6M) / 2 - 3M) + 1M
    scores = {}
    for ticker in OFFENSE:
        if ticker not in m12.index: continue
        
        r12 = m12[ticker]
        r6  = m6[ticker]
        r3  = m3[ticker]
        r1  = m1[ticker]
        
        # NaN 체크
        if np.isnan(r12): continue
        
        avg_long = (r12 + r6) / 2
        score = (avg_long - r3) + r1
        scores[ticker] = {
            "Score": score * 100,
            "12M_Trend": r12 # 절대 모멘텀 확인용
        }
    
    if not scores: return None, "계산 불가"

    # 6. 순위 산정
    df_scores = pd.DataFrame(scores).T
    df_scores = df_scores.sort_values("Score", ascending=False)
    
    best_ticker = df_scores.index[0]
    best_score = df_scores.iloc[0]['Score']
    best_trend = df_scores.iloc[0]['12M_Trend']
    
    # 7. 포지션 결정 (절대 모멘텀 필터)
    final_position = best_ticker if (best_score > 0 and best_trend > 0) else CASH
    
    return df_scores, final_position

# ==========================================
# 5. 메인 실행 화면
# ==========================================

# [변경] 탭 순서 변경: 나침판(tab_compass)을 맨 앞으로
tab_compass, tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧭 나침판", "🌍 섹터", "🏳️ 국가", "📊 기술적 분석", "💰 재무분석", "📂 엑셀 데이터 매칭"])

# -----------------------------------------------------------------------------
# [탭 1] 나침판 (가장 왼쪽으로 이동)
# -----------------------------------------------------------------------------
with tab_compass:
    st.markdown("### 🧭 투자 나침판 (Smoothed Momentum Strategy)")
    st.markdown("""
    이 탭은 **'전략 3 (평균 모멘텀)'** 로직을 기반으로 **현재 시점(Today)**에서 가장 매력적인 자산을 알려줍니다.
    
    **전략 로직:**
    1. **후보군:** QQQ(나스닥), SPY(S&P500), EFA(선진국), GLD(금), EEM(신흥국)
    2. **점수 산출:** `((12개월+6개월)/2 - 3개월) + 1개월` 수익률
    3. **방어 기제:** 1등 종목의 12개월 수익률이 마이너스면 **현금(BIL)** 보유
    """)
    
    if st.button("🚀 지금 어디에 투자해야 할까? (분석 시작)", type="primary"):
        with st.spinner("최근 2년치 데이터를 분석하여 방향을 잡는 중입니다..."):
            df_result, position = get_compass_signal()
            
            if df_result is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🎯 현재 추천 포지션: **{position}**")
                    if position == "BIL":
                        st.caption("🚨 시장 상황이 좋지 않습니다. 현금(초단기채)으로 대피하세요.")
                    else:
                        st.caption(f"🚀 상승 모멘텀이 가장 강한 **{position}**에 올라타세요!")
                
                with col2:
                    top_score = df_result.iloc[0]['Score']
                    st.metric("1등 모멘텀 점수", f"{top_score:.2f}점")

                st.markdown("---")
                st.markdown("#### 📊 자산별 상세 스코어 (높은 순)")
                
                df_display = df_result.copy()
                df_display['Score'] = df_display['Score'].apply(lambda x: f"{x:.2f}")
                df_display['12M_Trend'] = df_display['12M_Trend'].apply(lambda x: f"{x*100:.1f}%")
                df_display.columns = ["모멘텀 점수", "12개월 추세(절대)"]
                
                st.dataframe(df_display, use_container_width=True)
                
                st.info("""
                **해석 가이드:**
                * **모멘텀 점수:** 높을수록 상승세가 견고하고 최근 눌림목을 잘 소화한 종목입니다.
                * **12개월 추세:** 이 값이 마이너스(-)라면, 점수가 아무리 높아도 **하락장**으로 간주하여 현금(BIL)을 추천합니다.
                """)
            else:
                st.error(f"분석 실패: {position}")

# -----------------------------------------------------------------------------
# [탭 2] 섹터 (두 번째로 이동)
# -----------------------------------------------------------------------------
with tab1:
    cols = st.columns(12) 
    if cols[0].button("🌍 섹터"):
        etfs = get_etfs_from_sheet()
        if not etfs: st.warning("ETF 목록 없음")
        else:
            st.info("ETF 섹터 분석 중 (모멘텀 전략 3: Smoothed)...")
            res = analyze_momentum_strategy(etfs, "ETF")
            if not res.empty: st.dataframe(res, use_container_width=True)
            else: st.warning("데이터 부족")

# -----------------------------------------------------------------------------
# [탭 3] 국가 (기존 위치 유지)
# -----------------------------------------------------------------------------
with tab2:
    cols = st.columns(12)
    if cols[0].button("🏳️ 국가"):
        tickers = get_country_etfs_from_sheet()
        if not tickers: st.warning("국가 ETF 목록 없음")
        else:
            st.info(f"[국가 ETF] {len(tickers)}개 모멘텀(전략 3) 분석 시작...")
            res = analyze_momentum_strategy(tickers, "국가ETF")
            if not res.empty:
                st.success(f"[국가] {len(res)}개 분석 완료!")
                st.dataframe(res, use_container_width=True)
            else: st.warning("데이터 부족")

# -----------------------------------------------------------------------------
# [탭 4] 기술적 분석 (VCP 포함)
# -----------------------------------------------------------------------------
with tab3:
    cols = st.columns(12)
    
    # [NEW] VCP 버튼 (차트 검증 + 정렬 수정 + 개별주 필터링 삭제 + 2열 그리드 차트 + 주봉MACD)
    if cols[0].button("🌪️ VCP"):
        tickers = get_tickers_from_sheet()
        if not tickers: st.warning("종목 리스트(TGT) 없음")
        else:
            st.info(f"구글 시트에서 총 **{len(tickers)}**개 종목을 불러왔습니다.")
            
            # 진행상황 표시용
            status_text = st.empty()
            bar = st.progress(0)
            
            res = []
            chart_data_cache = {}
            
            # 카운터 변수
            count_total = len(tickers)
            
            for i, t in enumerate(tickers):
                status_text.text(f"⏳ 진행 중... ({i+1}/{count_total}) - {t}")
                bar.progress((i+1)/len(tickers))
                
                # 1. 데이터 다운로드 (즉시 시도)
                # 티커 클렌징 (공백 제거)
                t_clean = t.strip()
                
                try:
                    # smart_download가 내부적으로 ticker, ticker.KS 등 시도함
                    final_ticker, df = smart_download(t_clean, "1d", "2y")
                except:
                    continue

                if len(df) < 250: continue

                # 2. VCP 패턴 체크
                passed, info = check_vcp_pattern(df)
                if passed:
                    eps1w, eps1m, eps3m = get_eps_changes_from_db(final_ticker)
                    
                    # [NEW] 주봉 MACD 상태 계산
                    weekly_macd_status = get_weekly_macd_status(df)
                    
                    # 섹터 정보 (표시용으로만 가져오기)
                    sector = get_stock_sector(final_ticker)
                    
                    chart_data_cache[final_ticker] = {'df': df, 'info': info}
                    
                    res.append({
                        '종목코드': final_ticker, '섹터': sector, '현재가': f"{info['price']:,.0f}",
                        '비고': info['status'], 
                        '주봉MACD': weekly_macd_status, # [NEW] 주봉 MACD 컬럼 추가
                        '손절가': f"{info['stop_loss']:,.0f}", 
                        '목표가(3R)': f"{info['target_price']:,.0f}",
                        '스퀴즈': info['squeeze'],
                        '1W변화': eps1w, '1M변화': eps1m, '3M변화': eps3m,
                        'Pivot': f"{info['pivot']:,.0f}" 
                    })
            bar.empty()
            status_text.empty() 
            
            st.success(f"✅ 분석 완료! 총 {count_total}개 전체 종목을 검사했습니다.")
            
            if res:
                # [수정] 비고 열을 내림차순(ascending=False)으로 정렬하여 4단계가 위로 오게 함
                df_res = pd.DataFrame(res).sort_values("비고", ascending=False)
                st.dataframe(df_res, use_container_width=True)
                
                # [NEW] 4단계 돌파 종목 자동 차트 갤러리 (2열 그리드)
                breakout_targets = [r for r in res if "4단계" in r['비고']]

                if breakout_targets:
                    st.markdown("---")
                    st.markdown("### 🚀 돌파 종목 차트 갤러리 (Step 4)")
                    
                    # 그리드 레이아웃 생성
                    for i in range(0, len(breakout_targets), 2):
                        c1, c2 = st.columns(2)
                        
                        # 왼쪽 차트
                        item1 = breakout_targets[i]
                        ticker1 = item1['종목코드']
                        if ticker1 in chart_data_cache:
                            cached1 = chart_data_cache[ticker1]
                            fig1 = plot_vcp_chart(cached1['df'], ticker1, cached1['info'])
                            c1.plotly_chart(fig1, use_container_width=True)
                            c1.caption(f"**{ticker1}** ({item1['섹터']}) | {item1['주봉MACD']} | Pivot: {item1['Pivot']}")

                        # 오른쪽 차트 (홀수 개일 경우 에러 방지)
                        if i + 1 < len(breakout_targets):
                            item2 = breakout_targets[i+1]
                            ticker2 = item2['종목코드']
                            if ticker2 in chart_data_cache:
                                cached2 = chart_data_cache[ticker2]
                                fig2 = plot_vcp_chart(cached2['df'], ticker2, cached2['info'])
                                c2.plotly_chart(fig2, use_container_width=True)
                                c2.caption(f"**{ticker2}** ({item2['섹터']}) | {item2['주봉MACD']} | Pivot: {item2['Pivot']}")
                
                save_to_supabase(res, "VCP_Pattern")
            else: st.warning("VCP 조건(추세+수렴)을 만족하는 종목이 없습니다.")

    if cols[1].button("🚀 일봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info(f"[일봉 5-Factor] {len(tickers)}개 분석 시작...")
            bar = st.progress(0); res = []
            for i, t in enumerate(tickers):
                bar.progress((i+1)/len(tickers))
                rt, df = smart_download(t, "1d", "2y")
                passed, info = check_daily_condition(df)
                if passed:
                    eps1w, eps1m, eps3m = get_eps_changes_from_db(rt)
                    sector = get_stock_sector(rt)
                    res.append({
                        '종목코드': rt, '섹터': sector, '현재가': f"{info['price']:,.0f}",
                        'ATR(14)': f"{info['atr']:,.0f}", '스퀴즈': info['squeeze'],
                        '1W변화': eps1w, '1M변화': eps1m, '3M변화': eps3m,
                        '현52주신고가일': info['high_date'], '전52주신고가일': info['prev_date'],
                        '차이일': f"{info['diff_days']}일", 'BW현재': f"{info['bw_curr']:.4f}",
                        'MACD-V': f"{info['macdv']:.2f}", 'BW_Value': f"{info['bw_curr']:.4f}", 'MACD_V_Value': f"{info['macdv']:.2f}"
                    })
            bar.empty()
            if res:
                st.success(f"[일봉] {len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res).drop(columns=['BW_Value', 'MACD_V_Value']))
                save_to_supabase(res, "Daily_5Factor")
            else: st.warning("조건 만족 없음")

    if cols[2].button("📅 주봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info(f"[주봉] {len(tickers)}개 분석 시작...")
            bar = st.progress(0); res = []
            for i, t in enumerate(tickers):
                bar.progress((i+1)/len(tickers))
                rt, df = smart_download(t, "1wk", "2y")
                passed, info = check_weekly_condition(df)
                if passed:
                    eps1w, eps1m, eps3m = get_eps_changes_from_db(rt)
                    sector = get_stock_sector(rt)
                    res.append({
                        '종목코드': rt, '섹터': sector, '현재가': f"{info['price']:,.0f}",
                        'ATR(14주)': f"{info['atr']:,.0f}", 'BW현재': f"{info['bw_curr']:.4f}",
                        '1W변화': eps1w, '1M변화': eps1m, '3M변화': eps3m,
                        'BW(20주전)': f"{info['bw_past']:.4f}", 'BW변화': info['bw_change'],
                        'MACD-V': f"{info['macdv']:.2f}", 'BW_Value': f"{info['bw_curr']:.4f}", 'MACD_V_Value': f"{info['macdv']:.2f}"
                    })
            bar.empty()
            if res:
                st.success(f"[주봉] {len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res).drop(columns=['BW_Value', 'MACD_V_Value']))
                save_to_supabase(res, "Weekly")
            else: st.warning("조건 만족 없음")

    if cols[3].button("🗓️ 월봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info(f"[월봉] {len(tickers)}개 분석 시작...")
            bar = st.progress(0); res = []
            for i, t in enumerate(tickers):
                bar.progress((i+1)/len(tickers))
                rt, df = smart_download(t, "1mo", "max")
                passed, info = check_monthly_condition(df)
                if passed:
                    eps1w, eps1m, eps3m = get_eps_changes_from_db(rt)
                    sector = get_stock_sector(rt)
                    res.append({
                        '종목코드': rt, '섹터': sector, '현재가': f"{info['price']:,.0f}",
                        'ATH최고가': f"{info['ath_price']:,.0f}", 'ATH달성월': info['ath_date'],
                        '1W변화': eps1w, '1M변화': eps1m, '3M변화': eps3m,
                        '고권역(월수)': f"{info['month_count']}개월",
                        '현52주신고가일': info['ath_date'], 'BW_Value': str(info['month_count']), 'MACD_V_Value': "0"
                    })
            bar.empty()
            if res:
                st.success(f"[월봉] {len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res).drop(columns=['현52주신고가일', 'BW_Value', 'MACD_V_Value'], errors='ignore'))
                save_to_supabase(res, "Monthly_ATH")
            else: st.warning("조건 만족 없음")

    if cols[4].button("일+월봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info("일봉+월봉 분석 중...")
            bar = st.progress(0); res = []
            for i, t in enumerate(tickers):
                bar.progress((i+1)/len(tickers))
                rt, df_d = smart_download(t, "1d", "2y")
                pass_d, info_d = check_daily_condition(df_d)
                if not pass_d: continue
                _, df_m = smart_download(t, "1mo", "max")
                pass_m, info_m = check_monthly_condition(df_m)
                if not pass_m: continue
                sector = get_stock_sector(rt)
                eps1w, eps1m, eps3m = get_eps_changes_from_db(rt)
                res.append({
                    '종목코드': rt, '섹터': sector, '현재가': f"{info_d['price']:,.0f}",
                    '스퀴즈': info_d['squeeze'], 'ATH달성월': info_m['ath_date'],
                    '1W변화': eps1w, '1M변화': eps1m, '3M변화': eps3m,
                    '고권역(월수)': f"{info_m['month_count']}개월",
                    '현52주신고가일': info_d['high_date'], '전52주신고가일': info_d['prev_date'],
                    '차이일': f"{info_d['diff_days']}일", 'BW_Value': str(info_m['month_count']), 'MACD_V_Value': f"{info_d['macdv']:.2f}"
                })
            bar.empty()
            if res:
                st.success(f"[일+월봉] {len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res))
                save_to_supabase(res, "Daily_Monthly")
            else: st.warning("조건 만족 없음")

    if cols[5].button("일+주봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info("일봉+주봉 분석 중...")
            bar = st.progress(0); res = []
            for i, t in enumerate(tickers):
                bar.progress((i+1)/len(tickers))
                rt, df_d = smart_download(t, "1d", "2y")
                pass_d, info_d = check_daily_condition(df_d)
                if not pass_d: continue
                _, df_w = smart_download(t, "1wk", "2y")
                pass_w, info_w = check_weekly_condition(df_w)
                if not pass_w: continue
                sector = get_stock_sector(rt)
                eps1w, eps1m, eps3m = get_eps_changes_from_db(rt)
                res.append({
                    '종목코드': rt, '섹터': sector, '현재가': f"{info_d['price']:,.0f}",
                    '스퀴즈': info_d['squeeze'], '주봉BW': f"{info_w['bw_curr']:.4f}", '주봉BW변화': info_w['bw_change'],
                    '1W변화': eps1w, '1M변화': eps1m, '3M변화': eps3m,
                    '현52주신고가일': info_d['high_date'], '전52주신고가일': info_d['prev_date'],
                    '차이일': f"{info_d['diff_days']}일", 'BW_Value': f"{info_w['bw_curr']:.4f}", 'MACD_V_Value': f"{info_d['macdv']:.2f}"
                })
            bar.empty()
            if res:
                st.success(f"[일+주봉] {len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res))
                save_to_supabase(res, "Daily_Weekly")
            else: st.warning("조건 만족 없음")

    if cols[6].button("주+월봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info("주봉+월봉 분석 중...")
            bar = st.progress(0); res = []
            for i, t in enumerate(tickers):
                bar.progress((i+1)/len(tickers))
                rt, df_w = smart_download(t, "1wk", "2y")
                pass_w, info_w = check_weekly_condition(df_w)
                if not pass_w: continue
                _, df_m = smart_download(t, "1mo", "max")
                pass_m, info_m = check_monthly_condition(df_m)
                if not pass_m: continue
                sector = get_stock_sector(rt)
                eps1w, eps1m, eps3m = get_eps_changes_from_db(rt)
                res.append({
                    '종목코드': rt, '섹터': sector, '현재가': f"{info_w['price']:,.0f}",
                    '주봉BW': f"{info_w['bw_curr']:.4f}", '주봉BW변화': info_w['bw_change'],
                    '1W변화': eps1w, '1M변화': eps1m, '3M변화': eps3m,
                    'ATH달성월': info_m['ath_date'], '고권역(월수)': f"{info_m['month_count']}개월",
                    '현52주신고가일': info_m['ath_date'], 'BW_Value': f"{info_w['bw_curr']:.4f}", 'MACD_V_Value': f"{info_w['macdv']:.2f}"
                })
            bar.empty()
            if res:
                st.success(f"[주+월봉] {len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res))
                save_to_supabase(res, "Weekly_Monthly")
            else: st.warning("조건 만족 없음")

    if cols[7].button("⚡ 통합"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info("통합(일+주+월) 분석 중...")
            bar = st.progress(0); res = []
            for i, t in enumerate(tickers):
                bar.progress((i+1)/len(tickers))
                rt, df_d = smart_download(t, "1d", "2y")
                pass_d, info_d = check_daily_condition(df_d)
                if not pass_d: continue
                _, df_w = smart_download(t, "1wk", "2y")
                pass_w, info_w = check_weekly_condition(df_w)
                if not pass_w: continue
                _, df_m = smart_download(t, "1mo", "max")
                pass_m, info_m = check_monthly_condition(df_m)
                if not pass_m: continue
                sector = get_stock_sector(rt)
                eps1w, eps1m, eps3m = get_eps_changes_from_db(rt)
                res.append({
                    '종목코드': rt, '섹터': sector, '현재가': f"{info_d['price']:,.0f}",
                    'ATH최고가': f"{info_m['ath_price']:,.0f}", 'ATH달성월': info_m['ath_date'],
                    '해당월수': f"{info_m['month_count']}개월", '스퀴즈': info_d['squeeze'],
                    '1W변화': eps1w, '1M변화': eps1m, '3M변화': eps3m,
                    '현52주신고가일': info_d['high_date'], '전52주신고가일': info_d['prev_date'],
                    '차이일': f"{info_d['diff_days']}일", '주봉BW': f"{info_w['bw_curr']:.4f}",
                    '주봉BW변화': info_w['bw_change'], 'MACD-V': f"{info_w['macdv']:.2f}",
                    'BW_Value': f"{info_w['bw_curr']:.4f}", 'MACD_V_Value': f"{info_w['macdv']:.2f}"
                })
            bar.empty()
            if res:
                st.success(f"⚡ 통합 분석 완료! {len(res)}개 발견")
                st.dataframe(pd.DataFrame(res).drop(columns=['BW_Value', 'MACD_V_Value']))
                save_to_supabase(res, "Integrated_Triple")
            else: st.warning("3가지 조건을 모두 만족하는 종목이 없습니다.")

    if cols[8].button("🏆 컵핸들"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info("[컵핸들] 분석 중...")
            bar = st.progress(0); res = []
            for i, t in enumerate(tickers):
                bar.progress((i+1)/len(tickers))
                rt, df = smart_download(t, "1wk", "2y")
                pass_c, info = check_cup_handle_pattern(df)
                if pass_c:
                    df = calculate_common_indicators(df, True)
                    if df is None: continue 
                    curr = df.iloc[-1]
                    sector = get_stock_sector(rt)
                    eps1w, eps1m, eps3m = get_eps_changes_from_db(rt)
                    res.append({
                        '종목코드': rt, '섹터': sector, '현재가': f"{curr['Close']:,.0f}",
                        '패턴상세': f"깊이:{info['depth']}", '돌파가격': info['pivot'],
                        '1W변화': eps1w, '1M변화': eps1m, '3M변화': eps3m,
                        'BW_Value': f"{curr['BandWidth']:.4f}", 'MACD_V_Value': f"{curr['MACD_V']:.2f}"
                    })
            bar.empty()
            if res:
                st.success(f"[컵핸들] {len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res))
                save_to_supabase(res, "CupHandle")
            else: st.warning("조건 만족 없음")

    if cols[9].button("👤 역H&S"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info("[역H&S] 분석 중...")
            bar = st.progress(0); res = []
            for i, t in enumerate(tickers):
                bar.progress((i+1)/len(tickers))
                rt, df = smart_download(t, "1wk", "2y")
                pass_h, info = check_inverse_hs_pattern(df)
                if pass_h:
                    df = calculate_common_indicators(df, True)
                    if df is None: continue 
                    curr = df.iloc[-1]
                    sector = get_stock_sector(rt)
                    eps1w, eps1m, eps3m = get_eps_changes_from_db(rt)
                    res.append({
                        '종목코드': rt, '섹터': sector, '현재가': f"{curr['Close']:,.0f}",
                        '넥라인': info['Neckline'], '거래량급증': info['Vol_Ratio'],
                        '1W변화': eps1w, '1M변화': eps1m, '3M변화': eps3m,
                        'BW_Value': f"{curr['BandWidth']:.4f}", 'MACD_V_Value': f"{curr['MACD_V']:.2f}"
                    })
            bar.empty()
            if res:
                st.success(f"[역H&S] {len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res))
                save_to_supabase(res, "InverseHS")
            else: st.warning("조건 만족 없음")

    st.markdown("### 📉 저장된 종목 중 눌림목/급등주 찾기")
    if st.button("🔍 눌림목 & 급등 패턴 분석"):
        db_tickers = get_unique_tickers_from_db()
        if not db_tickers: st.warning("DB 데이터 없음")
        else:
            st.info(f"{len(db_tickers)}개 종목 재분석 중...")
            bar = st.progress(0); res = []
            for i, t in enumerate(db_tickers):
                bar.progress((i+1)/len(db_tickers))
                rt, df = smart_download(t, "1d", "2y")
                try:
                    df = calculate_common_indicators(df, False)
                    if df is None: continue
                    curr = df.iloc[-1]
                    cond = ""
                    if curr['MACD_V'] > 60: cond = "🔥 공격적 추세"
                    ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
                    if (curr['Close'] > ema20) and ((curr['Close']-ema20)/ema20 < 0.03): cond = "📉 20일선 눌림목"
                    if (curr['Close'] > curr['EMA200']) and (-100 <= curr['MACD_V'] <= -50): cond = "🧲 MACD-V 과매도"
                    if cond:
                        eps1w, eps1m, eps3m = get_eps_changes_from_db(rt)
                        res.append({
                            '종목코드': rt, '패턴': cond, '현재가': f"{curr['Close']:,.0f}",
                            '1W변화': eps1w, '1M변화': eps1m, '3M변화': eps3m,
                            'MACD-V': f"{curr['MACD_V']:.2f}", 'EMA20': f"{ema20:,.0f}"
                        })
                except: continue
            bar.empty()
            if res:
                st.success(f"{len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res), use_container_width=True)
            else: st.warning("조건 만족 없음")

with tab4:
    st.markdown("### 💰 재무 지표 분석 & EPS Trend (yfinance)")
    if st.button("📊 재무 지표 가져오기"):
        tickers = get_tickers_from_sheet()
        if not tickers: st.error("티커 없음")
        else:
            bar = st.progress(0); f_res = []
            for i, t in enumerate(tickers):
                bar.progress((i + 1) / len(tickers))
                real_ticker, _ = smart_download(t, "1d", "5d") 
                try:
                    tick = yf.Ticker(real_ticker)
                    info = tick.info
                    if not info: continue
                    mkt_cap = info.get('marketCap', 0)
                    mkt_cap_str = f"{mkt_cap/1000000000000:.1f}조" if mkt_cap > 1000000000000 else f"{mkt_cap/100000000:.0f}억" if mkt_cap else "-"
                    rev_growth = info.get('revenueGrowth', 0)
                    rev_str = f"{rev_growth*100:.1f}%" if rev_growth else "-"
                    eps_growth = info.get('earningsGrowth', 0)
                    eps_growth_str = f"{eps_growth*100:.1f}%" if eps_growth else "-"
                    fwd_eps = info.get('forwardEps', '-')
                    peg = info.get('pegRatio', '-')
                    try:
                        trend_data = tick.eps_trend
                        if trend_data:
                            curr_year_data = trend_data[0] 
                            curr_est = curr_year_data.get('current', 0)
                            ago30 = curr_year_data.get('30daysAgo', 0)
                            ago90 = curr_year_data.get('90daysAgo', 0)
                            trend_30 = "↗️" if curr_est > ago30 else "↘️" if curr_est < ago30 else "-"
                            trend_90 = "↗️" if curr_est > ago90 else "↘️" if curr_est < ago90 else "-"
                            eps_trend_str = f"30일{trend_30} | 90일{trend_90}"
                        else: eps_trend_str = "-"
                    except: eps_trend_str = "-"
                    rec = info.get('recommendationKey', '-').upper().replace('_', ' ')
                    target = info.get('targetMeanPrice')
                    curr_p = info.get('currentPrice', 0)
                    upside = f"{(target - curr_p) / curr_p * 100:.1f}%" if (target and curr_p) else "-"
                    eps1w, eps1m, eps3m = get_eps_changes_from_db(real_ticker)
                    f_res.append({
                        "종목": real_ticker, "섹터": info.get('sector', '-'), "산업": info.get('industry', '-'),
                        "시가총액": mkt_cap_str, "매출성장(YoY)": rev_str, "EPS성장(YoY)": eps_growth_str,
                        "선행EPS": fwd_eps, "PEG": peg, "EPS추세(올해)": eps_trend_str,
                        "1W변화": eps1w, "1M변화": eps1m, "3M변화": eps3m,
                        "투자의견": rec, "상승여력": upside
                    })
                except Exception as e: continue
            bar.empty()
            if f_res:
                df_fin = pd.DataFrame(f_res)
                st.success(f"✅ 총 {len(df_fin)}개 기업 재무/EPS 분석 완료")
                st.dataframe(df_fin, use_container_width=True)
            else: st.warning("데이터를 가져오지 못했습니다.")

with tab5:
    st.markdown("### 📂 엑셀 데이터 매칭 (퀀티와이즈 DB 연동)")
    col_upload, col_reset = st.columns([3, 1])
    with col_upload:
        uploaded_file = st.file_uploader("📥 quant_master.xlsx 파일을 드래그하여 업로드하세요", type=['xlsx'])
    with col_reset:
        st.write("") 
        st.write("") 
        if st.button("🗑️ [주의] DB 초기화 (전체 삭제)", type="primary"):
            try:
                supabase.table("quant_data").delete().neq("id", 0).execute()
                st.success("DB가 초기화되었습니다.")
                fetch_latest_quant_data_from_db.clear()
            except Exception as e:
                st.error(f"초기화 실패: {e}")

    show_debug_log = st.checkbox("🔍 디버깅 로그 보기")

    def parse_sheet_ticker_value(sheet_df, allowed_tickers, debug_mode=False):
        extracted = {}
        for index, row in sheet_df.iterrows():
            try:
                raw_ticker = str(row[0]).strip()
                if not raw_ticker or raw_ticker.lower() in ['code', 'ticker', 'nan', 'item type', 'comparison date']: continue
                norm_ticker = normalize_ticker_for_db_storage(raw_ticker)
                if debug_mode and "RKLB" in norm_ticker: st.write(f"📢 [DEBUG] 발견된 티커: {raw_ticker} -> 정규화: {norm_ticker}")
                if norm_ticker not in allowed_tickers: continue
                val = row[3] 
                if pd.isna(val): final_val = "-"
                else:
                    final_val = str(val).strip()
                    if final_val.lower() == 'nan' or final_val == "": final_val = "-"
                extracted[norm_ticker] = final_val
            except Exception: continue
        return extracted

    if uploaded_file and st.button("🔄 DB 업로드 및 분석 시작"):
        try:
            st.info("구글 시트에서 관리 종목(TGT) 목록을 불러오는 중...")
            tgt_stocks = get_tickers_from_sheet()
            tgt_etfs = [x[0] for x in get_etfs_from_sheet()]
            tgt_countries = [x[0] for x in get_country_etfs_from_sheet()]
            raw_targets = set(tgt_stocks + tgt_etfs + tgt_countries)
            allowed_db_tickers = set()
            for t in raw_targets:
                t_clean = t.split('.')[0] 
                t_clean = t_clean.split('-')[0]
                allowed_db_tickers.add(t_clean)
            
            st.success(f"관리 대상 종목 {len(allowed_db_tickers)}개를 확인했습니다. 필터링을 시작합니다.")
            xls = pd.read_excel(uploaded_file, sheet_name=None, header=None, dtype=str)
            sheet_map = {'1w': None, '1m': None, '3m': None}
            for sheet_name in xls.keys():
                s_name = sheet_name.lower().strip()
                if '1w' in s_name: sheet_map['1w'] = xls[sheet_name]
                elif '1m' in s_name: sheet_map['1m'] = xls[sheet_name]
                elif '3m' in s_name: sheet_map['3m'] = xls[sheet_name]
            
            if not (sheet_map['1w'] is not None and sheet_map['1m'] is not None and sheet_map['3m'] is not None):
                st.error("엑셀 파일에 1w, 1m, 3m 시트가 모두 있어야 합니다.")
            else:
                data_1w = parse_sheet_ticker_value(sheet_map['1w'], allowed_db_tickers, show_debug_log)
                data_1m = parse_sheet_ticker_value(sheet_map['1m'], allowed_db_tickers, show_debug_log)
                data_3m = parse_sheet_ticker_value(sheet_map['3m'], allowed_db_tickers, show_debug_log)
                all_tickers = set(data_1w.keys()) | set(data_1m.keys()) | set(data_3m.keys())
                
                if not all_tickers: st.warning("매칭되는 데이터 없음")
                else:
                    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                    existing_map = {}
                    try:
                        res = supabase.table("quant_data").select("*").gte("created_at", f"{today_str} 00:00:00").lte("created_at", f"{today_str} 23:59:59").execute()
                        if res.data:
                            for rec in res.data:
                                existing_map[rec['ticker']] = (str(rec.get('change_1w') or "-"), str(rec.get('change_1m') or "-"), str(rec.get('change_3m') or "-"))
                    except: pass
                    
                    rows_to_insert = []
                    skipped_count = 0
                    for t in all_tickers:
                        v_1w = data_1w.get(t, "-"); v_1m = data_1m.get(t, "-"); v_3m = data_3m.get(t, "-")
                        if t in existing_map:
                            e_1w, e_1m, e_3m = existing_map[t]
                            if (e_1w == v_1w) and (e_1m == v_1m) and (e_3m == v_3m):
                                skipped_count += 1
                                continue
                        rows_to_insert.append({"ticker": t, "change_1w": v_1w, "change_1m": v_1m, "change_3m": v_3m})
                    
                    if rows_to_insert:
                        chunk_size = 100
                        for i in range(0, len(rows_to_insert), chunk_size):
                            chunk = rows_to_insert[i:i+chunk_size]
                            supabase.table("quant_data").insert(chunk).execute()
                        st.success(f"✅ DB 업로드 완료! (신규: {len(rows_to_insert)}건, 중복생략: {skipped_count}건)")
                        fetch_latest_quant_data_from_db.clear()
                        GLOBAL_QUANT_DATA = fetch_latest_quant_data_from_db()
                    else: st.info(f"변동 사항이 없습니다. (중복 생략: {skipped_count}건)")
        except Exception as e: st.error(f"작업 실패: {e}")

    st.markdown("---")
    if st.button("데이터 조회하기"):
        try:
            response = supabase.table("quant_data").select("ticker, change_1w, change_1m, change_3m").order("created_at", desc=True).execute()
            if response.data: st.dataframe(pd.DataFrame(response.data), use_container_width=True)
            else: st.warning("데이터가 없습니다.")
        except Exception as e: st.error(f"조회 실패: {e}")

st.markdown("---")
with st.expander("🗄️ 전체 저장 기록 보기 / 관리"):
    col_e1, col_e2 = st.columns([1, 1])
    with col_e1:
        if st.button("🔄 기록 새로고침"):
            try:
                response = supabase.table("history").select("*").order("created_at", desc=True).limit(50).execute()
                if response.data: st.dataframe(pd.DataFrame(response.data), use_container_width=True)
            except Exception as e: st.error(str(e))
    with col_e2:
        if st.button("🧹 중복 데이터 정리 (최신본만 유지)"):
            remove_duplicates_from_db()
