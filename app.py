import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from supabase import create_client, Client
from scipy.signal import argrelextrema
import time
import re

# =========================================================
# [설정] Supabase 연결 정보
# =========================================================
SUPABASE_URL = "https://sgpzmkfproftswevwybm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNncHpta2Zwcm9mdHN3ZXZ3eWJtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ5OTQ0MDEsImV4cCI6MjA4MDU3MDQwMX0.VwStTHOr7_SqYrfwqol1E3ab89HsoUArV1q1s7UFAR4"

# ==========================================
# 1. 페이지 설정 및 DB 연결
# ==========================================
st.set_page_config(page_title="Pro 주식 검색기", layout="wide")
st.title("📈 Pro 주식 검색기: TTM Squeeze (50일) & 퀀티와이즈 통합")

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

def get_stock_sector(ticker):
    try:
        tick = yf.Ticker(ticker)
        info = tick.info
        quote_type = info.get('quoteType', '').upper()
        if 'ETF' in quote_type or 'FUND' in quote_type:
            name = info.get('shortName', '')
            if not name: name = info.get('longName', 'ETF')
            return f"[ETF] {name}"
        sector = info.get('sector', '')
        if not sector: sector = info.get('industry', '')
        if not sector: sector = info.get('shortName', '')
        if not sector: return "Unknown"
        translations = {
            'Technology': '기술', 'Healthcare': '헬스케어', 'Financial Services': '금융',
            'Consumer Cyclical': '임의소비재', 'Industrials': '산업재', 'Basic Materials': '소재',
            'Energy': '에너지', 'Utilities': '유틸리티', 'Real Estate': '부동산',
            'Communication Services': '통신', 'Consumer Defensive': '필수소비재',
            'Semiconductors': '반도체'
        }
        return translations.get(sector, sector)
    except:
        return "Unknown"

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
# [핵심 로직] 정규화 및 DB 캐시
# ==============================================================================
def normalize_ticker_for_db_storage(t):
    """
    QuantWise 엑셀 티커를 DB/Yahoo Finance 공통 포맷으로 변환
    """
    if not t: return ""
    t_str = str(t).upper().strip()
    
    # 1. 한국 주식: 'A'로 시작하고 뒤가 6자리 숫자인 경우 (예: A005930 -> 005930)
    if t_str.startswith('A') and len(t_str) == 7 and t_str[1:].isdigit():
        return t_str[1:]

    # 2. 미국 주식 (-US)
    if t_str.endswith("-US"):
        clean = t_str[:-3]  # -US 제거
        return clean.replace('.', '-')

    # 3. 홍콩 (-HK)
    if t_str.endswith("-HK"):
        return t_str[:-3] + ".HK"

    # 4. 일본 (-JP)
    if t_str.endswith("-JP"):
        return t_str[:-3] + ".T"
        
    # 5. 기존 한국 포맷 (-KS, -KQ) 제거
    if t_str.endswith("-KS"): return t_str[:-3]
    if t_str.endswith("-KQ"): return t_str[:-3]

    # 6. 기타 하이픈 처리
    if '-' in t_str and not any(x in t_str for x in ['.HK', '.T']):
         return t_str.split('-')[0]

    return t_str

def normalize_ticker_for_app_lookup(t):
    if not t: return ""
    t_str = str(t).upper().strip()
    if t_str.endswith(".KS"): return t_str[:-3]
    if t_str.endswith(".KQ"): return t_str[:-3]
    if '.' in t_str and not any(x in t_str for x in ['.HK', '.T', '.KS', '.KQ']):
        return t_str.replace('.', '-')
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
    # 주봉/월봉용 기존 지표 계산 함수
    if len(df) < 100: return None
    df = df.copy()
    period = 20 if is_weekly else 60
    df[f'EMA{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    df[f'STD{period}'] = df['Close'].rolling(window=period).std()
    df['BB_UP'] = df[f'EMA{period}'] + (2 * df[f'STD{period}'])
    df['BB_LO'] = df[f'EMA{period}'] - (2 * df[f'STD{period}'])
    df['BandWidth'] = (df['BB_UP'] - df['BB_LO']) / df[f'EMA{period}']
    df['MACD_V'], df['MACD_V_Signal'] = calculate_macdv(df, 12, 26, 9)
    ema_fast_c = df['Close'].ewm(span=20, adjust=False).mean()
    ema_slow_c = df['Close'].ewm(span=200, adjust=False).mean()
    df['MACD_Line_Custom'] = ema_fast_c - ema_slow_c
    df['MACD_Signal_Custom'] = df['MACD_Line_Custom'].ewm(span=20, adjust=False).mean()
    df['MACD_OSC_Custom'] = df['MACD_Line_Custom'] - df['MACD_Signal_Custom']
    df['Change'] = df['Close'].diff()
    df['Vol_Up'] = np.where(df['Change'] > 0, df['Volume'], 0)
    df['Vol_Down'] = np.where(df['Change'] < 0, df['Volume'], 0)
    df['Vol_Flat'] = np.where(df['Change'] == 0, df['Volume'], 0)
    roll_up = df['Vol_Up'].rolling(window=20).sum()
    roll_down = df['Vol_Down'].rolling(window=20).sum()
    roll_flat = df['Vol_Flat'].rolling(window=20).sum()
    df['VR20'] = ((roll_up + roll_flat/2) / (roll_down + roll_flat/2 + 1e-9)) * 100
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = tr.ewm(span=14, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['VolSMA20'] = df['Volume'].rolling(window=20).mean()
    return df

# -----------------------------------------------------------------------------
# [변경됨] 50일 기준 TTM Squeeze가 적용된 일봉 계산 로직
# -----------------------------------------------------------------------------
def calculate_daily_indicators(df):
    if len(df) < 260: return None
    df = df.copy()

    # 1. 기준선 (Basis) - SMA 50 (중장기 추세)
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # 2. 볼린저 밴드 (50, 2.0)
    df['STD50'] = df['Close'].rolling(window=50).std()
    df['BB50_UP'] = df['SMA50'] + (2.0 * df['STD50'])
    df['BB50_LO'] = df['SMA50'] - (2.0 * df['STD50'])
    df['BW50'] = (df['BB50_UP'] - df['BB50_LO']) / df['SMA50'] # 밴드폭

    # 3. 켈트너 채널 (50, 1.5)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # ATR 50 (SMA 방식)
    df['ATR50'] = df['TR'].rolling(window=50).mean()
    
    # KC 승수 1.5 적용 (진성 스퀴즈)
    kc_mult = 1.5 
    df['KC50_UP'] = df['SMA50'] + (kc_mult * df['ATR50'])
    df['KC50_LO'] = df['SMA50'] - (kc_mult * df['ATR50'])

    # 4. TTM Squeeze 판별 (BB가 KC 안으로 들어옴)
    df['TTM_Squeeze'] = (df['BB50_UP'] < df['KC50_UP']) & (df['BB50_LO'] > df['KC50_LO'])

    # 5. 기존 지표들 (돈키언 등)
    df['Donchian_High_50'] = df['High'].rolling(window=50).max().shift(1)
    df['Change'] = df['Close'].diff()
    df['Vol_Up'] = np.where(df['Change'] > 0, df['Volume'], 0)
    df['Vol_Down'] = np.where(df['Change'] < 0, df['Volume'], 0)
    df['Vol_Flat'] = np.where(df['Change'] == 0, df['Volume'], 0)
    roll_up = df['Vol_Up'].rolling(window=50).sum()
    roll_down = df['Vol_Down'].rolling(window=50).sum()
    roll_flat = df['Vol_Flat'].rolling(window=50).sum()
    df['VR50'] = ((roll_up + roll_flat/2) / (roll_down + roll_flat/2 + 1e-9)) * 100
    
    # MACD Custom
    ema_fast = df['Close'].ewm(span=20, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=200, adjust=False).mean()
    df['MACD_Line_C'] = ema_fast - ema_slow
    df['MACD_Signal_C'] = df['MACD_Line_C'].ewm(span=20, adjust=False).mean()
    df['MACD_OSC_C'] = df['MACD_Line_C'] - df['MACD_Signal_C']
    
    # 표시용 ATR14
    df['ATR14'] = df['TR'].ewm(span=14, adjust=False).mean()
    
    # MACD-V
    df['MACD_V'], _ = calculate_macdv(df, 12, 26, 9)
    
    # EMA200 (눌림목용)
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    return df

# -----------------------------------------------------------------------------
# [변경됨] 50일 TTM Squeeze가 반영된 일봉 조건 체크 함수
# -----------------------------------------------------------------------------
def check_daily_condition(df):
    if len(df) < 260: return False, None
    df = calculate_daily_indicators(df)
    if df is None: return False, None
    
    curr = df.iloc[-1]
    
    # 1. [필수] 가격 돌파 (돈키언 or BB상단 돌파)
    dc_cond = (df['Close'] > df['Donchian_High_50']).iloc[-3:].any()
    bb_cond = (df['Close'] > df['BB50_UP']).iloc[-3:].any()
    mandatory = dc_cond or bb_cond
    
    # 2. [선택] 보조 조건들
    vr_cond = (df['VR50'].iloc[-3:] > 110).any()
    bw_cond = (df['BW50'].iloc[-51] > curr['BW50']) if len(df)>55 else False
    macd_cond = curr['MACD_OSC_C'] > 0
    
    optional_count = sum([vr_cond, bw_cond, macd_cond])
    
    if mandatory and (optional_count >= 2):
        # [변경] TTM Squeeze (50일, 1.5 ATR) 발생 여부 체크
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
    if curr['Close'] > curr['BB_UP']:
        bw_past = df['BandWidth'].iloc[-21]
        bw_change = "감소" if bw_past > curr['BandWidth'] else "증가"
        return True, {'price': curr['Close'], 'atr': curr['ATR14'], 'bw_curr': curr['BandWidth'], 'bw_past': bw_past, 'bw_change': bw_change, 'macdv': curr['MACD_V']}
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

# -----------------------------------------------------------------------------
# [변경됨] 섹터 분석 함수 (모멘텀 점수 수정, TTM Squeeze 표시, 52주신고가 정보 추가)
# -----------------------------------------------------------------------------
def analyze_sector_trend():
    etfs = get_etfs_from_sheet()
    if not etfs: st.warning("ETF 목록 없음"); return []
    st.write(f"📊 총 {len(etfs)}개 ETF 분석 중...")
    
    results = []; pbar = st.progress(0)
    for i, (t, n) in enumerate(etfs):
        pbar.progress((i+1)/len(etfs))
        rt, df = smart_download(t, "1d", "2y")
        if len(df)<30: continue
        
        # 일봉 지표 계산 (TTM Squeeze 확인을 위해)
        df = calculate_daily_indicators(df)
        if df is None: continue
        
        c = df['Close']; h = df['High']
        curr=c.iloc[-1]
        
        # TTM Squeeze 발생 여부 (최근 5일)
        squeeze_on = df['TTM_Squeeze'].iloc[-5:].any() if 'TTM_Squeeze' in df.columns else False
        
        # 보조지표들
        ema20=c.ewm(span=20).mean(); ema50=c.ewm(span=50).mean(); ema60=c.ewm(span=60).mean()
        ema100=c.ewm(span=100).mean(); ema200=c.ewm(span=200).mean()
        
        bb_up = df['BB50_UP']
        dc_h = df['Donchian_High_50']
        macdv = df['MACD_V']
        atr = df['ATR14'].iloc[-1]
        
        bb_bk = "O" if (c>bb_up).iloc[-3:].any() else "-"
        dc_bk = "O" if (c>dc_h).iloc[-3:].any() else "-"
        align = "⭐ 정배열" if (curr>ema20.iloc[-1] and curr>ema60.iloc[-1] and curr>ema100.iloc[-1] and curr>ema200.iloc[-1]) else "-"
        long_tr = "📈 상승" if (ema60.iloc[-1]>ema100.iloc[-1]>ema200.iloc[-1]) else "-"
        
        # [변경됨] 모멘텀 점수 공식: (6개월 수익률 * 0.5) + (12개월 수익률 * 0.5)
        r6 = c.pct_change(126).iloc[-1] if len(c)>126 else 0
        r12 = c.pct_change(252).iloc[-1] if len(c)>252 else 0
        score = (r6 * 0.5 + r12 * 0.5) * 100
        
        # [추가됨] 52주 신고가 관련 정보 계산
        if len(df) >= 252:
            win_52 = df.iloc[-252:]
            high_idx = win_52['Close'].idxmax()
            high_52_date = high_idx.strftime('%Y-%m-%d')
            
            # 현재 신고가일 이전의 신고가 찾기 (전고점)
            prev_win = win_52[win_52.index < high_idx]
            if len(prev_win) > 0:
                prev_idx = prev_win['Close'].idxmax()
                prev_date = prev_idx.strftime('%Y-%m-%d')
                diff_days = (high_idx - prev_idx).days
            else:
                prev_date = "-"
                diff_days = 0
        else:
            high_52_date = "-"
            prev_date = "-"
            diff_days = 0
        
        results.append({
            "ETF": rt, 
            "모멘텀점수": score, 
            "TTM Squeeze(50일)": "🔥" if squeeze_on else "-",
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
    if len(df)<70: return False, None
    df['SMA30']=df['Close'].rolling(30).mean(); curr=df.iloc[-1]; prev=df.iloc[-2]
    if curr['Close']<=curr['SMA30'] or curr['SMA30']<=prev['SMA30']: return False, "추세약함"
    sub = df.iloc[-75:]
    r_win = sub.iloc[-15:-1]; 
    if len(r_win)==0: return False, "데이터부족"
    r_peak = r_win['High'].max(); r_idx = r_win['High'].idxmax()
    l_area = sub[sub.index < r_idx].iloc[:-7]
    if len(l_area)==0: return False, "좌측고점없음"
    l_peak = l_area['High'].max(); l_idx = l_area['High'].idxmax()
    if not (0.9*l_peak <= r_peak <= 1.1*l_peak): return False, "고점불일치"
    cup = sub[(sub.index>l_idx)&(sub.index<r_idx)]
    if len(cup)==0: return False, "컵바닥없음"
    bot = cup['Low'].min(); depth = (l_peak-bot)/l_peak
    if not (0.15<=depth<=0.50): return False, "깊이부적절"
    h_area = df[df.index>r_idx]; h_w = len(h_area)
    if h_w>10: return False, "핸들길어짐"
    if curr['Close']<=r_peak: return False, "미돌파"
    return True, {"depth":f"{depth*100:.1f}%", "handle_weeks":f"{h_w}주", "pivot":f"{r_peak:,.0f}"}

def check_inverse_hs_pattern(df):
    if len(df)<50: return False, None
    p = df['Close'].values; p_idx, t_idx = find_extrema(df, 3)
    if len(t_idx)<3: return False, "저점부족"
    for i in range(len(t_idx)-3, len(t_idx)-1):
        if i<0: continue
        ls=t_idx[i]; h=t_idx[i+1]; rs=t_idx[i+2]
        if (len(p)-rs)>20: continue
        if not (p[h]<p[ls] and p[h]<p[rs]): continue
        if abs(p[ls]-p[rs])/((p[ls]+p[rs])/2)>0.15: continue
        neck1 = np.max(p[ls:h]); neck2 = np.max(p[h:rs])
        neck_idx1 = ls + np.argmax(p[ls:h]); neck_idx2 = h + np.argmax(p[h:rs])
        if neck_idx2==neck_idx1: continue
        slope = (neck2-neck1)/(neck_idx2-neck_idx1); inter = neck1-(slope*neck_idx1)
        proj = slope*(len(p)-1)+inter
        if p[-1]>proj:
            vol_avg=df['Volume'].iloc[-20:].mean(); curr_vol=df['Volume'].iloc[-1]
            return True, {"Neckline":f"{proj:,.0f}", "Breakout":"Yes", "Vol_Ratio":f"{curr_vol/vol_avg:.1f}배"}
    return False, None

def check_pullback_pattern(df):
    if len(df) < 60: return False, None
    df['EMA60'] = df['Close'].ewm(span=60).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['VolSMA20'] = df['Volume'].rolling(20).mean()
    curr = df.iloc[-1]
    if curr['Close'] < curr['EMA60']: return False, "추세 이탈"
    recent_high = df['High'].iloc[-10:].max()
    if curr['Close'] > (recent_high * 0.97): return False, "고점"
    dist = (curr['Close'] - curr['EMA20']) / curr['EMA20']
    if dist < -0.03: return False, "지지선 붕괴"
    if dist > 0.08: return False, "이격도 큼"
    if curr['Volume'] > curr['VolSMA20']: return False, "매도세"
    return True, {"pattern": "20일선 눌림목", "support": "EMA20"}

# ==========================================
# 5. 메인 실행 화면
# ==========================================

st.write("주식 분석 시스템 (5-Factor 전략, MACD-V, TTM Squeeze 50일)")
if not supabase: st.warning("⚠️ DB 연결 키 오류")

tab1, tab2, tab3, tab4 = st.tabs(["📊 신규 종목 발굴", "📉 저장된 종목 눌림목 찾기", "💰 재무분석", "📂 엑셀 데이터 매칭"])

with tab1:
    cols = st.columns(11) 
    
    if cols[0].button("🌍 섹터"):
        st.info("ETF 섹터 분석 중...")
        res = analyze_sector_trend()
        if not res.empty: st.dataframe(res, use_container_width=True)
        else: st.warning("데이터 부족")

    if cols[1].button("🏳️ 국가"):
        tickers = get_country_etfs_from_sheet()
        if tickers:
            st.info(f"[국가 ETF] {len(tickers)}개 일봉 5-Factor 분석 시작...")
            bar = st.progress(0); res = []
            for i, (t, n) in enumerate(tickers):
                bar.progress((i+1)/len(tickers))
                rt, df = smart_download(t, "1d", "2y")
                passed, info = check_daily_condition(df)
                if passed:
                    eps1w, eps1m, eps3m = get_eps_changes_from_db(rt)
                    res.append({
                        '종목코드': rt, '국가/ETF명': n, '현재가': f"{info['price']:,.0f}",
                        'ATR(14)': f"{info['atr']:,.0f}", '스퀴즈': info['squeeze'],
                        '1W변화': eps1w, '1M변화': eps1m, '3M변화': eps3m,
                        '현52주신고가일': info['high_date'], '전52주신고가일': info['prev_date'],
                        '차이일': f"{info['diff_days']}일", 'BW현재': f"{info['bw_curr']:.4f}",
                        'MACD-V': f"{info['macdv']:.2f}", 'BW_Value': f"{info['bw_curr']:.4f}", 'MACD_V_Value': f"{info['macdv']:.2f}"
                    })
            bar.empty()
            if res:
                st.success(f"[국가] {len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res).drop(columns=['BW_Value', 'MACD_V_Value']))
                save_to_supabase(res, "Country_Daily")
            else: st.warning("조건 만족 종목 없음")

    if cols[2].button("🚀 일봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info(f"[일봉 5-Factor + TTM Squeeze] {len(tickers)}개 분석 시작...")
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

    if cols[3].button("📅 주봉"):
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

    if cols[4].button("🗓️ 월봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info(f"[월봉 ATH] {len(tickers)}개 분석 시작...")
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

    if cols[5].button("일+월봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info("일봉(5-Factor) + 월봉(ATH) 교차 분석 중...")
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

    if cols[6].button("일+주봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info("일봉(5-Factor) + 주봉(BB) 교차 분석 중...")
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

    if cols[7].button("주+월봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info("주봉(BB) + 월봉(ATH) 교차 분석 중...")
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

    if cols[8].button("⚡ 통합"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info("[통합] 일+주+월봉 모두 만족하는 종목 검색 중...")
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

    if cols[9].button("🏆 컵핸들"):
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

    if cols[10].button("👤 역H&S"):
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

with tab2:
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
                    # 일봉 계산 로직 활용 (SMA50, MACD-V 등 포함됨)
                    df = calculate_daily_indicators(df)
                    if df is None: continue
                    curr = df.iloc[-1]
                    cond = ""
                    if curr['MACD_V'] > 60: cond = "🔥 공격적 추세"
                    
                    # 눌림목 체크 (20일선 기준) - calculate_common_indicators 로직 일부 차용
                    ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
                    if (curr['Close'] > ema20) and ((curr['Close']-ema20)/ema20 < 0.03):
                        cond = "📉 20일선 눌림목"
                    
                    if (curr['Close'] > curr['EMA200']) and (-100 <= curr['MACD_V'] <= -50):
                         cond = "🧲 MACD-V 과매도"
                    
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

with tab3:
    st.markdown("### 💰 재무 지표 분석 & EPS Trend (yfinance)")
    st.info("yfinance 데이터를 기반으로 핵심 재무 지표 및 EPS 추정치 변화를 분석합니다.")
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

# ==============================================================================
# [NEW] 4. 엑셀 데이터 매칭 탭 (DB 저장 & 초기화 & 화이트리스트 적용)
# ==============================================================================
with tab4:
    st.markdown("### 📂 엑셀 데이터 매칭 (퀀티와이즈 DB 연동)")
    st.info("퀀티와이즈 엑셀(quant_master.xlsx)을 업로드하여 Supabase DB에 저장합니다.\n\n"
            "**[주의사항]**\n"
            "- Supabase DB의 `quant_data` 테이블 컬럼이 **TEXT** 타입이어야 합니다.\n"
            "**[화이트리스트 적용]**\n"
            "- 구글 시트(TGT)에 있는 종목만 필터링하여 저장합니다.\n")
    
    col_upload, col_reset = st.columns([3, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader("📥 quant_master.xlsx 파일을 드래그하여 업로드하세요", type=['xlsx'])
    
    # [DB 초기화 버튼]
    with col_reset:
        st.write("") # 줄맞춤
        st.write("") 
        if st.button("🗑️ [주의] DB 초기화 (전체 삭제)", type="primary"):
            try:
                # 모든 데이터 삭제 (id가 0이 아닌 모든 행)
                supabase.table("quant_data").delete().neq("id", 0).execute()
                st.success("DB가 초기화되었습니다. 이제 파일을 업로드하세요.")
                fetch_latest_quant_data_from_db.clear()
            except Exception as e:
                st.error(f"초기화 실패 (Supabase 권한 확인 필요): {e}")

    # [디버깅 옵션]
    show_debug_log = st.checkbox("🔍 디버깅 로그 보기 (왜 저장이 안 되는지 확인)")

    # --- 서브 함수: 엑셀 시트 파싱 (화이트리스트 필터링 추가, 문자열 처리) ---
    def parse_sheet_ticker_value(sheet_df, allowed_tickers, debug_mode=False):
        extracted = {}
        for index, row in sheet_df.iterrows():
            try:
                raw_ticker = str(row[0]).strip()
                if not raw_ticker or raw_ticker.lower() in ['code', 'ticker', 'nan', 'item type', 'comparison date']:
                    continue
                
                # 1. 정규화 (Quant -> DB Format)
                norm_ticker = normalize_ticker_for_db_storage(raw_ticker)
                
                # [디버깅] 특정 티커가 어떻게 처리되는지 확인
                if debug_mode and "RKLB" in norm_ticker:
                    st.write(f"📢 [DEBUG] 발견된 티커: {raw_ticker} -> 정규화: {norm_ticker} -> 화이트리스트 포함 여부: {norm_ticker in allowed_tickers}")

                # 2. [핵심] 화이트리스트 필터링
                if norm_ticker not in allowed_tickers:
                    continue

                # 3. [핵심] 값 가져오기 (문자열 그대로)
                val = row[3] # D열
                if pd.isna(val):
                    final_val = "-"
                else:
                    final_val = str(val).strip()
                    # 문자열 "nan" 또는 빈 값 처리
                    if final_val.lower() == 'nan' or final_val == "":
                        final_val = "-"
                
                extracted[norm_ticker] = final_val
            except Exception:
                continue
        return extracted

    if uploaded_file and st.button("🔄 DB 업로드 및 분석 시작"):
        try:
            # 0. 구글 시트에서 관리 종목(Target) 가져오기
            st.info("구글 시트에서 관리 종목(TGT) 목록을 불러오는 중...")
            tgt_stocks = get_tickers_from_sheet()
            tgt_etfs = [x[0] for x in get_etfs_from_sheet()]
            tgt_countries = [x[0] for x in get_country_etfs_from_sheet()]
            
            # 관리 종목 합치기 및 정규화
            raw_targets = set(tgt_stocks + tgt_etfs + tgt_countries)
            allowed_db_tickers = set()
            for t in raw_targets:
                # 구글 시트에 있는 티커를 DB 저장 포맷으로 변환
                # 예: 005930.KS -> 005930, AAPL -> AAPL
                t_clean = t.split('.')[0] 
                t_clean = t_clean.split('-')[0]
                allowed_db_tickers.add(t_clean)
            
            st.success(f"관리 대상 종목 {len(allowed_db_tickers)}개를 확인했습니다. 필터링을 시작합니다.")

            if show_debug_log:
                if "RKLB" in allowed_db_tickers:
                    st.success("✅ RKLB가 관리 종목(TGT) 목록에 포함되어 있습니다.")
                else:
                    st.error("❌ RKLB가 관리 종목(TGT) 목록에 없습니다! 구글 시트를 확인하세요.")

            # 1. 엑셀 파일 읽기 (모든 데이터를 문자열로 읽기)
            # [중요] dtype=str 옵션을 줘서 처음부터 문자로 읽어들임
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
                # 2. 파싱 (화이트리스트 전달)
                data_1w = parse_sheet_ticker_value(sheet_map['1w'], allowed_db_tickers, show_debug_log)
                data_1m = parse_sheet_ticker_value(sheet_map['1m'], allowed_db_tickers, show_debug_log)
                data_3m = parse_sheet_ticker_value(sheet_map['3m'], allowed_db_tickers, show_debug_log)
                
                # 3. 통합
                all_tickers = set(data_1w.keys()) | set(data_1m.keys()) | set(data_3m.keys())
                
                if not all_tickers:
                    st.warning("엑셀 파일에서 관리 종목(TGT)과 일치하는 데이터를 찾지 못했습니다.")
                else:
                    # 4. DB 중복 체크 (문자열 비교)
                    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                    existing_map = {}
                    try:
                        res = supabase.table("quant_data")\
                            .select("*")\
                            .gte("created_at", f"{today_str} 00:00:00")\
                            .lte("created_at", f"{today_str} 23:59:59")\
                            .execute()
                        if res.data:
                            for rec in res.data:
                                existing_map[rec['ticker']] = (
                                    str(rec.get('change_1w') or "-"),
                                    str(rec.get('change_1m') or "-"),
                                    str(rec.get('change_3m') or "-")
                                )
                    except:
                        pass
                    
                    rows_to_insert = []
                    skipped_count = 0
                    
                    for t in all_tickers:
                        v_1w = data_1w.get(t, "-")
                        v_1m = data_1m.get(t, "-")
                        v_3m = data_3m.get(t, "-")
                        
                        # 중복 체크 (문자열 그대로 비교)
                        if t in existing_map:
                            e_1w, e_1m, e_3m = existing_map[t]
                            if (e_1w == v_1w) and (e_1m == v_1m) and (e_3m == v_3m):
                                skipped_count += 1
                                continue
                        
                        rows_to_insert.append({
                            "ticker": t,
                            "change_1w": v_1w,
                            "change_1m": v_1m,
                            "change_3m": v_3m
                        })
                    
                    if rows_to_insert:
                        # 100개씩 나눠서 저장
                        chunk_size = 100
                        for i in range(0, len(rows_to_insert), chunk_size):
                            chunk = rows_to_insert[i:i+chunk_size]
                            supabase.table("quant_data").insert(chunk).execute()
                        
                        st.success(f"✅ DB 업로드 완료! (TGT 필터링 적용됨. 신규: {len(rows_to_insert)}건, 중복생략: {skipped_count}건)")
                        
                        # 캐시 초기화
                        fetch_latest_quant_data_from_db.clear()
                        GLOBAL_QUANT_DATA = fetch_latest_quant_data_from_db()
                    else:
                        st.info(f"변동 사항이 없습니다. (중복 생략: {skipped_count}건)")
                
        except Exception as e:
            st.error(f"작업 실패: {e}")

    st.markdown("---")
    st.markdown("#### 👁️ 현재 DB 저장 데이터 (전체 조회)")
    if st.button("데이터 조회하기"):
        try:
            # id, created_at 제외하고 필요한 컬럼만 선택
            # limit 제거하여 전체 조회
            response = supabase.table("quant_data")\
                .select("ticker, change_1w, change_1m, change_3m")\
                .order("created_at", desc=True)\
                .execute()
            
            if response.data:
                df_view = pd.DataFrame(response.data)
                # 컬럼 이름이 그대로 나오지만, 순서 보장을 위해 명시적 선택 가능 (이미 select에서 지정했으므로 생략 가능)
                st.dataframe(df_view, use_container_width=True)
            else:
                st.warning("데이터가 없습니다.")
        except Exception as e:
            st.error(f"조회 실패: {e}")

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
