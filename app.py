import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client
from scipy.signal import argrelextrema

# =========================================================
# [설정] Supabase 연결 정보
# =========================================================
SUPABASE_URL = "https://sgpzmkfproftswevwybm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNncHpta2Zwcm9mdHN3ZXZ3eWJtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ5OTQ0MDEsImV4cCI6MjA4MDU3MDQwMX0.VwStTHOr7_SqYrfwqol1E3ab89HsoUArV1q1s7UFAR4"

# ==========================================
# 1. 페이지 설정 및 DB 연결
# ==========================================
st.set_page_config(page_title="Pro 주식 검색기", layout="wide")
st.title("📈 Pro 주식 검색기: 섹터 종합 분석 & 5-Factor")

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
    """ETF 목록 읽어오기"""
    try:
        df = pd.read_csv(ETF_CSV_URL, header=None)
        etf_list = []
        for index, row in df.iterrows():
            ticker = str(row[0]).strip()
            # 헤더나 빈 값 제외
            if not ticker or ticker.lower() in ['ticker', 'symbol', '종목코드', '티커']:
                continue
            name = str(row[1]).strip() if len(row) > 1 else ticker
            etf_list.append((ticker, name))
        return etf_list
    except Exception as e:
        st.error(f"ETF 시트 읽기 실패: {e}")
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
            st.success(f"🧹 중복된 {len(ids_to_remove)}개 데이터를 삭제했습니다.")
        else:
            st.info("삭제할 중복 데이터가 없습니다.")
    except Exception as e:
        st.error(f"중복 제거 실패: {e}")

def smart_download(ticker, interval="1d", period="2y"):
    ticker = ticker.replace('/', '-')
    candidates = [ticker]
    if ticker.isdigit() and len(ticker) == 6:
        candidates = [f"{ticker}.KS", f"{ticker}.KQ", ticker]
    
    for t in candidates:
        try:
            df = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=False)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return t, df
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
    
    # 분모 0 방지
    macd_v = (macd_line / (atr + 1e-9)) * 100
    macd_v_signal = macd_v.ewm(span=signal, adjust=False).mean()
    return macd_v, macd_v_signal

def calculate_common_indicators(df, is_weekly=False):
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

def calculate_daily_indicators(df):
    if len(df) < 260: return None
    df = df.copy()
    
    # 1. BB (50일 EMA, 2시그마)
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['STD50'] = df['Close'].rolling(window=50).std()
    df['BB50_UP'] = df['EMA50'] + (2 * df['STD50'])
    
    # 2. Donchian Channel (50일)
    df['Donchian_High_50'] = df['High'].rolling(window=50).max().shift(1)
    
    # 3. VR (50일)
    df['Change'] = df['Close'].diff()
    df['Vol_Up'] = np.where(df['Change'] > 0, df['Volume'], 0)
    df['Vol_Down'] = np.where(df['Change'] < 0, df['Volume'], 0)
    df['Vol_Flat'] = np.where(df['Change'] == 0, df['Volume'], 0)
    roll_up = df['Vol_Up'].rolling(window=50).sum()
    roll_down = df['Vol_Down'].rolling(window=50).sum()
    roll_flat = df['Vol_Flat'].rolling(window=50).sum()
    df['VR50'] = ((roll_up + roll_flat/2) / (roll_down + roll_flat/2 + 1e-9)) * 100
    
    # 4. BW (60일 EMA, 2시그마)
    df['EMA60'] = df['Close'].ewm(span=60, adjust=False).mean()
    df['STD60'] = df['Close'].rolling(window=60).std()
    df['BB60_UP'] = df['EMA60'] + (2 * df['STD60'])
    df['BB60_LO'] = df['EMA60'] - (2 * df['STD60'])
    df['BW60'] = (df['BB60_UP'] - df['BB60_LO']) / df['EMA60']
    
    # 5. MACD Custom (20, 200, 20)
    ema_fast = df['Close'].ewm(span=20, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=200, adjust=False).mean()
    df['MACD_Line_C'] = ema_fast - ema_slow
    df['MACD_Signal_C'] = df['MACD_Line_C'].ewm(span=20, adjust=False).mean()
    df['MACD_OSC_C'] = df['MACD_Line_C'] - df['MACD_Signal_C']
    
    # 6. ATR (14일)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = tr.ewm(span=14, adjust=False).mean()

    # MACD-V (DB용)
    df['MACD_V'], _ = calculate_macdv(df, 12, 26, 9)

    return df

# [수정] 데이터 제한 완화 및 ETF명 삭제
def analyze_sector_trend():
    etfs = get_etfs_from_sheet()
    if not etfs:
        st.warning("ETF 목록을 불러오지 못했습니다.")
        return []

    st.write(f"📊 총 {len(etfs)}개 ETF에 대해 분석을 시도합니다.")

    # SPY는 비교 기준이므로 데이터가 충분해야 함 (260일)
    spy_ticker, spy_df = smart_download("SPY", interval="1d", period="2y")
    if len(spy_df) < 260:
        st.error("SPY 데이터 부족으로 분석 불가")
        return []

    spy_close = spy_df['Close']
    spy_r1m = spy_close.pct_change(21).iloc[-1]
    spy_r3m = spy_close.pct_change(63).iloc[-1]
    spy_r6m = spy_close.pct_change(126).iloc[-1]
    spy_r12m = spy_close.pct_change(252).iloc[-1]

    results = []
    skipped_count = 0
    progress_bar = st.progress(0)
    
    for i, (ticker, name) in enumerate(etfs):
        progress_bar.progress((i + 1) / len(etfs))
        real_ticker, df = smart_download(ticker, interval="1d", period="2y")
        
        # [수정] 260일 -> 30일로 완화 (최소한의 MA, ATR 계산용)
        if len(df) < 30: 
            skipped_count += 1
            continue
        
        close = df['Close']
        high = df['High']
        
        # EMAs
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean() 
        ema60 = close.ewm(span=60, adjust=False).mean()
        ema100 = close.ewm(span=100, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        
        curr_price = close.iloc[-1]
        
        # BB(50, 2)
        std50 = close.rolling(window=50).std()
        bb50_up = ema50 + (2 * std50)
        
        # Donchian(50)
        donchian_50 = high.rolling(window=50).max().shift(1)
        
        # ATR(14)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr14 = tr.ewm(span=14, adjust=False).mean().iloc[-1]
        
        # MACD-V
        macdv, _ = calculate_macdv(df, 12, 26, 9)
        curr_macdv = macdv.iloc[-1]

        # Conditions
        bb_check = (close > bb50_up).iloc[-3:]
        bb_breakout = "O" if bb_check.any() else "-"
        
        dc_check = (close > donchian_50).iloc[-3:]
        dc_breakout = "O" if dc_check.any() else "-"
        
        e20 = ema20.iloc[-1]; e60 = ema60.iloc[-1]; e100 = ema100.iloc[-1]; e200 = ema200.iloc[-1]
        is_aligned = (curr_price > e20) and (curr_price > e60) and (curr_price > e100) and (curr_price > e200)
        trend_align = "⭐ 정배열" if is_aligned else "-"
        
        is_long_trend = (e60 > e100) and (e100 > e200)
        long_trend_str = "📈 상승" if is_long_trend else "-"
        
        # [수정] 기간별 수익률 계산 (데이터 없으면 0 처리)
        r1m = close.pct_change(21).iloc[-1] if len(close) > 21 else 0
        r3m = close.pct_change(63).iloc[-1] if len(close) > 63 else 0
        r6m = close.pct_change(126).iloc[-1] if len(close) > 126 else 0
        r12m = close.pct_change(252).iloc[-1] if len(close) > 252 else 0
        
        rs_score = (
            0.25 * (r1m - spy_r1m) +
            0.25 * (r3m - spy_r3m) +
            0.25 * (r6m - spy_r6m) +
            0.25 * (r12m - spy_r12m)
        ) * 100

        results.append({
            "ETF": real_ticker,
            # "ETF명": name,  <-- [수정] 삭제됨
            "모멘텀점수": rs_score,
            "BB(50,2)돌파": bb_breakout,
            "돈키언(50)돌파": dc_breakout,
            "정배열": trend_align,
            "장기추세(60>100>200)": long_trend_str,
            "MACD-V": f"{curr_macdv:.2f}",
            "ATR": f"{atr14:.2f}",
            "현재가": curr_price
        })
    
    progress_bar.empty()
    
    if skipped_count > 0:
        st.warning(f"⚠️ 데이터 극소(30일 미만)로 {skipped_count}개 ETF 제외됨")

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(by="모멘텀점수", ascending=False)
        df_res['모멘텀점수'] = df_res['모멘텀점수'].apply(lambda x: f"{x:.2f}")
        df_res['현재가'] = df_res['현재가'].apply(lambda x: f"{x:,.2f}")
    
    return df_res

def check_cup_handle_pattern(df):
    if len(df) < 70: return False, None
    df = df.copy()
    
    df['SMA30'] = df['Close'].rolling(window=30).mean()
    df['VolSMA20'] = df['Volume'].rolling(window=20).mean()
    curr = df.iloc[-1]
    prev = df.iloc[-2]

    if curr['Close'] <= curr['SMA30']: return False, "추세 약함"
    if curr['SMA30'] <= prev['SMA30']: return False, "추세 약함"

    lookback = 75 
    subset = df.iloc[-lookback:]
    right_peak_window = subset.iloc[-15:-1] 
    if len(right_peak_window) == 0: return False, "데이터 부족"
    right_peak_price = right_peak_window['High'].max()
    
    right_peak_idx = right_peak_window['High'].idxmax()
    left_search_area = subset[subset.index < right_peak_idx].iloc[:-7]
    if len(left_search_area) == 0: return False, "왼쪽 고점 없음"
    left_peak_price = left_search_area['High'].max()
    
    if not (0.90 * left_peak_price <= right_peak_price <= 1.10 * left_peak_price): return False, "고점 불일치"
    
    cup_body = subset[(subset.index > left_peak_idx) & (subset.index < right_peak_idx)]
    if len(cup_body) == 0: return False, "컵 바닥 없음"
    bottom_price = cup_body['Low'].min()
    depth_pct = (left_peak_price - bottom_price) / left_peak_price
    if not (0.15 <= depth_pct <= 0.50): return False, "깊이 부적절"
    
    bottom_range = bottom_price * 1.10
    if cup_body[cup_body['Low'] <= bottom_range].shape[0] < 2: return False, "V자 반등"

    handle_area = df[(df.index > right_peak_idx) & (df.index < df.index[-1])]
    handle_weeks = len(handle_area)
    if handle_weeks > 10: return False, "핸들 너무 김"
    if handle_weeks > 0:
        handle_low = handle_area['Low'].min()
        cup_midpoint = bottom_price + (left_peak_price - bottom_price) * 0.5
        if handle_low < cup_midpoint: return False, "핸들 너무 깊음"

    breakout_level = right_peak_price
    if curr['Close'] <= breakout_level: return False, "미돌파"
    if curr['Volume'] < (curr['VolSMA20'] * 1.4): return False, "거래량 부족"

    return True, {"depth": f"{depth_pct*100:.1f}%", "handle_weeks": f"{handle_weeks}주", "pivot": f"{breakout_level:,.0f}"}

def check_inverse_hs_pattern(df):
    if len(df) < 50: return False, None
    df = df.copy()
    prices = df['Close'].values
    peaks_idx, troughs_idx = find_extrema(df, order=3)
    if len(troughs_idx) < 3: return False, "저점 부족"
    
    found_pattern = False
    details = {}
    
    for i in range(len(troughs_idx) - 3, len(troughs_idx) - 1):
        if i < 0: continue
        ls_idx = troughs_idx[i]
        head_idx = troughs_idx[i+1]
        rs_idx = troughs_idx[i+2]
        
        if (len(prices) - rs_idx) > 20: continue
        ls_price = prices[ls_idx]
        head_price = prices[head_idx]
        rs_price = prices[rs_idx]
        
        if not (head_price < ls_price and head_price < rs_price): continue
        avg_shoulder_price = (ls_price + rs_price) / 2
        if abs(ls_price - rs_price) / avg_shoulder_price > 0.15: continue
            
        neck1_range = prices[ls_idx:head_idx]
        if len(neck1_range) == 0: continue
        neck1_price = np.max(neck1_range)
        neck1_offset = np.argmax(neck1_range)
        neck1_idx = ls_idx + neck1_offset
        
        neck2_range = prices[head_idx:rs_idx]
        if len(neck2_range) == 0: continue
        neck2_price = np.max(neck2_range)
        neck2_offset = np.argmax(neck2_range)
        neck2_idx = head_idx + neck2_offset
        
        if neck2_idx == neck1_idx: continue
        slope = (neck2_price - neck1_price) / (neck2_idx - neck1_idx)
        intercept = neck1_price - (slope * neck1_idx)
        
        current_idx = len(prices) - 1
        projected_neck_price = (slope * current_idx) + intercept
        current_close = prices[current_idx]
        
        if current_close > projected_neck_price:
            vol_avg = df['Volume'].iloc[-20:].mean()
            current_vol = df['Volume'].iloc[-1]
            found_pattern = True
            details = {
                "Neckline": f"{projected_neck_price:,.0f}",
                "Breakout": "Yes",
                "Vol_Ratio": f"{current_vol/vol_avg:.1f}배"
            }
            break
            
    return found_pattern, details

def check_pullback_pattern(df):
    if len(df) < 60: return False, None
    df = df.copy()
    curr = df.iloc[-1]
    
    if curr['Close'] < curr['EMA60']: return False, "추세 이탈"
    
    recent_10days = df.iloc[-10:]
    recent_high = recent_10days['High'].max()
    if curr['Close'] > (recent_high * 0.97): return False, "고점 (조정 안받음)"
        
    dist_from_ema20 = (curr['Close'] - curr['EMA20']) / curr['EMA20']
    if dist_from_ema20 < -0.03: return False, "지지선 붕괴"
    if dist_from_ema20 > 0.08: return False, "이격도 큼"
    
    if curr['Volume'] > curr['VolSMA20']: return False, "매도세 강함"
        
    return True, {"pattern": "20일선 눌림목", "support": f"EMA20"}

# ==========================================
# 5. 메인 실행 화면
# ==========================================

st.write("주식 분석 시스템 (5-Factor 전략, MACD-V, 패턴 분석)")
if not supabase: st.warning("⚠️ DB 연결 키 오류")

tab1, tab2 = st.tabs(["📊 신규 종목 발굴", "📉 저장된 종목 눌림목 찾기"])

with tab1:
    cols = st.columns(5)
    
    # [1] ETF 섹터 추세 확인
    if cols[0].button("🌍 추세 섹터 확인"):
        st.info("ETF 섹터 추세 및 RS 모멘텀을 분석합니다... (모든 ETF 조회)")
        df_sector = analyze_sector_trend()
        if not df_sector.empty:
            st.success(f"✅ 총 {len(df_sector)}개 ETF 섹터 분석 결과 (모멘텀 순)")
            st.dataframe(df_sector, use_container_width=True)
        else:
            st.warning("분석할 데이터가 부족합니다.")

    # [2] 일봉 분석 (업데이트된 5-Factor 로직)
    if cols[1].button("🚀 일봉 분석"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info(f"[일봉 5-Factor] {len(tickers)}개 분석 시작...")
            progress_bar = st.progress(0)
            results = []
            for i, raw_ticker in enumerate(tickers):
                progress_bar.progress((i + 1) / len(tickers))
                if not raw_ticker: continue
                real_ticker, df = smart_download(raw_ticker, interval="1d")
                if len(df) == 0: continue
                try:
                    df = calculate_daily_indicators(df)
                    if df is None: continue
                    curr = df.iloc[-1]
                    
                    # 5-Factor 스크리닝
                    # 1. BB 돌파 (최근 3일 내)
                    bb_cond = df['Close'] > df['BB50_UP']
                    cond1 = bb_cond.iloc[-3:].any()
                    
                    # 2. 돈키언 채널 돌파 (최근 3일 내)
                    dc_cond = df['Close'] > df['Donchian_High_50']
                    cond2 = dc_cond.iloc[-3:].any()
                    
                    # 3. VR 급등 (최근 3일 내)
                    vr_check = df['VR50'].iloc[-3:]
                    cond3 = (vr_check > 110).any()
                    
                    # 4. BW 수축
                    if len(df) > 55:
                        bw_past_50 = df['BW60'].iloc[-51]
                        cond4 = bw_past_50 > curr['BW60']
                    else: cond4 = False
                    
                    # 5. MACD OSC > 0
                    cond5 = curr['MACD_OSC_C'] > 0
                    
                    if cond1 and cond2 and cond3 and cond4 and cond5:
                        sector = get_stock_sector(real_ticker)
                        window_52w = df.iloc[-252:]
                        curr_high_date = window_52w['Close'].idxmax().strftime('%Y-%m-%d')
                        prev_win = window_52w[window_52w.index < window_52w['Close'].idxmax()]
                        prev_high_date = prev_win['Close'].idxmax().strftime('%Y-%m-%d') if len(prev_win)>0 else "-"
                        diff_days = (window_52w['Close'].idxmax() - prev_win['Close'].idxmax()).days if len(prev_win)>0 else 0
                        
                        bw_str = f"{curr['BW60']:.4f}"
                        if curr['BW60'] < 0.25: bw_str += " (low_vol)"
                        
                        results.append({
                            '종목코드': real_ticker, 
                            '섹터': sector, 
                            '현재가': f"{curr['Close']:,.0f}",
                            'ATR(14)': f"{curr['ATR14']:,.0f}",
                            '현52주신고가일': curr_high_date, 
                            '전52주신고가일': prev_high_date,
                            '차이일': f"{diff_days}일",
                            'BW현재': bw_str,
                            'BW_Value': f"{curr['BW60']:.4f}",
                            'MACD-V': f"{curr['MACD_V']:.2f}", 
                            'MACD_V_Value': f"{curr['MACD_V']:.2f}"
                        })
                except Exception as e: continue
            
            progress_bar.empty()
            if results:
                st.success(f"[일봉 5-Factor] {len(results)}개 발견!")
                st.dataframe(pd.DataFrame(results).drop(columns=['BW_Value', 'MACD_V_Value']))
                save_to_supabase(results, "Daily_5Factor")
            else: st.warning("5가지 복합 조건을 모두 만족하는 종목이 없습니다.")

    # [B] 주봉 분석
    if cols[2].button("📅 주봉 분석"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info(f"[주봉] {len(tickers)}개 분석 시작...")
            progress_bar = st.progress(0)
            results = []
            for i, raw_ticker in enumerate(tickers):
                progress_bar.progress((i + 1) / len(tickers))
                if not raw_ticker: continue
                real_ticker, df = smart_download(raw_ticker, interval="1wk")
                if len(df) == 0: continue
                try:
                    df = calculate_common_indicators(df, is_weekly=True)
                    if df is None: continue
                    curr = df.iloc[-1]
                    if curr['Close'] > curr['BB_UP']:
                        sector = get_stock_sector(real_ticker)
                        window_52w = df.iloc[-52:]
                        curr_high_date = window_52w['Close'].idxmax().strftime('%Y-%m-%d')
                        prev_win = window_52w[window_52w.index < window_52w['Close'].idxmax()]
                        prev_high_date = prev_win['Close'].idxmax().strftime('%Y-%m-%d') if len(prev_win)>0 else "-"
                        diff_days = (window_52w['Close'].idxmax() - prev_win['Close'].idxmax()).days if len(prev_win)>0 else 0
                        
                        bw_curr = curr['BandWidth']
                        bw_past = df['BandWidth'].iloc[-21] 
                        bw_diff = bw_past - bw_curr
                        bw_status = "감소" if bw_diff > 0 else "증가"
                        bw_str = f"{bw_curr:.4f}"
                        if bw_curr < 0.25: bw_str += " (low_vol)"
                        
                        results.append({
                            '종목코드': real_ticker, '섹터': sector, '현재가': f"{curr['Close']:,.0f}",
                            'ATR(14)': f"{curr['ATR14']:,.0f}",
                            '현52주신고가일': curr_high_date, 
                            '전52주신고가일': prev_high_date,
                            '차이일': f"{diff_days}일",
                            'BW현재': bw_str,
                            'BW(20주전)': f"{bw_past:.4f}",
                            'BW변화': bw_status,
                            'BW_Value': f"{bw_curr:.4f}",
                            'MACD-V': f"{curr['MACD_V']:.2f}", 
                            'MACD_V_Value': f"{curr['MACD_V']:.2f}"
                        })
                except: continue
            progress_bar.empty()
            if results:
                st.success(f"[주봉] {len(results)}개 발견!")
                st.dataframe(pd.DataFrame(results).drop(columns=['BW_Value', 'MACD_V_Value']))
                save_to_supabase(results, "Weekly")
            else: st.warning("조건 만족 없음")

    # [C] 컵위드핸들
    if cols[3].button("🏆 컵핸들"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info(f"[컵핸들] {len(tickers)}개 분석 시작...")
            progress_bar = st.progress(0)
            results = []
            for i, raw_ticker in enumerate(tickers):
                progress_bar.progress((i + 1) / len(tickers))
                if not raw_ticker: continue
                real_ticker, df = smart_download(raw_ticker, interval="1wk")
                if len(df) == 0: continue
                try:
                    is_cup, details = check_cup_handle_pattern(df)
                    if is_cup:
                        df_indic = calculate_common_indicators(df, is_weekly=True)
                        curr = df_indic.iloc[-1]
                        sector = get_stock_sector(real_ticker)
                        window_52w = df.iloc[-52:]
                        curr_high_date = window_52w['Close'].idxmax().strftime('%Y-%m-%d')
                        results.append({
                            '종목코드': real_ticker, '섹터': sector, '현재가': f"{curr['Close']:,.0f}",
                            '패턴상세': f"깊이:{details['depth']}", '돌파가격': details['pivot'],
                            '현52주신고가일': curr_high_date,
                            'BW_Value': f"{curr['BandWidth']:.4f}",
                            'MACD_V_Value': f"{curr['MACD_V']:.2f}"
                        })
                except: continue
            progress_bar.empty()
            if results:
                st.success(f"🏆 컵위드핸들 {len(results)}개 발견!")
                st.dataframe(pd.DataFrame(results).drop(columns=['BW_Value', 'MACD_V_Value']))
                save_to_supabase(results, "CupHandle")
            else: st.warning("조건 만족 없음")

    # [D] 역헤드앤숄더
    if cols[4].button("👤 역H&S"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info(f"[역H&S] {len(tickers)}개 분석 시작...")
            progress_bar = st.progress(0)
            results = []
            for i, raw_ticker in enumerate(tickers):
                progress_bar.progress((i + 1) / len(tickers))
                if not raw_ticker: continue
                real_ticker, df = smart_download(raw_ticker, interval="1wk")
                if len(df) == 0: continue
                try:
                    is_invhs, details = check_inverse_hs_pattern(df)
                    if is_invhs:
                        df_indic = calculate_common_indicators(df, is_weekly=True)
                        curr = df_indic.iloc[-1]
                        sector = get_stock_sector(real_ticker)
                        window_52w = df.iloc[-52:]
                        curr_high_date = window_52w['Close'].idxmax().strftime('%Y-%m-%d')
                        results.append({
                            '종목코드': real_ticker, '섹터': sector, '현재가': f"{curr['Close']:,.0f}",
                            '넥라인': details['Neckline'], '거래량급증': details['Vol_Ratio'],
                            '현52주신고가일': curr_high_date,
                            'BW_Value': f"{curr['BandWidth']:.4f}",
                            'MACD_V_Value': f"{curr['MACD_V']:.2f}"
                        })
                except: continue
            progress_bar.empty()
            if results:
                st.success(f"👤 역헤드앤숄더 {len(results)}개 발견!")
                st.dataframe(pd.DataFrame(results).drop(columns=['BW_Value', 'MACD_V_Value']))
                save_to_supabase(results, "InverseHS")
            else: st.warning("조건 만족 없음")

with tab2:
    st.markdown("### 📉 저장된 종목 중 눌림목/급등주 찾기")
    if st.button("🔍 눌림목 & 급등 패턴 분석"):
        db_tickers = get_unique_tickers_from_db()
        if not db_tickers: st.warning("DB 데이터 없음")
        else:
            st.info(f"{len(db_tickers)}개 종목 재분석 중...")
            progress_bar = st.progress(0)
            pullback_results = []
            for i, raw_ticker in enumerate(db_tickers):
                progress_bar.progress((i + 1) / len(db_tickers))
                real_ticker, df = smart_download(raw_ticker, interval="1d")
                if len(df) == 0: continue
                try:
                    # 눌림목 분석용 지표 계산 (common indicator 사용)
                    df = calculate_common_indicators(df, is_weekly=False)
                    if df is None: continue
                    curr = df.iloc[-1]
                    
                    cond = ""
                    if curr['MACD_V'] > 60: cond = "🔥 공격적 추세"
                    
                    is_pullback, details = check_pullback_pattern(df)
                    if is_pullback: cond = f"📉 {details['pattern']}"

                    if (curr['Close'] > curr['EMA200']) and (-100 <= curr['MACD_V'] <= -50):
                         cond = "🧲 MACD-V 과매도"
                    
                    if cond:
                        pullback_results.append({
                            '종목코드': real_ticker, '패턴': cond,
                            '현재가': f"{curr['Close']:,.0f}",
                            'MACD-V': f"{curr['MACD_V']:.2f}",
                            'EMA20': f"{curr['EMA20']:,.0f}"
                        })
                except: continue
            progress_bar.empty()
            if pullback_results:
                st.success(f"{len(pullback_results)}개 발견!")
                st.dataframe(pd.DataFrame(pullback_results), use_container_width=True)
            else: st.warning("조건 만족 종목 없음")

st.markdown("---")
with st.expander("🗄️ 전체 저장 기록 보기 / 관리"):
    col_e1, col_e2 = st.columns([1, 1])
    
    with col_e1:
        if st.button("🔄 기록 새로고침"):
            try:
                response = supabase.table("history").select("*").order("created_at", desc=True).limit(50).execute()
                if response.data:
                    st.dataframe(pd.DataFrame(response.data), use_container_width=True)
            except Exception as e: st.error(str(e))
            
    with col_e2:
        if st.button("🧹 중복 데이터 정리 (최신본만 유지)"):
            remove_duplicates_from_db()
