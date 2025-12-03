import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ==========================================
# 1. 페이지 설정
# ==========================================
st.set_page_config(page_title="주식 조건 검색기", layout="wide")
st.title("📈 주식 기술적 지표 필터링 시스템")

# ==========================================
# 2. 구글 시트 연결 설정
# ==========================================
SHEET_ID = '1NVThO1z2HHF0TVXVRGmbVsSU_Svyjg8fxd7E90z2o8A'
GID = '0' 
CSV_URL = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}'

# ==========================================
# 3. 함수 정의
# ==========================================

def get_tickers_from_sheet():
    try:
        df = pd.read_csv(CSV_URL, header=None)
        tickers = sorted(list(set([str(x).strip() for x in df[0] if str(x).strip()])))
        return tickers
    except Exception as e:
        st.error(f"구글 시트 읽기 실패: {e}")
        return []

def smart_download(ticker, interval="1d"):
    """가격 데이터 다운로드"""
    ticker = ticker.replace('/', '-')
    candidates = [ticker]
    if ticker.isdigit() and len(ticker) == 6:
        candidates = [f"{ticker}.KS", f"{ticker}.KQ", ticker]
    
    for t in candidates:
        try:
            df = yf.download(t, period="2y", interval=interval, progress=False, auto_adjust=False)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return t, df
        except:
            continue
    return ticker, pd.DataFrame()

def get_stock_sector(ticker):
    """(신규) 종목의 섹터 정보를 가져옵니다."""
    try:
        # Ticker 객체 생성 후 info에서 sector 가져오기
        info = yf.Ticker(ticker).info
        sector = info.get('sector', 'N/A') # 섹터 (예: Technology)
        # industry = info.get('industry', 'N/A') # 산업군 (필요하면 주석 해제)
        
        # 영문 섹터명을 한글로 간단히 매핑 (선택사항, 필요 없으면 주석 처리)
        translations = {
            'Technology': '기술', 'Healthcare': '헬스케어', 'Financial Services': '금융',
            'Consumer Cyclical': '임의소비재', 'Industrials': '산업재', 'Basic Materials': '소재',
            'Energy': '에너지', 'Utilities': '유틸리티', 'Real Estate': '부동산',
            'Communication Services': '통신', 'Consumer Defensive': '필수소비재'
        }
        return translations.get(sector, sector)
    except:
        return "Unknown"

# --- [1] 일봉 계산 함수 ---
def calculate_daily_indicators(df):
    if len(df) < 260: return None
    df = df.copy()
    
    df['EMA60'] = df['Close'].ewm(span=60, adjust=False).mean()
    df['STD60'] = df['Close'].rolling(window=60).std()
    df['BB_UP'] = df['EMA60'] + (2 * df['STD60'])
    df['BB_LO'] = df['EMA60'] - (2 * df['STD60'])
    df['BandWidth'] = (df['BB_UP'] - df['BB_LO']) / df['EMA60']

    ema_fast = df['Close'].ewm(span=20, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=200, adjust=False).mean()
    df['MACD_Line_Custom'] = ema_fast - ema_slow
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

    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line_Day'] = exp12 - exp26
    df['MACD_Signal_Day'] = df['MACD_Line_Day'].ewm(span=9, adjust=False).mean()
    df['Day_Buy_Signal'] = (df['MACD_Line_Day'] > df['MACD_Signal_Day']) & \
                           (df['MACD_Line_Day'].shift(1) <= df['MACD_Signal_Day'].shift(1))
    return df

# --- [2] 주봉 계산 함수 ---
def calculate_weekly_indicators(df):
    if len(df) < 60: return None
    df = df.copy()

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_UP'] = df['EMA20'] + (2 * df['STD20'])
    df['BB_LO'] = df['EMA20'] - (2 * df['STD20'])
    df['BandWidth'] = (df['BB_UP'] - df['BB_LO']) / df['EMA20']

    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_OSC'] = df['MACD_Line'] - df['MACD_Signal']

    df['Change'] = df['Close'].diff()
    df['Vol_Up'] = np.where(df['Change'] > 0, df['Volume'], 0)
    df['Vol_Down'] = np.where(df['Change'] < 0, df['Volume'], 0)
    df['Vol_Flat'] = np.where(df['Change'] == 0, df['Volume'], 0)
    roll_up = df['Vol_Up'].rolling(window=20).sum()
    roll_down = df['Vol_Down'].rolling(window=20).sum()
    roll_flat = df['Vol_Flat'].rolling(window=20).sum()
    df['VR20'] = ((roll_up + roll_flat/2) / (roll_down + roll_flat/2 + 1e-9)) * 100

    return df

def check_weekly_macd_signal_helper(ticker):
    try:
        w_df = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=False)
        if len(w_df) < 50: return False
        if isinstance(w_df.columns, pd.MultiIndex):
            w_df.columns = w_df.columns.get_level_values(0)
        
        exp12 = w_df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = w_df['Close'].ewm(span=26, adjust=False).mean()
        line = exp12 - exp26
        signal = line.ewm(span=9, adjust=False).mean()
        
        for i in range(1, 4):
            if i >= len(w_df): break
            if (line.iloc[-i] > signal.iloc[-i]) and (line.iloc[-(i+1)] <= signal.iloc[-(i+1)]):
                return True
        return False
    except:
        return False

# ==========================================
# 4. 메인 실행 화면
# ==========================================

st.write("구글 시트(1NVTh...)의 종목을 분석합니다. 원하는 분석 버튼을 누르세요.")

col1, col2 = st.columns(2)

# ==========================================
# [A] 일봉 분석 로직
# ==========================================
if col1.button("🚀 일봉 분석 시작 (Daily)"):
    tickers = get_tickers_from_sheet()
    
    if not tickers:
        st.error("시트에서 종목을 읽어오지 못했습니다. 공유 설정을 확인해주세요.")
    else:
        st.info(f"[일봉] {len(tickers)}개 종목 분석 시작...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        for i, raw_ticker in enumerate(tickers):
            progress_bar.progress((i + 1) / len(tickers))
            status_text.text(f"일봉 분석 중: {raw_ticker}")
            
            if not raw_ticker: continue
            
            real_ticker, df = smart_download(raw_ticker, interval="1d")
            if len(df) == 0: continue
                
            try:
                df = calculate_daily_indicators(df)
                if df is None: continue
                
                curr = df.iloc[-1]
                
                # === 조건: Price > BB Upper (60, 2) ===
                if curr['Close'] > curr['BB_UP']:
                    
                    # [조건 충족 시 섹터 조회] - 속도를 위해 여기서만 호출
                    sector = get_stock_sector(real_ticker)

                    window_52w = df.iloc[-252:]
                    curr_high_date_val = window_52w['Close'].idxmax()
                    curr_high_date_str = curr_high_date_val.strftime('%Y-%m-%d')
                    
                    prev_window = window_52w[window_52w.index < curr_high_date_val]
                    if len(prev_window) > 0:
                        prev_high_date_val = prev_window['Close'].idxmax()
                        prev_high_date_str = prev_high_date_val.strftime('%Y-%m-%d')
                        diff_days = (curr_high_date_val - prev_high_date_val).days
                    else:
                        prev_high_date_str = "-"
                        diff_days = 0
                    
                    bw_val = curr['BandWidth']
                    bw_str = f"{bw_val:.4f}"
                    if bw_val < 0.25: bw_str += " (low_vol)"

                    week_sig = "Yes" if check_weekly_macd_signal_helper(real_ticker) else "No"

                    results.append({
                        '종목코드': real_ticker,
                        '섹터': sector, # 섹터 열 추가
                        '현재가': f"{curr['Close']:,.0f}",
                        '기준': '일봉(60,2) 돌파',
                        '현52주신고가일': curr_high_date_str,
                        '전52주신고가일': prev_high_date_str,
                        '차이일': f"{diff_days}일",
                        'BW': bw_str,
                        'MACD_OSC>0': "Yes" if curr['MACD_OSC_Custom'] > 0 else "No",
                        'VR>180': "Yes" if curr['VR20'] > 180 else f"No ({curr['VR20']:.0f})",
                        '일봉MACD매수': "Yes" if df['Day_Buy_Signal'].iloc[-3:].any() else "No",
                        '주봉MACD매수': week_sig
                    })
            except Exception as e:
                continue
                
        status_text.text("일봉 분석 완료!")
        progress_bar.empty()
        
        if len(results) > 0:
            st.success(f"[일봉] 조건을 만족하는 {len(results)}개 종목 발견!")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("[일봉] 조건 만족 종목 없음.")

# ==========================================
# [B] 주봉 분석 로직
# ==========================================
if col2.button("📅 주봉 분석 시작 (Weekly)"):
    tickers = get_tickers_from_sheet()
    
    if not tickers:
        st.error("시트에서 종목을 읽어오지 못했습니다. 공유 설정을 확인해주세요.")
    else:
        st.info(f"[주봉] {len(tickers)}개 종목 분석 시작...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        for i, raw_ticker in enumerate(tickers):
            progress_bar.progress((i + 1) / len(tickers))
            status_text.text(f"주봉 분석 중: {raw_ticker}")
            
            if not raw_ticker: continue
            
            real_ticker, df = smart_download(raw_ticker, interval="1wk")
            if len(df) == 0: continue
                
            try:
                df = calculate_weekly_indicators(df)
                if df is None: continue
                
                curr = df.iloc[-1]
                
                # === 조건: Price > BB Upper (20주, 2) ===
                if curr['Close'] > curr['BB_UP']:
                    
                    # === 조건 추가: MACD OSC > 0 ===
                    if curr['MACD_OSC'] > 0:
                        
                        # [조건 충족 시 섹터 조회]
                        sector = get_stock_sector(real_ticker)

                        window_52w = df.iloc[-52:]
                        curr_high_date_val = window_52w['Close'].idxmax()
                        curr_high_date_str = curr_high_date_val.strftime('%Y-%m-%d')
                        
                        prev_window = window_52w[window_52w.index < curr_high_date_val]
                        if len(prev_window) > 0:
                            prev_high_date_val = prev_window['Close'].idxmax()
                            prev_high_date_str = prev_high_date_val.strftime('%Y-%m-%d')
                            diff_days = (curr_high_date_val - prev_high_date_val).days
                        else:
                            prev_high_date_str = "-"
                            diff_days = 0

                        bw_val = curr['BandWidth']
                        bw_str = f"{bw_val:.4f}"
                        if bw_val < 0.25: bw_str += " (low_vol)"

                        results.append({
                            '종목코드': real_ticker,
                            '섹터': sector, # 섹터 열 추가
                            '현재가': f"{curr['Close']:,.0f}",
                            '기준': '주봉(20,2) 돌파',
                            '현52주신고가일': curr_high_date_str,
                            '전52주신고가일': prev_high_date_str,
                            '차이일': f"{diff_days}일",
                            'BW(20주)': bw_str,
                            'MACD(12,26,9)': f"{curr['MACD_Line']:.2f}",
                            'MACD_OSC>0': "Yes",
                            'VR(20주)>180': "Yes" if curr['VR20'] > 180 else f"No ({curr['VR20']:.0f})"
                        })
            except Exception as e:
                continue
                
        status_text.text("주봉 분석 완료!")
        progress_bar.empty()
        
        if len(results) > 0:
            st.success(f"[주봉] 조건을 만족하는 {len(results)}개 종목 발견!")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("[주봉] 조건 만족 종목 없음.")
