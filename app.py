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
# 2. 구글 시트 연결 설정 (원래 링크 적용 완료)
# ==========================================
# 요청하신 링크의 ID: 1NVThO1z2HHF0TVXVRGmbVsSU_Svyjg8fxd7E90z2o8A
SHEET_ID = '1NVThO1z2HHF0TVXVRGmbVsSU_Svyjg8fxd7E90z2o8A'
GID = '0' 
CSV_URL = f'https://docs.google.com/spreadsheets/d/{target}/export?format=csv&gid={GID}'

# ==========================================
# 3. 함수 정의
# ==========================================

# 캐시(ttl) 제거: 버튼 누를 때마다 시트를 새로 읽어옵니다. (즉시 반영)
def get_tickers_from_sheet():
    try:
        # 인증 없이 CSV로 바로 읽어오기
        df = pd.read_csv(CSV_URL, header=None)
        # 1열(0번 인덱스) 데이터 가져오기 + 문자열 변환 + 정렬
        tickers = sorted(list(set([str(x).strip() for x in df[0] if str(x).strip()])))
        return tickers
    except Exception as e:
        st.error(f"구글 시트 읽기 실패: {e}")
        return []

def smart_download(ticker):
    # 특수문자 변환 (BRK/B -> BRK-B)
    ticker = ticker.replace('/', '-')
    candidates = [ticker]
    # 한국 주식 처리 (숫자 6자리인 경우)
    if ticker.isdigit() and len(ticker) == 6:
        candidates = [f"{ticker}.KS", f"{ticker}.KQ", ticker]
    
    for t in candidates:
        try:
            # auto_adjust=False로 원본 데이터 유지
            df = yf.download(t, period="2y", progress=False, auto_adjust=False)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return t, df
        except:
            continue
    return ticker, pd.DataFrame()

def calculate_indicators(df):
    if len(df) < 260: return None
    df = df.copy()
    
    # 볼린저 밴드 (60, 2)
    df['EMA60'] = df['Close'].ewm(span=60, adjust=False).mean()
    df['STD60'] = df['Close'].rolling(window=60).std()
    df['BB_UP'] = df['EMA60'] + (2 * df['STD60'])
    df['BB_LO'] = df['EMA60'] - (2 * df['STD60'])
    df['BandWidth'] = (df['BB_UP'] - df['BB_LO']) / df['EMA60']

    # MACD Custom (20, 200, 20)
    ema_fast = df['Close'].ewm(span=20, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=200, adjust=False).mean()
    df['MACD_Line_Custom'] = ema_fast - ema_slow
    df['MACD_Signal_Custom'] = df['MACD_Line_Custom'].ewm(span=20, adjust=False).mean()
    df['MACD_OSC_Custom'] = df['MACD_Line_Custom'] - df['MACD_Signal_Custom']

    # VR (Volume Ratio)
    df['Change'] = df['Close'].diff()
    df['Vol_Up'] = np.where(df['Change'] > 0, df['Volume'], 0)
    df['Vol_Down'] = np.where(df['Change'] < 0, df['Volume'], 0)
    df['Vol_Flat'] = np.where(df['Change'] == 0, df['Volume'], 0)
    roll_up = df['Vol_Up'].rolling(window=20).sum()
    roll_down = df['Vol_Down'].rolling(window=20).sum()
    roll_flat = df['Vol_Flat'].rolling(window=20).sum()
    df['VR20'] = ((roll_up + roll_flat/2) / (roll_down + roll_flat/2 + 1e-9)) * 100

    # Day MACD (12, 26, 9)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line_Day'] = exp12 - exp26
    df['MACD_Signal_Day'] = df['MACD_Line_Day'].ewm(span=9, adjust=False).mean()
    df['Day_Buy_Signal'] = (df['MACD_Line_Day'] > df['MACD_Signal_Day']) & \
                           (df['MACD_Line_Day'].shift(1) <= df['MACD_Signal_Day'].shift(1))
    return df

def check_weekly_macd_signal(ticker):
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

st.write("구글 시트(1NVTh...)의 종목을 분석합니다. 시트를 수정하고 버튼을 누르세요.")

if st.button("🚀 분석 시작하기"):
    tickers = get_tickers_from_sheet()
    
    if not tickers:
        st.error("시트에서 종목을 읽어오지 못했습니다. 시트 공유 설정(링크가 있는 모든 사용자)을 확인해주세요.")
    else:
        st.info(f"시트에서 {len(tickers)}개의 종목을 읽어왔습니다. 분석을 시작합니다...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        for i, raw_ticker in enumerate(tickers):
            progress_bar.progress((i + 1) / len(tickers))
            status_text.text(f"분석 중 ({i+1}/{len(tickers)}): {raw_ticker}")
            
            if not raw_ticker: continue
            
            real_ticker, df = smart_download(raw_ticker)
            if len(df) == 0: continue
                
            try:
                df = calculate_indicators(df)
                if df is None: continue
                
                curr = df.iloc[-1]
                
                # === 조건: 현재가 > 볼린저밴드 상단 ===
                if curr['Close'] > curr['BB_UP']:
                    # 날짜 및 지표 계산
                    window_52w = df.iloc[-252:]
                    curr_high_date_val = window_52w['Close'].idxmax()
                    curr_high_date_str = curr_high_date_val.strftime('%Y-%m-%d')
                    
                    # 전 52주 신고가: 현재 신고가 날짜 '이전' 데이터 중에서 찾기
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

                    week_sig = "Yes" if check_weekly_macd_signal(real_ticker) else "No"

                    results.append({
                        '종목코드': real_ticker,
                        '현재가': f"{curr['Close']:,.0f}",
                        '상단돌파': 'Yes',
                        '현52주신고가일': curr_high_date_str,
                        '전52주신고가일': prev_high_date_str,
                        '차이일': f"{diff_days}일",
                        'BW(60,2)': bw_str,
                        'MACD_OSC>0': "Yes" if curr['MACD_OSC_Custom'] > 0 else "No",
                        'VR>180': "Yes" if curr['VR20'] > 180 else f"No ({curr['VR20']:.0f})",
                        '일봉MACD매수': "Yes" if df['Day_Buy_Signal'].iloc[-3:].any() else "No",
                        '주봉MACD매수': week_sig
                    })
            except Exception as e:
                continue
                
        status_text.text("분석 완료!")
        progress_bar.empty()
        
        if len(results) > 0:
            st.success(f"조건을 만족하는 {len(results)}개 종목을 발견했습니다!")
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
        else:
            st.warning("조건을 만족하는 종목이 없습니다.")