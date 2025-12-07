import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
# Supabase 라이브러리 (DB 저장용)
from supabase import create_client, Client

# =========================================================
# [설정] Supabase 연결 정보 (공유해주신 키 적용 완료)
# =========================================================
SUPABASE_URL = "https://sgpzmkfproftswevwybm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNncHpta2Zwcm9mdHN3ZXZ3eWJtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ5OTQ0MDEsImV4cCI6MjA4MDU3MDQwMX0.VwStTHOr7_SqYrfwqol1E3ab89HsoUArV1q1s7UFAR4"

# ==========================================
# 1. 페이지 설정 및 DB 연결 초기화
# ==========================================
st.set_page_config(page_title="주식 조건 검색기", layout="wide")
st.title("📈 주식 기술적 지표 필터링 & DB 자동 저장")

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
GID = '0' 
CSV_URL = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}'

# ==========================================
# 3. 함수 정의 (데이터 수집/계산/저장)
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
    """한국/해외 주식 티커 처리 및 데이터 다운로드"""
    ticker = ticker.replace('/', '-')
    candidates = [ticker]
    # 숫자 6자리인 경우 한국 주식(.KS, .KQ) 시도
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
    """종목의 섹터(산업군) 정보 조회"""
    try:
        info = yf.Ticker(ticker).info
        sector = info.get('sector', 'N/A')
        # 한글 변환 맵핑
        translations = {
            'Technology': '기술', 'Healthcare': '헬스케어', 'Financial Services': '금융',
            'Consumer Cyclical': '임의소비재', 'Industrials': '산업재', 'Basic Materials': '소재',
            'Energy': '에너지', 'Utilities': '유틸리티', 'Real Estate': '부동산',
            'Communication Services': '통신', 'Consumer Defensive': '필수소비재'
        }
        return translations.get(sector, sector)
    except:
        return "Unknown"

def save_to_supabase(data_list, strategy_name):
    """분석 결과를 Supabase DB에 저장하는 함수"""
    if not supabase:
        st.error("⚠️ DB 연결 실패: Supabase 설정 오류")
        return

    rows_to_insert = []
    for item in data_list:
        # 데이터 정제 (콤마 제거, 문자열 변환 등)
        note_text = f"신고가:{item['현52주신고가일']}"
        if 'BW' in item: note_text += f" / BW:{item['BW']}"
        if 'BW(20주)' in item: note_text += f" / BW:{item['BW(20주)']}"

        rows_to_insert.append({
            "ticker": str(item['종목코드']),
            "sector": str(item.get('섹터', 'Unknown')),
            "price": str(item['현재가']).replace(',', ''), # 숫자만 저장하기 위해 콤마 제거
            "strategy": strategy_name,
            "note": note_text
        })
    
    try:
        # Supabase 'history' 테이블에 데이터 삽입
        data, count = supabase.table("history").insert(rows_to_insert).execute()
        st.toast(f"✅ {len(rows_to_insert)}개 종목이 Supabase DB에 성공적으로 저장되었습니다!", icon="💾")
    except Exception as e:
        st.error(f"DB 저장 중 에러 발생: {e}")
        st.info("💡 팁: Supabase Table Editor에서 'history' 테이블과 컬럼(ticker, sector, price, strategy, note)이 있는지 확인해주세요.")

# --- 지표 계산 로직 (일봉) ---
def calculate_daily_indicators(df):
    if len(df) < 260: return None
    df = df.copy()
    
    # 볼린저 밴드 (60일)
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

    # VR (20일)
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

# --- 지표 계산 로직 (주봉) ---
def calculate_weekly_indicators(df):
    if len(df) < 60: return None
    df = df.copy()

    # 볼린저 밴드 (20주)
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_UP'] = df['EMA20'] + (2 * df['STD20'])
    df['BB_LO'] = df['EMA20'] - (2 * df['STD20'])
    df['BandWidth'] = (df['BB_UP'] - df['BB_LO']) / df['EMA20']

    # MACD (12, 26, 9)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_OSC'] = df['MACD_Line'] - df['MACD_Signal']

    # VR (20주)
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

st.write("구글 시트의 종목을 분석하고 **조건 만족 시 Supabase DB에 자동 저장**합니다.")

if not supabase:
    st.error("⚠️ Supabase 연결 실패. 키 설정을 다시 확인해주세요.")

col1, col2 = st.columns(2)

# ==========================================
# [A] 일봉 분석 로직
# ==========================================
if col1.button("🚀 일봉 분석 (Daily)"):
    tickers = get_tickers_from_sheet()
    
    if not tickers:
        st.error("구글 시트 읽기 실패")
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
                
                # 조건: Price > BB Upper (60, 2)
                if curr['Close'] > curr['BB_UP']:
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
                        '섹터': sector,
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
            st.success(f"[일봉] {len(results)}개 종목 발견! (DB 저장 시도 중...)")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            # DB 저장 함수 호출
            save_to_supabase(results, "Daily")
        else:
            st.warning("[일봉] 조건 만족 종목 없음.")

# ==========================================
# [B] 주봉 분석 로직
# ==========================================
if col2.button("📅 주봉 분석 (Weekly)"):
    tickers = get_tickers_from_sheet()
    
    if not tickers:
        st.error("구글 시트 읽기 실패")
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
                
                # 조건: Price > BB Upper (20주, 2)
                if curr['Close'] > curr['BB_UP']:
                    # 조건: MACD OSC > 0
                    if curr['MACD_OSC'] > 0:
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
                            '섹터': sector,
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
            st.success(f"[주봉] {len(results)}개 종목 발견! (DB 저장 시도 중...)")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            # DB 저장 함수 호출
            save_to_supabase(results, "Weekly")
        else:
            st.warning("[주봉] 조건 만족 종목 없음.")
