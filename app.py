import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client

# =========================================================
# [설정] Supabase 연결 정보
# =========================================================
SUPABASE_URL = "https://sgpzmkfproftswevwybm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNncHpta2Zwcm9mdHN3ZXZ3eWJtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ5OTQ0MDEsImV4cCI6MjA4MDU3MDQwMX0.VwStTHOr7_SqYrfwqol1E3ab89HsoUArV1q1s7UFAR4"

# ==========================================
# 1. 페이지 설정 및 DB 연결
# ==========================================
st.set_page_config(page_title="Pro 주식 검색기 (MACD-V)", layout="wide")
st.title("📈 Pro 주식 검색기: MACD-V & 눌림목 분석")

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

def get_unique_tickers_from_db():
    """DB에 저장된 티커들을 중복 제거하여 가져오기 (눌림목 분석용)"""
    if not supabase: return []
    try:
        # 모든 기록 가져오기 (행 제한 1000개. 데이터 많으면 range 등 페이징 필요)
        response = supabase.table("history").select("ticker").execute()
        if response.data:
            # 중복 제거
            unique_tickers = list(set([row['ticker'] for row in response.data]))
            return unique_tickers
        return []
    except Exception as e:
        st.error(f"DB 읽기 실패: {e}")
        return []

def smart_download(ticker, interval="1d"):
    ticker = ticker.replace('/', '-')
    candidates = [ticker]
    if ticker.isdigit() and len(ticker) == 6:
        candidates = [f"{ticker}.KS", f"{ticker}.KQ", ticker]
    
    for t in candidates:
        try:
            # MACD-V 정확도를 위해 데이터 충분히(2년)
            df = yf.download(t, period="2y", interval=interval, progress=False, auto_adjust=False)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return t, df
        except:
            continue
    return ticker, pd.DataFrame()

def get_stock_sector(ticker):
    try:
        info = yf.Ticker(ticker).info
        sector = info.get('sector', 'N/A')
        translations = {
            'Technology': '기술', 'Healthcare': '헬스케어', 'Financial Services': '금융',
            'Consumer Cyclical': '임의소비재', 'Industrials': '산업재', 'Basic Materials': '소재',
            'Energy': '에너지', 'Utilities': '유틸리티', 'Real Estate': '부동산',
            'Communication Services': '통신', 'Consumer Defensive': '필수소비재'
        }
        return translations.get(sector, sector)
    except:
        return "Unknown"

# [핵심] MACD-V 계산 함수 (ATR 정규화)
def calculate_macdv(df, short=12, long=26, signal=9):
    # 1. 일반 MACD
    ema_fast = df['Close'].ewm(span=short, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=long, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    
    # 2. ATR(Average True Range) 계산 (기간은 보통 Slow 기간인 26 사용)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # ATR Smoothing (EMA 방식)
    atr = tr.ewm(span=long, adjust=False).mean()
    
    # 3. MACD-V = (MACD / ATR) * 100
    # 분모 0 방지
    macd_v = (macd_line / (atr + 1e-9)) * 100
    
    # Signal Line
    macd_v_signal = macd_v.ewm(span=signal, adjust=False).mean()
    
    return macd_v, macd_v_signal

def calculate_common_indicators(df, is_weekly=False):
    """일봉/주봉 공통 지표 계산 (MACD-V 포함)"""
    if len(df) < 100: return None
    df = df.copy()

    # --- 1. 볼린저 밴드 & BW ---
    # 주봉이면 20, 일봉이면 60
    period = 20 if is_weekly else 60
    df[f'EMA{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    df[f'STD{period}'] = df['Close'].rolling(window=period).std()
    df['BB_UP'] = df[f'EMA{period}'] + (2 * df[f'STD{period}'])
    df['BB_LO'] = df[f'EMA{period}'] - (2 * df[f'STD{period}'])
    df['BandWidth'] = (df['BB_UP'] - df['BB_LO']) / df[f'EMA{period}']

    # --- 2. MACD-V (신규 지표) ---
    # 표준 파라미터 (12, 26, 9) 사용
    df['MACD_V'], df['MACD_V_Signal'] = calculate_macdv(df, 12, 26, 9)

    # --- 3. 일반 보조지표 ---
    # MACD Custom (20, 200, 20) - 기존 요청 유지
    ema_fast_c = df['Close'].ewm(span=20, adjust=False).mean()
    ema_slow_c = df['Close'].ewm(span=200, adjust=False).mean()
    df['MACD_Line_Custom'] = ema_fast_c - ema_slow_c
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

    # 일봉 눌림목용 200 EMA 추가
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Day MACD Signal (기존)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line_Day'] = exp12 - exp26
    df['MACD_Signal_Day'] = df['MACD_Line_Day'].ewm(span=9, adjust=False).mean()
    df['Day_Buy_Signal'] = (df['MACD_Line_Day'] > df['MACD_Signal_Day']) & \
                           (df['MACD_Line_Day'].shift(1) <= df['MACD_Signal_Day'].shift(1))

    return df

def check_weekly_macd_signal_helper(ticker):
    try:
        w_df = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=False)
        if len(w_df) < 50: return False
        if isinstance(w_df.columns, pd.MultiIndex): w_df.columns = w_df.columns.get_level_values(0)
        
        # 일반 MACD 계산
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

def save_to_supabase(data_list, strategy_name):
    """업데이트된 DB 스키마에 맞춰 저장"""
    if not supabase:
        st.error("⚠️ DB 연결 실패")
        return

    rows_to_insert = []
    for item in data_list:
        rows_to_insert.append({
            "ticker": str(item['종목코드']),
            "sector": str(item.get('섹터', 'Unknown')),
            "price": str(item['현재가']).replace(',', ''),
            "strategy": strategy_name,
            # [수정] 신규 컬럼 매핑
            "high_date": str(item.get('현52주신고가일', '')),
            "bw": str(item.get('BW_Value', '')), # 숫자값 저장
            "macd_v": str(item.get('MACD_V_Value', '')) # 숫자값 저장
        })
    
    try:
        supabase.table("history").insert(rows_to_insert).execute()
        st.toast(f"✅ {len(rows_to_insert)}개 종목 DB 저장 완료! (MACD-V 포함)", icon="💾")
    except Exception as e:
        st.error(f"DB 저장 실패: {e}")
        st.info("팁: Supabase 'history' 테이블에 high_date, bw, macd_v 컬럼을 추가했는지 확인하세요.")

# ==========================================
# 4. 메인 실행 화면
# ==========================================

st.write("주식 분석 시스템 (MACD-V 지표 탑재)")
if not supabase: st.warning("⚠️ DB 연결 키 오류")

# 탭으로 기능 구분
tab1, tab2 = st.tabs(["📊 신규 종목 발굴", "📉 저장된 종목 눌림목 찾기"])

# ==========================================
# [TAB 1] 기존 일봉/주봉 분석
# ==========================================
with tab1:
    col1, col2 = st.columns(2)
    
    # [A] 일봉 분석
    if col1.button("🚀 일봉 분석 (Daily)"):
        tickers = get_tickers_from_sheet()
        if not tickers: st.error("시트 읽기 실패")
        else:
            st.info(f"[일봉] {len(tickers)}개 분석 시작...")
            progress_bar = st.progress(0)
            results = []
            
            for i, raw_ticker in enumerate(tickers):
                progress_bar.progress((i + 1) / len(tickers))
                if not raw_ticker: continue
                
                real_ticker, df = smart_download(raw_ticker, interval="1d")
                if len(df) == 0: continue
                
                try:
                    df = calculate_common_indicators(df, is_weekly=False)
                    if df is None: continue
                    curr = df.iloc[-1]
                    
                    if curr['Close'] > curr['BB_UP']:
                        sector = get_stock_sector(real_ticker)
                        # 날짜 계산
                        window_52w = df.iloc[-252:]
                        curr_high_date_val = window_52w['Close'].idxmax()
                        curr_high_date_str = curr_high_date_val.strftime('%Y-%m-%d')
                        prev_window = window_52w[window_52w.index < curr_high_date_val]
                        if len(prev_window) > 0:
                            prev_high_date_val = prev_window['Close'].idxmax()
                            prev_high_date_str = prev_high_date_val.strftime('%Y-%m-%d')
                            diff_days = (curr_high_date_val - prev_high_date_val).days
                        else:
                            prev_high_date_str = "-"; diff_days = 0
                        
                        bw_val = curr['BandWidth']
                        bw_str = f"{bw_val:.4f}"
                        if bw_val < 0.25: bw_str += " (low_vol)"

                        macdv_val = curr['MACD_V']

                        results.append({
                            '종목코드': real_ticker,
                            '섹터': sector,
                            '현재가': f"{curr['Close']:,.0f}",
                            '현52주신고가일': curr_high_date_str,
                            '전52주신고가일': prev_high_date_str,
                            '차이일': f"{diff_days}일",
                            'BW': bw_str,
                            'BW_Value': f"{bw_val:.4f}", # DB 저장용 순수 값
                            'MACD-V': f"{macdv_val:.2f}",
                            'MACD_V_Value': f"{macdv_val:.2f}", # DB 저장용
                            'MACD_OSC>0': "Yes" if curr['MACD_OSC_Custom'] > 0 else "No",
                            'VR>180': "Yes" if curr['VR20'] > 180 else f"No"
                        })
                except: continue
            
            progress_bar.empty()
            if results:
                st.success(f"[일봉] {len(results)}개 발견 및 저장!")
                st.dataframe(pd.DataFrame(results).drop(columns=['BW_Value', 'MACD_V_Value']))
                save_to_supabase(results, "Daily")
            else: st.warning("조건 만족 없음")

    # [B] 주봉 분석
    if col2.button("📅 주봉 분석 (Weekly)"):
        tickers = get_tickers_from_sheet()
        if not tickers: st.error("시트 읽기 실패")
        else:
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
                        # 일반 MACD OSC > 0 조건 유지
                        if (curr['MACD_Line_Custom'] - curr['MACD_Signal_Custom']) > 0: # 로직상 Custom OSC 사용했었음.
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
                                prev_high_date_str = "-"; diff_days = 0
                            
                            bw_val = curr['BandWidth']
                            bw_str = f"{bw_val:.4f}"
                            if bw_val < 0.25: bw_str += " (low_vol)"

                            macdv_val = curr['MACD_V']

                            results.append({
                                '종목코드': real_ticker,
                                '섹터': sector,
                                '현재가': f"{curr['Close']:,.0f}",
                                '현52주신고가일': curr_high_date_str,
                                '전52주신고가일': prev_high_date_str,
                                '차이일': f"{diff_days}일",
                                'BW(20주)': bw_str,
                                'BW_Value': f"{bw_val:.4f}",
                                'MACD-V': f"{macdv_val:.2f}",
                                'MACD_V_Value': f"{macdv_val:.2f}",
                                'MACD_OSC>0': "Yes",
                                'VR>180': "Yes" if curr['VR20'] > 180 else f"No"
                            })
                except: continue
            
            progress_bar.empty()
            if results:
                st.success(f"[주봉] {len(results)}개 발견 및 저장!")
                st.dataframe(pd.DataFrame(results).drop(columns=['BW_Value', 'MACD_V_Value']))
                save_to_supabase(results, "Weekly")
            else: st.warning("조건 만족 없음")

# ==========================================
# [TAB 2] 눌림목 찾기 (신규 기능)
# ==========================================
with tab2:
    st.markdown("### 📉 저장된 종목 중 눌림목/급등주 찾기")
    st.write("DB에 저장된 과거 종목들을 다시 불러와 현재 시점의 MACD-V 패턴을 분석합니다.")
    
    if st.button("🔍 눌림목 & 급등 패턴 분석 시작"):
        # 1. DB에서 중복제거된 티커 읽기
        db_tickers = get_unique_tickers_from_db()
        
        if not db_tickers:
            st.warning("DB에 저장된 종목이 없습니다. 먼저 '신규 종목 발굴'을 실행해주세요.")
        else:
            st.info(f"DB에서 중복 제거된 {len(db_tickers)}개 종목을 분석합니다...")
            progress_bar = st.progress(0)
            pullback_results = []
            
            for i, raw_ticker in enumerate(db_tickers):
                progress_bar.progress((i + 1) / len(db_tickers))
                
                # 데이터 다운로드 (일봉 기준 분석이 메인)
                real_ticker, df = smart_download(raw_ticker, interval="1d")
                if len(df) == 0: continue
                
                try:
                    df = calculate_common_indicators(df, is_weekly=False)
                    if df is None: continue
                    curr = df.iloc[-1]
                    macdv = curr['MACD_V']
                    price = curr['Close']
                    ema200 = curr['EMA200']
                    
                    # --- [조건 로직] ---
                    condition_type = None
                    
                    # 1-1. 공격적 추세 추종: MACD-V > 60
                    if macdv > 60:
                        condition_type = "🔥 공격적 추세 (MACD-V > 60)"
                    
                    # 1-2. 눌림목: 주가 > 200EMA AND MACD-V가 -50 ~ -100 사이
                    elif (price > ema200) and (-100 <= macdv <= -50):
                        condition_type = "🧲 눌림목 (200EMA위 & 과매도)"
                    
                    if condition_type:
                        # 52주 신고가 정보
                        window_52w = df.iloc[-252:]
                        curr_high_date_val = window_52w['Close'].idxmax()
                        curr_high_date_str = curr_high_date_val.strftime('%Y-%m-%d')
                        prev_window = window_52w[window_52w.index < curr_high_date_val]
                        if len(prev_window) > 0:
                            prev_high_date_val = prev_window['Close'].idxmax()
                            prev_high_date_str = prev_high_date_val.strftime('%Y-%m-%d')
                            diff_days = (curr_high_date_val - prev_high_date_val).days
                        else: prev_high_date_str = "-"; diff_days = 0
                        
                        pullback_results.append({
                            '종목코드': real_ticker,
                            '패턴': condition_type,
                            '현재가': f"{price:,.0f}",
                            'MACD-V': f"{macdv:.2f}",
                            '현52주신고가일': curr_high_date_str,
                            '전52주신고가일': prev_high_date_str,
                            '차이일': f"{diff_days}일"
                        })

                except: continue
            
            progress_bar.empty()
            
            if pullback_results:
                st.success(f"조건을 만족하는 {len(pullback_results)}개 종목 발견!")
                st.dataframe(pd.DataFrame(pullback_results), use_container_width=True)
            else:
                st.warning("조건(공격적 추세 또는 눌림목)을 만족하는 종목이 없습니다.")

# (하단 히스토리 조회 기능 유지)
st.markdown("---")
with st.expander("🗄️ 전체 저장 기록 보기"):
    if st.button("🔄 기록 새로고침"):
        try:
            response = supabase.table("history").select("*").order("created_at", desc=True).execute()
            if response.data:
                df_hist = pd.DataFrame(response.data)
                # 보기 좋게 컬럼 순서 정렬
                cols = ['created_at', 'ticker', 'price', 'strategy', 'high_date', 'bw', 'macd_v', 'note']
                # 실제 존재하는 컬럼만 선택
                valid_cols = [c for c in cols if c in df_hist.columns]
                st.dataframe(df_hist[valid_cols], use_container_width=True)
            else: st.info("데이터 없음")
        except Exception as e: st.error(str(e))
