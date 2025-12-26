import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from supabase import create_client, Client
from scipy.signal import argrelextrema
import time

# =========================================================
# [설정] Supabase 연결 정보
# =========================================================
SUPABASE_URL = "https://sgpzmkfproftswevwybm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNncHpta2Zwcm9mdHN3ZXZ3eWJtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQ5OTQ0MDEsImV4cCI6MjA4MDU3MDQwMX0.VwStTHOr7_SqYrfwqol1E3ab89HsoUArV1q1s7UFAR4"

# ==========================================
# 1. 페이지 설정 및 DB 연결
# ==========================================
st.set_page_config(page_title="Pro 주식 검색기", layout="wide")
st.title("📈 Pro 주식 검색기: 섹터/기술적/재무 통합 분석")

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
    df['BB50_LO'] = df['EMA50'] - (2 * df['STD50'])
    df['BW50'] = (df['BB50_UP'] - df['BB50_LO']) / df['EMA50']
    
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

    df['MACD_V'], _ = calculate_macdv(df, 12, 26, 9)

    return df

# --- 분석 헬퍼 함수 ---
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
        squeeze = (df['BB50_UP'] < df['Donchian_High_50']).iloc[-5:].any()
        win_52 = df.iloc[-252:]
        high_52_date = win_52['Close'].idxmax().strftime('%Y-%m-%d')
        prev_win = win_52[win_52.index < win_52['Close'].idxmax()]
        prev_date = prev_win['Close'].idxmax().strftime('%Y-%m-%d') if len(prev_win)>0 else "-"
        diff_days = (win_52['Close'].idxmax() - prev_win['Close'].idxmax()).days if len(prev_win)>0 else 0
        
        return True, {
            'price': curr['Close'], 'atr': curr['ATR14'], 'high_date': high_52_date,
            'prev_date': prev_date, 'diff_days': diff_days, 'bw_curr': curr['BW50'],
            'macdv': curr['MACD_V'], 'squeeze': "🔥Squeeze" if squeeze else "-"
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
        return True, {
            'price': curr['Close'], 'atr': curr['ATR14'], 'bw_curr': curr['BandWidth'],
            'bw_past': bw_past, 'bw_change': bw_change, 'macdv': curr['MACD_V']
        }
    return False, None

def check_monthly_condition(df):
    if len(df) < 12: return False, None
    ath_price = df['High'].max()
    curr_price = df['Close'].iloc[-1]
    if curr_price >= ath_price * 0.90:
        ath_idx = df['High'].idxmax()
        month_count = (df['Close'] >= ath_price * 0.90).sum()
        return True, {
            'price': curr_price, 'ath_price': ath_price,
            'ath_date': ath_idx.strftime('%Y-%m'), 'month_count': month_count
        }
    return False, None

# ==========================================
# 섹터/패턴 분석 함수
# ==========================================
def analyze_sector_trend():
    etfs = get_etfs_from_sheet()
    if not etfs: st.warning("ETF 목록 없음"); return []
    st.write(f"📊 총 {len(etfs)}개 ETF 분석 중...")
    spy_t, spy_df = smart_download("SPY", "1d", "2y")
    if len(spy_df) < 260: st.error("SPY 데이터 부족"); return []
    spy_c = spy_df['Close']
    spy_r1m = spy_c.pct_change(21).iloc[-1]; spy_r3m = spy_c.pct_change(63).iloc[-1]
    spy_r6m = spy_c.pct_change(126).iloc[-1]; spy_r12m = spy_c.pct_change(252).iloc[-1]
    
    results = []; pbar = st.progress(0)
    for i, (t, n) in enumerate(etfs):
        pbar.progress((i+1)/len(etfs))
        rt, df = smart_download(t, "1d", "2y")
        if len(df)<30: continue
        c = df['Close']; h = df['High']
        ema20=c.ewm(span=20).mean(); ema50=c.ewm(span=50).mean(); ema60=c.ewm(span=60).mean()
        ema100=c.ewm(span=100).mean(); ema200=c.ewm(span=200).mean()
        curr=c.iloc[-1]
        bb_up = ema50 + (2*c.rolling(50).std())
        dc_h = h.rolling(50).max().shift(1)
        tr = pd.concat([h-df['Low'], (h-c.shift()).abs(), (df['Low']-c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=14).mean().iloc[-1]
        macdv, _ = calculate_macdv(df)
        
        bb_bk = "O" if (c>bb_up).iloc[-3:].any() else "-"
        dc_bk = "O" if (c>dc_h).iloc[-3:].any() else "-"
        align = "⭐ 정배열" if (curr>ema20.iloc[-1] and curr>ema60.iloc[-1] and curr>ema100.iloc[-1] and curr>ema200.iloc[-1]) else "-"
        long_tr = "📈 상승" if (ema60.iloc[-1]>ema100.iloc[-1]>ema200.iloc[-1]) else "-"
        
        r1 = c.pct_change(21).iloc[-1] if len(c)>21 else 0
        r3 = c.pct_change(63).iloc[-1] if len(c)>63 else 0
        r6 = c.pct_change(126).iloc[-1] if len(c)>126 else 0
        r12 = c.pct_change(252).iloc[-1] if len(c)>252 else 0
        score = (0.25*(r1-spy_r1m) + 0.25*(r3-spy_r3m) + 0.25*(r6-spy_r6m) + 0.25*(r12-spy_r12m))*100
        
        results.append({"ETF":rt, "모멘텀점수":score, "BB(50,2)돌파":bb_bk, "돈키언(50)돌파":dc_bk, "정배열":align, "장기추세":long_tr, "MACD-V":f"{macdv.iloc[-1]:.2f}", "ATR":f"{atr:.2f}", "현재가":curr})
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

st.write("주식 분석 시스템 (5-Factor 전략, MACD-V, 재무 분석)")
if not supabase: st.warning("⚠️ DB 연결 키 오류")

tab1, tab2, tab3 = st.tabs(["📊 신규 종목 발굴", "📉 저장된 종목 눌림목 찾기", "💰 재무분석"])

with tab1:
    cols = st.columns(10) 
    
    # 1. 추세 섹터
    if cols[0].button("🌍 섹터"):
        st.info("ETF 섹터 분석 중...")
        res = analyze_sector_trend()
        if not res.empty: st.dataframe(res, use_container_width=True)
        else: st.warning("데이터 부족")

    # 2. 일봉 분석
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
                    sector = get_stock_sector(rt)
                    res.append({
                        '종목코드': rt, '섹터': sector, '현재가': f"{info['price']:,.0f}",
                        'ATR(14)': f"{info['atr']:,.0f}", '스퀴즈': info['squeeze'],
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

    # 3. 주봉 분석
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
                    sector = get_stock_sector(rt)
                    res.append({
                        '종목코드': rt, '섹터': sector, '현재가': f"{info['price']:,.0f}",
                        'ATR(14주)': f"{info['atr']:,.0f}", 'BW현재': f"{info['bw_curr']:.4f}",
                        'BW(20주전)': f"{info['bw_past']:.4f}", 'BW변화': info['bw_change'],
                        'MACD-V': f"{info['macdv']:.2f}", 'BW_Value': f"{info['bw_curr']:.4f}", 'MACD_V_Value': f"{info['macdv']:.2f}"
                    })
            bar.empty()
            if res:
                st.success(f"[주봉] {len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res).drop(columns=['BW_Value', 'MACD_V_Value']))
                save_to_supabase(res, "Weekly")
            else: st.warning("조건 만족 없음")

    # 4. 월봉 분석
    if cols[3].button("🗓️ 월봉"):
        tickers = get_tickers_from_sheet()
        if tickers:
            st.info(f"[월봉 ATH] {len(tickers)}개 분석 시작...")
            bar = st.progress(0); res = []
            for i, t in enumerate(tickers):
                bar.progress((i+1)/len(tickers))
                rt, df = smart_download(t, "1mo", "max")
                passed, info = check_monthly_condition(df)
                if passed:
                    sector = get_stock_sector(rt)
                    res.append({
                        '종목코드': rt, '섹터': sector, '현재가': f"{info['price']:,.0f}",
                        'ATH최고가': f"{info['ath_price']:,.0f}", 'ATH달성월': info['ath_date'],
                        '고권역(월수)': f"{info['month_count']}개월",
                        '현52주신고가일': info['ath_date'], 'BW_Value': str(info['month_count']), 'MACD_V_Value': "0"
                    })
            bar.empty()
            if res:
                st.success(f"[월봉] {len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res).drop(columns=['현52주신고가일', 'BW_Value', 'MACD_V_Value'], errors='ignore'))
                save_to_supabase(res, "Monthly_ATH")
            else: st.warning("조건 만족 없음")

    # [NEW] 5. 일+월봉 분석 (교차)
    if cols[4].button("일+월봉"):
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
                res.append({
                    '종목코드': rt, '섹터': sector, '현재가': f"{info_d['price']:,.0f}",
                    '스퀴즈': info_d['squeeze'], 'ATH달성월': info_m['ath_date'],
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

    # [NEW] 6. 일+주봉 분석 (교차)
    if cols[5].button("일+주봉"):
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
                res.append({
                    '종목코드': rt, '섹터': sector, '현재가': f"{info_d['price']:,.0f}",
                    '스퀴즈': info_d['squeeze'], 
                    '주봉BW': f"{info_w['bw_curr']:.4f}", '주봉BW변화': info_w['bw_change'],
                    '현52주신고가일': info_d['high_date'], '전52주신고가일': info_d['prev_date'],
                    '차이일': f"{info_d['diff_days']}일", 'BW_Value': f"{info_w['bw_curr']:.4f}", 'MACD_V_Value': f"{info_d['macdv']:.2f}"
                })
            bar.empty()
            if res:
                st.success(f"[일+주봉] {len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res))
                save_to_supabase(res, "Daily_Weekly")
            else: st.warning("조건 만족 없음")

    # [NEW] 7. 주+월봉 분석 (교차)
    if cols[6].button("주+월봉"):
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
                res.append({
                    '종목코드': rt, '섹터': sector, '현재가': f"{info_w['price']:,.0f}",
                    '주봉BW': f"{info_w['bw_curr']:.4f}", '주봉BW변화': info_w['bw_change'],
                    'ATH달성월': info_m['ath_date'], '고권역(월수)': f"{info_m['month_count']}개월",
                    '현52주신고가일': info_m['ath_date'], 'BW_Value': f"{info_w['bw_curr']:.4f}", 'MACD_V_Value': f"{info_w['macdv']:.2f}"
                })
            bar.empty()
            if res:
                st.success(f"[주+월봉] {len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res))
                save_to_supabase(res, "Weekly_Monthly")
            else: st.warning("조건 만족 없음")

    # 8. 통합 분석
    if cols[7].button("⚡ 통합"):
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
                res.append({
                    '종목코드': rt, '섹터': sector, '현재가': f"{info_d['price']:,.0f}",
                    'ATH최고가': f"{info_m['ath_price']:,.0f}", 'ATH달성월': info_m['ath_date'],
                    '해당월수': f"{info_m['month_count']}개월", '스퀴즈': info_d['squeeze'],
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

    # 9. 컵위드핸들
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
                    curr = df.iloc[-1]
                    sector = get_stock_sector(rt)
                    res.append({
                        '종목코드': rt, '섹터': sector, '현재가': f"{curr['Close']:,.0f}",
                        '패턴상세': f"깊이:{info['depth']}", '돌파가격': info['pivot'],
                        'BW_Value': f"{curr['BandWidth']:.4f}", 'MACD_V_Value': f"{curr['MACD_V']:.2f}"
                    })
            bar.empty()
            if res:
                st.success(f"[컵핸들] {len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res))
                save_to_supabase(res, "CupHandle")
            else: st.warning("조건 만족 없음")

    # 10. 역H&S
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
                    curr = df.iloc[-1]
                    sector = get_stock_sector(rt)
                    res.append({
                        '종목코드': rt, '섹터': sector, '현재가': f"{curr['Close']:,.0f}",
                        '넥라인': info['Neckline'], '거래량급증': info['Vol_Ratio'],
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
                    df = calculate_common_indicators(df, False)
                    if df is None: continue
                    curr = df.iloc[-1]
                    cond = ""
                    if curr['MACD_V'] > 60: cond = "🔥 공격적 추세"
                    
                    ema20 = df['Close'].ewm(span=20).mean().iloc[-1]
                    if (curr['Close'] > ema20) and ((curr['Close']-ema20)/ema20 < 0.03):
                        cond = "📉 20일선 눌림목"

                    if (curr['Close'] > curr['EMA200']) and (-100 <= curr['MACD_V'] <= -50):
                         cond = "🧲 MACD-V 과매도"
                    
                    if cond:
                        res.append({
                            '종목코드': rt, '패턴': cond, '현재가': f"{curr['Close']:,.0f}",
                            'MACD-V': f"{curr['MACD_V']:.2f}", 'EMA20': f"{ema20:,.0f}"
                        })
                except: continue
            bar.empty()
            if res:
                st.success(f"{len(res)}개 발견!")
                st.dataframe(pd.DataFrame(res), use_container_width=True)
            else: st.warning("조건 만족 종목 없음")

# [NEW] 재무분석 탭
with tab3:
    st.markdown("### 💰 재무 지표 분석")
    st.info("yfinance 데이터를 기반으로 핵심 재무 지표를 분석합니다.")
    if st.button("📊 재무 지표 가져오기"):
        tickers = get_tickers_from_sheet()
        if not tickers: st.error("티커 없음")
        else:
            bar = st.progress(0); f_res = []
            for i, t in enumerate(tickers):
                bar.progress((i+1)/len(tickers))
                rt, _ = smart_download(t, "1d", "5d")
                try:
                    tick = yf.Ticker(rt); info = tick.info
                    if not info: continue
                    mc = info.get('marketCap', 0)
                    mc_str = f"{mc/1000000000000:.1f}조" if mc > 1000000000000 else f"{mc/100000000:.0f}억" if mc else "-"
                    per = info.get('trailingPE', info.get('forwardPE', '-'))
                    if isinstance(per, (int, float)): per = f"{per:.2f}"
                    eps = info.get('trailingEps', info.get('forwardEps', '-'))
                    div = info.get('dividendYield', 0)
                    div_str = f"{div*100:.2f}%" if div else "-"
                    pbr = info.get('priceToBook', '-'); roe = info.get('returnOnEquity', '-')
                    if isinstance(pbr, (int, float)): pbr = f"{pbr:.2f}"
                    if isinstance(roe, (int, float)): roe = f"{roe*100:.2f}%"
                    f_res.append({
                        "종목": rt, "기업명": info.get('shortName', '-'), "시가총액": mc_str,
                        "PER": per, "EPS": eps, "PBR": pbr, "ROE": roe, "배당수익률": div_str,
                        "목표주가": info.get('targetMeanPrice', '-')
                    })
                except: continue
            bar.empty()
            if f_res:
                st.success(f"✅ 총 {len(f_res)}개 기업 재무 분석 완료")
                st.dataframe(pd.DataFrame(f_res), use_container_width=True)
            else: st.warning("데이터 실패")

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
