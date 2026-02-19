import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone
from supabase import create_client, Client
import time
import concurrent.futures

# =========================================================
# [설정] 페이지 및 Supabase
# =========================================================
st.set_page_config(page_title="Pro 주식 검색기 (Stable)", layout="wide")

try:
    SUPABASE_URL = st.secrets["supabase"]["url"]
    SUPABASE_KEY = st.secrets["supabase"]["key"]
except:
    st.error("Secrets 설정 필요")
    st.stop()

@st.cache_resource
def init_supabase():
    try: return create_client(SUPABASE_URL, SUPABASE_KEY)
    except: return None

supabase = init_supabase()

# =========================================================
# 1. 시트 데이터 로드 (캐시 1시간 유지)
# =========================================================
SHEET_ID = '1NVThO1z2HHF0TVXVRGmbVsSU_Svyjg8fxd7E90z2o8A'

@st.cache_data(ttl=3600)
def get_tickers_from_sheet():
    url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=0'
    try:
        df = pd.read_csv(url, header=None)
        return sorted(list(set([str(x).strip() for x in df[0] if str(x).strip()])))
    except: return []

@st.cache_data(ttl=3600)
def get_etfs_from_sheet():
    url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=2023286696'
    try:
        df = pd.read_csv(url, header=None)
        etf_list = []
        for _, row in df.iterrows():
            raw = str(row[0]).strip()
            if not raw or raw.lower() in ['ticker', 'symbol', 'nan']: continue
            ticker = raw.split(':')[-1].strip() if ':' in raw else raw
            name = str(row[1]).strip() if len(row) > 1 else ticker
            if ticker: etf_list.append((ticker, name))
        return etf_list
    except: return []

@st.cache_data(ttl=3600)
def get_country_etfs_from_sheet():
    url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid=1247750129'
    try:
        df = pd.read_csv(url, header=None)
        etf_list = []
        for _, row in df.iterrows():
            raw = str(row[0]).strip()
            if not raw or raw.lower() in ['ticker', 'symbol', 'nan']: continue
            ticker = raw.split(':')[-1].strip() if ':' in raw else raw
            name = str(row[1]).strip() if len(row) > 1 else ticker
            if ticker: etf_list.append((ticker, name))
        return etf_list
    except: return []

# =========================================================
# 2. 안정적인 다운로드 및 섹터 정보 (재시도 로직 추가)
# =========================================================

def smart_download_robust(ticker, interval="1d", period="2y"):
    """
    [안정성 강화] 실패 시 3회 재시도, 0.5초 대기
    """
    ticker = str(ticker).strip()
    if ':' in ticker: ticker = ticker.split(':')[-1]
    ticker = ticker.replace('/', '-')
    
    candidates = [ticker]
    if ticker.isdigit() and len(ticker) == 6:
        candidates = [f"{ticker}.KS", f"{ticker}.KQ", ticker]
    
    for t in candidates:
        for attempt in range(3): # 3회 재시도
            try:
                # threads=False 필수 (충돌 방지)
                df = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    return t, df
            except:
                time.sleep(0.5) # 실패 시 잠시 대기
                continue
    return ticker, pd.DataFrame()

@st.cache_data(ttl=86400) # 섹터 정보는 하루동안 캐싱
def get_stock_sector(ticker):
    try:
        tick = yf.Ticker(ticker)
        # info 호출은 느리므로 실패시 그냥 Unknown 반환
        meta = tick.info
        if not meta: return "Unknown"
        
        qt = meta.get('quoteType', '').upper()
        if 'ETF' in qt or 'FUND' in qt: return f"[ETF] {meta.get('shortName', 'ETF')}"
        
        sec = meta.get('sector', '') or meta.get('industry', '')
        trans = {'Technology':'기술','Healthcare':'헬스케어','Financial Services':'금융','Industrials':'산업재',
                 'Basic Materials':'소재','Energy':'에너지','Utilities':'유틸리티','Real Estate':'부동산',
                 'Consumer Cyclical':'임의소비재','Consumer Defensive':'필수소비재','Communication Services':'통신'}
        return trans.get(sec, sec)
    except: return "Unknown"

@st.cache_data(ttl=600)
def fetch_quant_db():
    if not supabase: return {}
    try:
        r = supabase.table("quant_data").select("ticker,change_1w,change_1m,change_3m").order("created_at", desc=True).execute()
        if not r.data: return {}
        df = pd.DataFrame(r.data).drop_duplicates('ticker')
        return {row['ticker']: {'1w':row.get('change_1w','-'), '1m':row.get('change_1m','-'), '3m':row.get('change_3m','-')} for _,row in df.iterrows()}
    except: return {}

GLOBAL_QUANT = fetch_quant_db()

def get_eps(ticker):
    t = str(ticker).upper().split('.')[0]
    if t in GLOBAL_QUANT: return GLOBAL_QUANT[t]['1w'], GLOBAL_QUANT[t]['1m'], GLOBAL_QUANT[t]['3m']
    return "-","-","-"

def save_db(data, strategy):
    if not supabase or not data: return
    try:
        rows = []
        for i in data:
            rows.append({
                "ticker": str(i.get('종목코드','')), "sector": str(i.get('섹터','-')),
                "price": str(i.get('현재가','0')).replace(',',''), "strategy": strategy,
                "high_date": str(i.get('현52주신고가일','')), "bw": str(i.get('BW_Value','')),
                "macd_v": str(i.get('MACD_V_Value',''))
            })
        supabase.table("history").insert(rows).execute()
        st.toast("저장 완료", icon="💾")
    except: pass

# =========================================================
# 3. 지표 계산 (공통)
# =========================================================

def calc_indicators(df, weekly=False):
    if len(df) < 60: return None
    df = df.copy()
    
    # 이평선 및 볼린저
    p = 20 if weekly else 50
    df['MA'] = df['Close'].rolling(p).mean()
    df['STD'] = df['Close'].rolling(p).std()
    df['BB_UP'] = df['MA'] + 2*df['STD']
    df['BB_LO'] = df['MA'] - 2*df['STD']
    df['BW'] = (df['BB_UP'] - df['BB_LO']) / df['MA']
    
    # MACD-V
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd = ema12 - ema26
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=26).mean()
    df['MACD_V'] = (macd / (atr + 1e-9)) * 100
    
    return df

# =========================================================
# 4. 분석 Task 함수들 (Global Scope)
# =========================================================

def task_vcp(t):
    try:
        real_t, df = smart_download_robust(t, "1d", "2y")
        if len(df) < 200: return None
        
        # VCP 로직 (간소화)
        df = calc_indicators(df)
        curr = df.iloc[-1]
        ma50 = curr['MA']; ma200 = df['Close'].rolling(200).mean().iloc[-1]
        
        # 1. 추세 조건
        if not (curr['Close'] > ma200 and ma50 > ma200): return None
        
        # 2. 변동성 축소 확인
        sub = df.iloc[-60:]
        p1=sub.iloc[:20]; p2=sub.iloc[20:40]; p3=sub.iloc[40:]
        r1=(p1['High'].max()-p1['Low'].min())/p1['High'].max()
        r2=(p2['High'].max()-p2['Low'].min())/p2['High'].max()
        r3=(p3['High'].max()-p3['Low'].min())/p3['High'].max()
        
        if not ((r3 < r2) or (r2 < r1) or (r3 < 0.12)): return None
        
        status = "3단계 (수렴)"
        pivot = p3['High'].max()
        if curr['Close'] > pivot: status = "4단계 (돌파!🚀)"
        
        e1,e2,e3 = get_eps(real_t)
        return {
            '종목코드':real_t, '섹터':get_stock_sector(real_t), '현재가':curr['Close'], 
            '비고':status, 'Pivot':pivot, '1W변화':e1, '1M변화':e2,
            'chart_df':df, 'pivot':pivot # 차트용
        }
    except: return None

def task_daily(t):
    try:
        real_t, df = smart_download_robust(t, "1d", "2y")
        if len(df)<200: return None
        df = calc_indicators(df)
        curr = df.iloc[-1]
        
        # 볼린저 상단 or 신고가 근처
        donchian = df['High'].rolling(50).max().shift(1).iloc[-1]
        if (curr['Close'] > donchian) or (curr['Close'] > curr['BB_UP']):
             e1,e2,e3 = get_eps(real_t)
             return {
                 '종목코드':real_t, '섹터':get_stock_sector(real_t), '현재가':curr['Close'],
                 'BW_Value':curr['BW'], 'MACD_V_Value':curr['MACD_V'], '1W변화':e1
             }
    except: return None

def task_momentum(item):
    t, n = item
    try:
        real_t, df = smart_download_robust(t, "1d", "2y")
        if len(df) < 60: return None
        c = df['Close']
        r12 = c.pct_change(252).iloc[-1] if len(c)>252 else 0
        r6 = c.pct_change(126).iloc[-1] if len(c)>126 else 0
        r3 = c.pct_change(63).iloc[-1] if len(c)>63 else 0
        r1 = c.pct_change(21).iloc[-1] if len(c)>21 else 0
        score = ((r12+r6)/2 - r3 + r1) * 100
        
        return {'종목코드':f"{real_t} ({n})", '모멘텀점수':score, '현재가':c.iloc[-1]}
    except: return None

# =========================================================
# 5. 실행 함수 (정렬 보장)
# =========================================================

def run_analysis_stable(items, func, workers=4):
    """
    workers=4 로 제한하여 API 안정성 확보
    결과를 종목코드 기준으로 정렬하여 항상 같은 순서 보장
    """
    results = []
    bar = st.progress(0)
    status = st.empty()
    total = len(items)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(func, item): item for item in items}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                res = future.result()
                if res: results.append(res)
            except: pass
            
            # 진행률 표시
            prog = (i+1)/total
            bar.progress(prog)
            status.text(f"⏳ 안정적 분석 진행 중... {i+1}/{total}")
            
    bar.empty()
    status.empty()
    
    # [핵심] 결과 정렬 (이게 없으면 매번 순서가 바뀜)
    if results:
        # 딕셔너리에 '모멘텀점수'가 있으면 점수순, 아니면 종목코드순 정렬
        if '모멘텀점수' in results[0]:
            results.sort(key=lambda x: x['모멘텀점수'], reverse=True)
        else:
            results.sort(key=lambda x: x['종목코드'])
            
    return results

# =========================================================
# 6. 메인 UI
# =========================================================

# Session State
if 'vcp_res' not in st.session_state: st.session_state.vcp_res = None
if 'daily_res' not in st.session_state: st.session_state.daily_res = None
if 'etf_res' not in st.session_state: st.session_state.etf_res = None

tab1, tab2, tab3, tab4 = st.tabs(["🧭 나침판/ETF", "📊 기술적분석", "💰 재무/데이터", "🛠️ 설정"])

# --- 1. 나침판 & ETF ---
with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🧭 나침판")
        if st.button("나침판 실행"):
            OFFENSE = ["QQQ", "SCHD", "IMTM", "GLD", "EMGF"]
            try:
                data = yf.download(OFFENSE + ["BIL"], period="2y", progress=False)['Close']
                # 월봉 리샘플링
                m = data.resample('ME').last()
                scores = []
                for t in OFFENSE:
                    if t in m.columns:
                        r12=m[t].pct_change(12).iloc[-1]; r6=m[t].pct_change(6).iloc[-1]
                        r3=m[t].pct_change(3).iloc[-1]; r1=m[t].pct_change(1).iloc[-1]
                        score = ((r12+r6)/2 - r3 + r1)*100
                        scores.append({'Ticker':t, 'Score':score})
                df = pd.DataFrame(scores).sort_values('Score', ascending=False)
                best = df.iloc[0]['Ticker'] if df.iloc[0]['Score'] > 0 else "BIL"
                st.success(f"추천: {best}")
                st.dataframe(df)
            except: st.error("데이터 부족")

    with col_b:
        st.markdown("#### 🌍 ETF 분석")
        if st.button("ETF 모멘텀 분석"):
            etfs = get_etfs_from_sheet()
            if etfs:
                st.session_state.etf_res = run_analysis_stable(etfs, task_momentum, workers=5)
        
        if st.session_state.etf_res:
            df = pd.DataFrame(st.session_state.etf_res)
            st.dataframe(df.style.format({'모멘텀점수':'{:.2f}', '현재가':'{:,.0f}'}), use_container_width=True)

# --- 2. 기술적 분석 ---
with tab2:
    c1, c2 = st.columns(2)
    
    with c1:
        if st.button("🌪️ VCP 분석 (안정모드)"):
            ts = get_tickers_from_sheet()
            if ts: st.session_state.vcp_res = run_analysis_stable(ts, task_vcp, workers=5)
            
    with c2:
        if st.button("🚀 일봉 5-Factor"):
            ts = get_tickers_from_sheet()
            if ts: st.session_state.daily_res = run_analysis_stable(ts, task_daily, workers=5)
            
    # VCP 결과
    if st.session_state.vcp_res:
        st.write("---")
        st.markdown("##### 🌪️ VCP 결과")
        # 차트 분리
        disp = []
        charts = {}
        for r in st.session_state.vcp_res:
            row = r.copy()
            charts[row['종목코드']] = {'df':row.pop('chart_df'), 'pivot':row.pop('pivot')}
            row['현재가'] = f"{row['현재가']:,.0f}"
            row['Pivot'] = f"{row['Pivot']:,.0f}"
            disp.append(row)
        
        st.dataframe(pd.DataFrame(disp), use_container_width=True)
        save_db(disp, "VCP")
        
        # 돌파 차트
        bk = [k for k,v in charts.items() if "돌파" in ([x for x in disp if x['종목코드']==k][0]['비고'])]
        if bk:
            st.write("🔥 돌파 종목 차트")
            for i in range(0, len(bk), 2):
                cc1, cc2 = st.columns(2)
                t1 = bk[i]
                fig1 = go.Figure(data=[go.Candlestick(x=charts[t1]['df'].index, open=charts[t1]['df']['Open'], high=charts[t1]['df']['High'], low=charts[t1]['df']['Low'], close=charts[t1]['df']['Close'])])
                fig1.add_hline(y=charts[t1]['pivot'], line_dash="dot", line_color="red")
                fig1.update_layout(title=t1, height=350, template="plotly_dark", xaxis_rangeslider_visible=False)
                cc1.plotly_chart(fig1, use_container_width=True)
                
                if i+1 < len(bk):
                    t2 = bk[i+1]
                    fig2 = go.Figure(data=[go.Candlestick(x=charts[t2]['df'].index, open=charts[t2]['df']['Open'], high=charts[t2]['df']['High'], low=charts[t2]['df']['Low'], close=charts[t2]['df']['Close'])])
                    fig2.add_hline(y=charts[t2]['pivot'], line_dash="dot", line_color="red")
                    fig2.update_layout(title=t2, height=350, template="plotly_dark", xaxis_rangeslider_visible=False)
                    cc2.plotly_chart(fig2, use_container_width=True)

    # 일봉 결과
    if st.session_state.daily_res:
        st.write("---")
        st.markdown("##### 🚀 일봉 결과")
        df = pd.DataFrame(st.session_state.daily_res)
        st.dataframe(df.style.format({'현재가':'{:,.0f}', 'BW_Value':'{:.4f}', 'MACD_V_Value':'{:.2f}'}), use_container_width=True)
        save_db(st.session_state.daily_res, "Daily")

# --- 3. 재무/데이터 ---
with tab3:
    if st.button("데이터 관리"):
        st.info("기능 준비중입니다 (기존 코드 참고)")

# --- 4. 설정 ---
with tab4:
    if st.button("DB 기록 보기"):
        r = supabase.table("history").select("*").order("created_at", desc=True).limit(20).execute()
        st.dataframe(pd.DataFrame(r.data))
