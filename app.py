import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# 페이지 설정
st.set_page_config(page_title="퀀트 투자 나침반", layout="wide")

# --- 데이터 수집 함수 (에러 없는 자체 수집 로직) ---
@st.cache_data(ttl=3600) # 1시간마다 데이터 갱신
def get_current_data():
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=450) # 모멘텀 계산용 과거 데이터
    
    # 1. 야후 파이낸스 ETF 데이터 수집
    assets = {
        'QQQ': 'QQQ', 'SCHD': 'SCHD', 'GLD': 'GLD', 
        'USO': 'USO', 'TLT': 'TLT', 'BIL': 'BIL'
    }
    
    prices = pd.DataFrame()
    for name, ticker in assets.items():
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            prices[name] = data['Adj Close'].iloc[:, 0] if 'Adj Close' in data.columns.get_level_values(0) else data['Close'].iloc[:, 0]
        else:
            prices[name] = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            
    # 2. FRED 거시 경제 지표 수집 (에러 모듈 없이 URL 직접 호출)
    def fetch_fred(series_id):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_date.strftime('%Y-%m-%d')}"
        df = pd.read_csv(url, index_col='DATE', parse_dates=True, na_values='.')
        return df[series_id]

    try:
        effr = fetch_fred('EFFR')
        dgs3mo = fetch_fred('DGS3MO')
        macro = pd.DataFrame({'EFFR': effr, 'DGS3MO': dgs3mo}).ffill()
    except Exception:
        macro = pd.DataFrame({'EFFR': [0.0], 'DGS3MO': [0.0]}, index=[end_date])
    
    return prices.ffill(), macro

# --- 투자 로직 함수 ---
def get_investment_signal(prices, macro, method):
    # 월말 데이터 추출
    monthly = prices.resample('M').last()
    
    # 최신 금리 지표
    last_macro = macro.iloc[-1]
    effr, dgs3mo = last_macro['EFFR'], last_macro['DGS3MO']
    
    # 모멘텀 점수 계산
    def calc_mom(p_series, m):
        curr = p_series.iloc[-1]
        if m == '3m':
            return (curr / p_series.iloc[-4]) - 1
        elif m == 'weighted': # 0.5*(6m+12m)-3m+1m
            m1 = (curr / p_series.iloc[-2]) - 1
            m3 = (curr / p_series.iloc[-4]) - 1
            m6 = (curr / p_series.iloc[-7]) - 1
            m12 = (curr / p_series.iloc[-13]) - 1
            return 0.5 * (m6 + m12) - m3 + m1
            
    mom_scores = {col: calc_mom(monthly[col], method) for col in prices.columns}
    
    qqq_m, schd_m = mom_scores['QQQ'], mom_scores['SCHD']
    gld_m, uso_m = mom_scores['GLD'], mom_scores['USO']
    
    # 의사결정 트리 (성장장 OR 조건 유지)
    if max(qqq_m, schd_m) > 0:
        asset = 'QQQ' if qqq_m > schd_m else 'SCHD'
        reason = "🌱 성장장 유지 (주식 모멘텀 양수)"
    elif max(gld_m, uso_m) > 0:
        asset = 'GLD' if gld_m > uso_m else 'USO'
        reason = "🛢️ 원자재 장세 (주식 하락 및 원자재 모멘텀 양수)"
    else:
        asset = 'TLT' if effr > dgs3mo else 'BIL'
        reason = f"❄️ 침체기/금리 국면 (모든 자산 모멘텀 음수) | 금리상태: EFFR({effr:.2f}%) vs 3개월물({dgs3mo:.2f}%)"
        
    return asset, reason, mom_scores

# --- UI 구성 ---
st.title("🧭 시니어 퀀트의 투자 나침반")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 전략 설명서 (Strategy Manual)")
    st.info("""
    **시장의 계절과 금리의 흐름을 읽는 동적 자산 배분 로직**
    
    1. **Step 1: 성장장 확인 (Growth)**
       - 나스닥(QQQ)이나 배당주(SCHD) 중 하나라도 추세가 살아있다면 주식에 투자합니다.
       - 나스닥이 강하면 **기술주 중심 성장장**, 배당주가 강하면 **가치주 중심 인플레 성장장**으로 판단합니다.
    
    2. **Step 2: 원자재 확인 (Commodity)**
       - 주식이 모두 꺾였을 때, 금(GLD)이나 원유(USO)의 추세가 살아있다면 원자재에 투자합니다.
       - 지정학적 위기나 고인플레이션 국면에서 계좌를 방어합니다.
    
    3. **Step 3: 침체 및 금리 확인 (Recession)**
       - 위 자산들이 모두 하락세일 때 가동되는 최종 방어막입니다.
       - **장단기 금리 역전(EFFR > DGS3MO)** 시 금리 인하를 기대하며 장기채(TLT)에 투자하고, 그렇지 않으면 단기채(BIL)로 현금을 보존합니다.
    """)

with col2:
    st.subheader("🚀 현재 시점 투자 자산 확인")
    st.write("아래 버튼을 클릭하면 가장 성과가 좋았던 두 가지 전략의 오늘자 포지션을 계산합니다.")
    
    if st.button("오늘의 투자 자산 보기", type="primary"):
        with st.spinner('실시간 시장 데이터를 분석 중입니다...'):
            prices, macro = get_current_data()
            
            # 1번, 2번 전략 계산
            asset3, reason3, scores3 = get_investment_signal(prices, macro, '3m')
            assetW, reasonW, scoresW = get_investment_signal(prices, macro, 'weighted')
            
            st.success(f"**기준일:** {datetime.date.today().strftime('%Y년 %m월 %d일')}")
            
            tab1, tab2 = st.tabs(["🥇 1번 전략 (3개월 모멘텀)", "🥈 2번 전략 (가중 모멘텀)"])
            
            with tab1:
                st.markdown(f"### 추천 자산: **{asset3}**")
                st.write(f"**판단 근거:** {reason3}")
                st.write("**📊 자산별 3개월 모멘텀 스코어:**")
                st.json({k: f"{v*100:.2f}%" for k, v in scores3.items()})
                
            with tab2:
                st.markdown(f"### 추천 자산: **{assetW}**")
                st.write(f"**판단 근거:** {reasonW}")
                st.write("**📊 자산별 가중 모멘텀 스코어:**")
                st.json({k: f"{v*100:.2f}%" for k, v in scoresW.items()})

st.markdown("---")
st.caption("주의: 본 서비스는 제공된 로직에 따른 데이터 분석 결과이며, 모든 투자의 책임은 투자자 본인에게 있습니다.")
