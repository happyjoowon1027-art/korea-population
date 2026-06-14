import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 웹앱 기본 설정 및 제목 (학술 논문 스타일)
st.set_page_config(page_title="대한민국 인구예측 시뮬레이터", layout="wide")
st.title("대한민국 인구 변천에 대한 수리생물학적 모형 적합성 분석 및 시뮬레이션")
st.markdown("---")

# 2. 데이터 불러오기 (GitHub/Streamlit Cloud 배포용)
@st.cache_data
def load_data():
    return pd.read_csv("korea.csv")

try:
    df = load_data()
except Exception as e:
    st.error("⚠️ korea.csv 파일을 불러올 수 없습니다.")
    st.error(str(e))
    st.stop()

# 시간 변수 설정 (1946년이 t=0)
df['t'] = df['Year'] - 1946
t_range = df['t'].values
years = df['Year'].values
actual_pop = df['Population'].values
N0 = 19369000  # 1946년 대한민국 초기 인구값 고정

# 3. 사이드바에 조절 바(Slider) 배치
st.sidebar.header("🔬 모델 파라미터 조절 (Parameters)")

st.sidebar.subheader("1. 맬서스 모형 설정")
r_malthus = st.sidebar.slider(
    "맬서스 성장률 (r_m)",
    min_value=0.0, max_value=0.05, value=0.014, step=0.001, format="%.3f"
)

st.sidebar.subheader("2. 로지스틱 모형 설정")
r_logistic = st.sidebar.slider(
    "로지스틱 성장률 (r_l)",
    min_value=0.0, max_value=0.05, value=0.023, step=0.001, format="%.3f"
)

K_logistic = st.sidebar.slider(
    "환경수용력 (K) [단위: 백만 명]",
    min_value=30.0, max_value=70.0, value=52.0, step=0.5
) * 1000000

# 4. 수리 모형 방정식 계산
pred_malthus = N0 * np.exp(r_malthus * t_range)

pred_logistic = K_logistic / (
    1 + ((K_logistic - N0) / N0) * np.exp(-r_logistic * t_range)
)

# 5. 평균절대오차(MAE) 계산
mae_malthus = np.mean(np.abs(actual_pop - pred_malthus))
mae_logistic = np.mean(np.abs(actual_pop - pred_logistic))

# 대시보드 상단에 오차 점수 표시
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="📊 맬서스 모형 평균 오차 (MAE)",
        value=f"{mae_malthus:,.0f} 명"
    )

with col2:
    st.metric(
        label="📈 로지스틱 모형 평균 오차 (MAE)",
        value=f"{mae_logistic:,.0f} 명"
    )

st.caption(
    "💡 팁: 오른쪽 범례(Legend)를 클릭하면 그래프를 숨기거나 다시 표시할 수 있습니다."
)

# 6. Plotly 인터랙티브 그래프 구현
fig = go.Figure()

# 실제 인구
fig.add_trace(
    go.Scatter(
        x=years,
        y=actual_pop,
        mode='lines+markers',
        name='실제 대한민국 인구',
        line=dict(color='black', width=3)
    )
)

# 맬서스 모형
fig.add_trace(
    go.Scatter(
        x=years,
        y=pred_malthus,
        mode='lines',
        name='맬서스 모형 예측',
        line=dict(color='red', width=2, dash='dash')
    )
)

# 로지스틱 모형
fig.add_trace(
    go.Scatter(
        x=years,
        y=pred_logistic,
        mode='lines',
        name='로지스틱 모형 예측',
        line=dict(color='blue', width=2.5)
    )
)

fig.update_layout(
    title="대한민국 실제 인구 및 수리 모델 예측 비교 (1946-2023)",
    xaxis_title="연도 (Year)",
    yaxis_title="인구수 (명)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255,255,255,0.5)"
    )
)

st.plotly_chart(fig, use_container_width=True)
