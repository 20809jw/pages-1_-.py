import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="서울 기온 예측기 - 모델 학습 기간 비교", layout="wide")
st.title("🧪 서울 연평균 기온 회귀 모델 평가 및 비교")

# 1. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL, encoding="utf-8")
        df["날짜"] = pd.to_datetime(df["날짜"])
        df["연도"] = df["날짜"].dt.year
        
        # 2025년 이하 데이터만 선택 & 결측치 제거
        df = df[(df["연도"] <= 2025) & (df["평균기온"].notna())]
        
        # 연도별 관측일수 및 평균기온 계산
        yearly_summary = df.groupby("연도").agg(
            관측일수=("평균기온", "count"),
            평균기온=("평균기온", "mean")
        ).reset_index()
        
        # 관측일수 300일 이상인 연도만 필터링
        filtered_data = yearly_summary[yearly_summary["관측일수"] >= 300].copy()
        filtered_data["elapsed_years"] = filtered_data["연도"] - 1908
        return filtered_data
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame()

data = load_data()

if not data.empty:
    # 순수 numpy 기반 평가 지표 계산 함수 (scikit-learn 미사용)
    def train_and_eval(train_df, test_df):
        x_train = train_df["elapsed_years"].values
        y_train = train_df["평균기온"].values
        
        slope, intercept = np.polyfit(x_train, y_train, 1)
        
        x_test = test_df["elapsed_years"].values
        y_test = test_df["평균기온"].values
        y_pred = slope * x_test + intercept
        
        # MAE, MSE, R2 계산
        mae = np.mean(np.abs(y_test - y_pred))
        mse = np.mean((y_test - y_pred) ** 2)
        
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            "slope": slope,
            "intercept": intercept,
            "rate_100y": slope * 100,
            "mae": mae,
            "mse": mse,
            "r2": r2,
            "train_len": len(train_df)
        }

    # 데이터 분할
    test_data = data[(data["연도"] >= 2006) & (data["연도"] <= 2025)]
    train_50y = data[(data["연도"] >= 1956) & (data["연도"] <= 2005)]
    train_100y = data[(data["연도"] >= 1906) & (data["연도"] <= 2005)]
    train_full = data.copy()

    # 모델 학습 및 평가
    res_50y = train_and_eval(train_50y, test_data)
    res_100y = train_and_eval(train_100y, test_data)
    res_full = train_and_eval(train_full, test_data)

    # 사이드바
    st.sidebar.header("예측 연도 선택")
    selected_year = st.sidebar.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)
    selected_elapsed = selected_year - 1908

    pred_50y = res_50y["slope"] * selected_elapsed + res_50y["intercept"]
    pred_100y = res_100y["slope"] * selected_elapsed + res_100y["intercept"]
    pred_full = res_full["slope"] * selected_elapsed + res_full["intercept"]

    # 표 출력
    st.subheader("📊 테스트 데이터(2006~2025년) 예측 성능 및 기울기 비교")
    summary_df = pd.DataFrame({
        "구분": ["최근 50년 학습 (1956~2005)", "최근 100년 학습 (1906~2005)", "전체 데이터 학습 (참고)"],
        "학습 데이터 수": [f"{res_50y['train_len']}개 해", f"{res_100y['train_len']}개 해", f"{res_full['train_len']}개 해"],
        "100년당 기온 상승량": [f"+{res_50y['rate_100y']:.2f} ℃", f"+{res_100y['rate_100y']:.2f} ℃", f"+{res_full['rate_100y']:.2f} ℃"],
        "MAE (평균 절대 오차)": [f"{res_50y['mae']:.4f}", f"{res_100y['mae']:.4f}", f"{res_full['mae']:.4f}"],
        "MSE (평균 제곱 오차)": [f"{res_50y['mse']:.4f}", f"{res_100y['mse']:.4f}", f"{res_full['mse']:.4f}"],
        "R² (결정계수)": [f"{res_50y['r2']:.4f}", f"{res_100y['r2']:.4f}", f"{res_full['r2']:.4f}"],
        f"{selected_year}년 예상기온": [f"{pred_50y:.2f} ℃", f"{pred_100y:.2f} ℃", f"{pred_full:.2f} ℃"]
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # 그래프 출력
    plot_years = np.arange(1900, 2101)
    plot_x = plot_years - 1908

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[data["연도"] <= 2005]["연도"], y=data[data["연도"] <= 2005]["평균기온"], mode="markers", name="과거 관측치 (~2005)", marker=dict(color="lightslategray", size=6)))
    fig.add_trace(go.Scatter(x=test_data["연도"], y=test_data["평균기온"], mode="markers", name="테스트 관측치 (2006~2025)", marker=dict(color="red", size=9, symbol="diamond")))
    fig.add_trace(go.Scatter(x=plot_years, y=res_50y["slope"] * plot_x + res_50y["intercept"], mode="lines", name=f"50년 학습 회귀선 (+{res_50y['rate_100y']:.2f}℃/100년)", line=dict(color="royalblue", width=2.5)))
    fig.add_trace(go.Scatter(x=plot_years, y=res_100y["slope"] * plot_x + res_100y["intercept"], mode="lines", name=f"100년 학습 회귀선 (+{res_100y['rate_100y']:.2f}℃/100년)", line=dict(color="mediumseagreen", width=2.5, dash="dash")))

    fig.update_layout(title="학습 기간별 회귀선 비교 및 테스트 데이터 예측", xaxis_title="연도", yaxis_title="평균기온 (℃)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
