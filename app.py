import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 예측기 - 모델 학습 기간 비교", layout="wide")
st.title("🧪 서울 연평균 기온 회귀 모델 평가 및 비교")

# 1. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
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

data = load_data()

# 2. 데이터셋 분할 및 회귀 모델 학습 함수
def train_and_eval(train_df, test_df):
    x_train = train_df["elapsed_years"].values
