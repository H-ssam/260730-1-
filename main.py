import re
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="전국 고령화 지도", layout="wide")
st.title("🗺️ 전국 고령화 지도")
st.caption("시군구별 65세 이상 인구 비율 (행정안전부 주민등록 인구)")

POP_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/population_yearly.csv.gz"
GEO_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/boundaries/sigungu_kr.geojson"


@st.cache_data(show_spinner="인구 데이터를 불러오는 중입니다...")
def load_population():
    # '코드' 열은 앞자리 0이 사라지지 않게 글자로 읽습니다
    return pd.read_csv(POP_URL, dtype={"코드": str})


@st.cache_data(show_spinner="지도 경계를 불러오는 중입니다...")
def load_geojson():
    return requests.get(GEO_URL, timeout=30).json()


df = load_population()
geojson = load_geojson()

# 1. 가장 최신 연도만 사용
latest_year = int(df["연도"].max())
df = df[df["연도"] == latest_year].copy()

# 2. '계_'로 시작하는 나이 열만 (남_·여_ 열까지 더하면 두 배가 됩니다)
total_cols = [c for c in df.columns if c.startswith("계_")]


def age_of(col):
    m = re.match(r"계_(\d+)세", col)
    return int(m.group(1)) if m else None


# 3. 그중 65세 이상 열만 ('계_65세' ~ '계_100세 이상')
elderly_cols = [c for c in total_cols if age_of(c) is not None and age_of(c) >= 65]

# 4. 동 단위로 전체 인구·고령 인구 계산
df["전체인구"] = df[total_cols].sum(axis=1)
df["고령인구"] = df[elderly_cols].sum(axis=1)

# 5. '코드' 앞 5자리 = 시군구 코드 → 시군구별로 묶어 비율 계산
df["시군구코드"] = df["코드"].str[:5]
grouped = df.groupby("시군구코드")[["전체인구", "고령인구"]].sum().reset_index()
grouped["고령화율"] = (grouped["고령인구"] / grouped["전체인구"] * 100).round(2)

# 경계 파일에서 코드 → 시군구·시도 이름 짝 만들기
names = pd.DataFrame([
    {
        "시군구코드": str(f["properties"]["코드"]),
        "시군구": f["properties"]["시군구"],
        "시도": f["properties"]["시도"],
    }
    for f in geojson["features"]
])
merged = grouped.merge(names, on="시군구코드", how="left")

# 6. 5단계 색 구간 (전국 시군구를 다섯 덩어리로 나눈 실제 경계값)
BINS = [0, 19, 23, 28, 38, 100]
LABELS = ["19% 미만", "19~23%", "23~28%", "28~38%", "38% 이상"]
COLORS = {
    "19% 미만": "#fee6ce",
    "19~23%": "#fdc086",
    "23~28%": "#f79646",
    "28~38%": "#e8590c",
    "38% 이상": "#a63603",
}
merged["단계"] = pd.cut(merged["고령화율"], bins=BINS, labels=LABELS, right=False)

# 7. 단계구분도 그리기 (배경 지도 타일 없이 경계만)
fig = px.choropleth(
    merged,
    geojson=geojson,
    locations="시군구코드",
    featureidkey="properties.코드",
    color="단계",
    category_orders={"단계": LABELS},
    color_discrete_map=COLORS,
    hover_name="시군구",
    hover_data={"고령화율": True, "시도": True, "시군구코드": False, "단계": False},
    labels={"고령화율": "65세 이상 비율(%)"},
)
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(
    margin=dict(l=0, r=0, t=10, b=0),
    height=700,
    legend_title_text=f"65세 이상 비율 ({latest_year}년)",
)

st.plotly_chart(fig, width="stretch")

# 8. 지도 아래 순위 표 두 개
c1, c2 = st.columns(2)
cols = ["시도", "시군구", "고령화율"]
with c1:
    st.subheader("🔴 고령화율 높은 곳 10")
    st.dataframe(merged.nlargest(10, "고령화율")[cols].reset_index(drop=True))
with c2:
    st.subheader("🟢 고령화율 낮은 곳 10")
    st.dataframe(merged.nsmallest(10, "고령화율")[cols].reset_index(drop=True))

# ============================================================
# 🔮 미래 고령화율 예측 (기존 app.py 맨 아래에 이어 붙이세요)
# 필요 조건: 기존 코드의 df, names, geojson, BINS, LABELS, COLORS,
#           load_population(), re, pd, px, st 가 이미 정의되어 있어야 합니다.
# 추가로 numpy만 새로 import 하면 됩니다.
# ============================================================

import numpy as np

st.markdown("---")
st.header("🔮 미래 고령화율 예측")
st.caption(
    "최근 연도별 추세를 시군구 단위로 선형 회귀하여 연장한 단순 예측입니다. "
    "실제 통계청 장래인구추계처럼 출생·사망·이동을 반영한 것이 아니므로 참고용으로만 봐주세요."
)


@st.cache_data(show_spinner="연도별 고령화율 추이를 계산하는 중입니다...")
def build_yearly_rate(df_raw: pd.DataFrame) -> pd.DataFrame:
    """연도 필터링 없이 전체 데이터로 시군구·연도별 고령화율을 계산"""
    d = df_raw.copy()
    total_cols = [c for c in d.columns if c.startswith("계_")]

    def age_of(col):
        m = re.match(r"계_(\d+)세", col)
        return int(m.group(1)) if m else None

    elderly_cols = [c for c in total_cols if age_of(c) is not None and age_of(c) >= 65]

    d["전체인구"] = d[total_cols].sum(axis=1)
    d["고령인구"] = d[elderly_cols].sum(axis=1)
    d["시군구코드"] = d["코드"].str[:5]

    yearly = (
        d.groupby(["시군구코드", "연도"])[["전체인구", "고령인구"]]
        .sum()
        .reset_index()
    )
    yearly["고령화율"] = (yearly["고령인구"] / yearly["전체인구"] * 100).round(2)
    return yearly


# load_population()은 이미 @st.cache_data로 캐싱되어 있어 재호출해도
# 네트워크 요청이 다시 발생하지 않습니다.
df_full = load_population()
yearly = build_yearly_rate(df_full)

available_years = sorted(yearly["연도"].unique())
last_year = available_years[-1]

target_year = st.slider(
    "예측할 연도 선택",
    min_value=last_year + 1,
    max_value=last_year + 30,
    value=last_year + 10,
)


def linear_forecast(group: pd.DataFrame, year: int) -> float:
    """시군구별 (연도, 고령화율) 선형회귀로 특정 연도 값을 예측"""
    if len(group) < 2:
        return float(group["고령화율"].iloc[-1])  # 데이터가 1개뿐이면 그대로 사용
    slope, intercept = np.polyfit(group["연도"], group["고령화율"], 1)
    pred = slope * year + intercept
    return float(np.clip(pred, 0, 100))  # 0~100% 범위를 벗어나지 않게 고정


forecast_rows = [
    {"시군구코드": code, "예측고령화율": round(linear_forecast(g, target_year), 2)}
    for code, g in yearly.groupby("시군구코드")
]
forecast_df = pd.DataFrame(forecast_rows)

forecast_merged = forecast_df.merge(names, on="시군구코드", how="left")
forecast_merged["단계"] = pd.cut(
    forecast_merged["예측고령화율"], bins=BINS, labels=LABELS, right=False
)

# ---- 예측 지도 ----
fig_forecast = px.choropleth(
    forecast_merged,
    geojson=geojson,
    locations="시군구코드",
    featureidkey="properties.코드",
    color="단계",
    category_orders={"단계": LABELS},
    color_discrete_map=COLORS,
    hover_name="시군구",
    hover_data={"예측고령화율": True, "시도": True, "시군구코드": False, "단계": False},
    labels={"예측고령화율": "예측 65세 이상 비율(%)"},
)
fig_forecast.update_geos(fitbounds="locations", visible=False)
fig_forecast.update_layout(
    margin=dict(l=0, r=0, t=10, b=0),
    height=700,
    legend_title_text=f"65세 이상 비율 예측 ({target_year}년)",
)
st.plotly_chart(fig_forecast, width="stretch")

# ---- 전국 평균 추이 + 예측 라인 차트 ----
national = yearly.groupby("연도")[["전체인구", "고령인구"]].sum().reset_index()
national["고령화율"] = (national["고령인구"] / national["전체인구"] * 100).round(2)

nat_slope, nat_intercept = np.polyfit(national["연도"], national["고령화율"], 1)
future_years = list(range(last_year + 1, target_year + 1))
future_rates = [
    round(float(np.clip(nat_slope * y + nat_intercept, 0, 100)), 2) for y in future_years
]

trend_df = pd.concat(
    [
        pd.DataFrame({"연도": national["연도"], "고령화율": national["고령화율"], "구분": "실측"}),
        pd.DataFrame({"연도": future_years, "고령화율": future_rates, "구분": "예측"}),
    ]
)

fig_trend = px.line(
    trend_df,
    x="연도",
    y="고령화율",
    color="구분",
    markers=True,
    title="전국 평균 고령화율 추이 및 예측",
)
st.plotly_chart(fig_trend, width="stretch")

# ---- 예측 순위표 ----
c3, c4 = st.columns(2)
fcols = ["시도", "시군구", "예측고령화율"]
with c3:
    st.subheader(f"🔴 {target_year}년 예측 고령화율 높은 곳 10")
    st.dataframe(forecast_merged.nlargest(10, "예측고령화율")[fcols].reset_index(drop=True))
with c4:
    st.subheader(f"🟢 {target_year}년 예측 고령화율 낮은 곳 10")
    st.dataframe(forecast_merged.nsmallest(10, "예측고령화율")[fcols].reset_index(drop=True))
