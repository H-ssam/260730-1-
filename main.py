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

# ============================================================
# 📊 지역별 전체 인구수 변화 (기존 app.py 아래에 이어 붙이세요)
# 필요 조건: 기존 코드의 df, names, geojson, load_population(), re, pd, px, st
#           가 이미 정의되어 있어야 합니다. numpy도 새로 import 합니다.
# ============================================================

st.markdown("---")
st.header("📊 지역별 전체 인구수 변화")
st.caption("고령 인구뿐 아니라 시군구 전체 인구가 어떻게 늘거나 줄었는지, 앞으로 어떻게 될지 살펴봅니다.")


@st.cache_data(show_spinner="연도별 전체 인구를 계산하는 중입니다...")
def build_yearly_population(df_raw: pd.DataFrame) -> pd.DataFrame:
    """연도 필터링 없이 전체 데이터로 시군구·연도별 총인구를 계산"""
    d = df_raw.copy()
    total_cols = [c for c in d.columns if c.startswith("계_")]  # 모든 연령 열 (남+여 합계)
    d["전체인구"] = d[total_cols].sum(axis=1)
    d["시군구코드"] = d["코드"].str[:5]

    yearly_pop = (
        d.groupby(["시군구코드", "연도"])["전체인구"]
        .sum()
        .reset_index()
    )
    return yearly_pop


df_full = load_population()
yearly_pop = build_yearly_population(df_full)

available_years = sorted(yearly_pop["연도"].unique())
first_year, last_year = available_years[0], available_years[-1]

# ---------------- 1. 과거 인구 증감 지도 ----------------
st.subheader("① 과거 인구 증감률")

col_a, col_b = st.columns(2)
with col_a:
    start_year = st.selectbox("기준 연도", available_years, index=0)
with col_b:
    end_year = st.selectbox("비교 연도", available_years, index=len(available_years) - 1)

wide = yearly_pop.pivot(index="시군구코드", columns="연도", values="전체인구")

if start_year not in wide.columns or end_year not in wide.columns:
    st.warning("선택한 연도의 데이터가 없습니다.")
else:
    change = wide[[start_year, end_year]].dropna().reset_index()
    change.columns = ["시군구코드", "시작인구", "종료인구"]
    change["증감인구"] = change["종료인구"] - change["시작인구"]
    change["증감률"] = ((change["종료인구"] / change["시작인구"] - 1) * 100).round(2)

    change_merged = change.merge(names, on="시군구코드", how="left")

    fig_change = px.choropleth(
        change_merged,
        geojson=geojson,
        locations="시군구코드",
        featureidkey="properties.코드",
        color="증감률",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,  # 0%를 기준으로 감소(빨강)/증가(파랑) 대비
        hover_name="시군구",
        hover_data={"증감률": True, "증감인구": True, "시도": True, "시군구코드": False},
        labels={"증감률": f"{start_year}→{end_year} 인구 증감률(%)"},
    )
    fig_change.update_geos(fitbounds="locations", visible=False)
    fig_change.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=700,
        coloraxis_colorbar_title=f"증감률(%)",
    )
    st.plotly_chart(fig_change, width="stretch")

    c1, c2 = st.columns(2)
    ccols = ["시도", "시군구", "시작인구", "종료인구", "증감률"]
    with c1:
        st.subheader("🔵 인구 늘어난 곳 10")
        st.dataframe(change_merged.nlargest(10, "증감률")[ccols].reset_index(drop=True))
    with c2:
        st.subheader("🔴 인구 줄어든 곳 10")
        st.dataframe(change_merged.nsmallest(10, "증감률")[ccols].reset_index(drop=True))

# ---------------- 2. 전국 총인구 추이 ----------------
st.subheader("② 전국 총인구 추이")

national_pop = yearly_pop.groupby("연도")["전체인구"].sum().reset_index()
fig_national = px.line(
    national_pop, x="연도", y="전체인구", markers=True, title="전국 총인구 추이"
)
st.plotly_chart(fig_national, width="stretch")

# ---------------- 3. 미래 인구수 예측 (CAGR 기반) ----------------
st.subheader("③ 미래 인구수 예측")
st.caption(
    "시군구별 최근 연평균 증감률(CAGR)을 앞으로도 유지한다고 가정한 단순 추세 예측입니다. "
    "실제로는 출산율·이동 인구 등 다양한 변수에 따라 달라질 수 있습니다."
)

target_year_pop = st.slider(
    "예측할 연도 선택",
    min_value=last_year + 1,
    max_value=last_year + 30,
    value=last_year + 10,
    key="pop_forecast_year",
)


def cagr_forecast(group: pd.DataFrame, year: int) -> float:
    """시군구별 (첫 연도 인구, 마지막 연도 인구)로 연평균증감률을 구해 미래 인구를 추정"""
    g = group.sort_values("연도")
    y0, y1 = g["연도"].iloc[0], g["연도"].iloc[-1]
    p0, p1 = g["전체인구"].iloc[0], g["전체인구"].iloc[-1]
    n_years = y1 - y0
    if n_years <= 0 or p0 <= 0:
        return float(p1)
    cagr = (p1 / p0) ** (1 / n_years) - 1
    forecast = p1 * (1 + cagr) ** (year - y1)
    return max(float(forecast), 0)  # 인구는 음수가 될 수 없음


pop_forecast_rows = [
    {"시군구코드": code, "예측인구": round(cagr_forecast(g, target_year_pop))}
    for code, g in yearly_pop.groupby("시군구코드")
]
pop_forecast_df = pd.DataFrame(pop_forecast_rows)
pop_forecast_merged = pop_forecast_df.merge(names, on="시군구코드", how="left")

# 최신 실측 인구 대비 증감률도 같이 표시
latest_pop = wide[last_year].reset_index()
latest_pop.columns = ["시군구코드", "최신인구"]
pop_forecast_merged = pop_forecast_merged.merge(latest_pop, on="시군구코드", how="left")
pop_forecast_merged["예측증감률"] = (
    (pop_forecast_merged["예측인구"] / pop_forecast_merged["최신인구"] - 1) * 100
).round(2)

fig_pop_forecast = px.choropleth(
    pop_forecast_merged,
    geojson=geojson,
    locations="시군구코드",
    featureidkey="properties.코드",
    color="예측증감률",
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    hover_name="시군구",
    hover_data={"예측인구": True, "예측증감률": True, "시도": True, "시군구코드": False},
    labels={"예측증감률": f"{last_year}→{target_year_pop} 예측 증감률(%)"},
)
fig_pop_forecast.update_geos(fitbounds="locations", visible=False)
fig_pop_forecast.update_layout(
    margin=dict(l=0, r=0, t=10, b=0),
    height=700,
    coloraxis_colorbar_title="예측 증감률(%)",
)
st.plotly_chart(fig_pop_forecast, width="stretch")

c3, c4 = st.columns(2)
pcols = ["시도", "시군구", "최신인구", "예측인구", "예측증감률"]
with c3:
    st.subheader(f"🔵 {target_year_pop}년 인구 증가 예상 10")
    st.dataframe(pop_forecast_merged.nlargest(10, "예측증감률")[pcols].reset_index(drop=True))
with c4:
    st.subheader(f"🔴 {target_year_pop}년 인구 감소 예상 10")
    st.dataframe(pop_forecast_merged.nsmallest(10, "예측증감률")[pcols].reset_index(drop=True))
