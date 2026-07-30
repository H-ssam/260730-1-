import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="전국 고령화 지도", layout="wide")
st.title("🗺️ 전국 고령화 지도")
st.caption("시군구별 65세 이상 인구 비율 (행정안전부 주민등록 인구)")

POP_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/population_yearly.csv.gz"
GEO_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/boundaries/sigungu_kr.geojson"

BASE_YEAR = 2026  # 전국 총인구 비교 기준 연도


# ============================================================
# 데이터 로딩 (모두 캐싱 — 앱이 다시 그려져도 재요청/재계산 없음)
# ============================================================

@st.cache_data(show_spinner="인구 데이터를 불러오는 중입니다...")
def load_population():
    # '코드' 열은 앞자리 0이 사라지지 않게 글자로 읽습니다
    return pd.read_csv(POP_URL, dtype={"코드": str})


@st.cache_data(show_spinner="지도 경계를 불러오는 중입니다...")
def load_geojson():
    return requests.get(GEO_URL, timeout=30).json()


def age_of(col: str):
    m = re.match(r"계_(\d+)세", col)
    return int(m.group(1)) if m else None


@st.cache_data(show_spinner="연도별 인구 통계를 계산하는 중입니다...")
def build_yearly_stats(df_raw: pd.DataFrame) -> pd.DataFrame:
    """시군구·연도별 전체인구/고령인구/고령화율을 한 번에 계산.
    (예전에는 고령화율용, 총인구용 함수가 따로 있어 같은 연산을 두 번 했습니다 — 여기서 통합)"""
    d = df_raw.copy()
    total_cols = [c for c in d.columns if c.startswith("계_")]  # 남_·여_ 까지 더하면 두 배가 되므로 계_만 사용
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


@st.cache_data(show_spinner="지도 숫자 라벨 위치를 계산하는 중입니다...")
def build_centroids(geojson_data: dict) -> pd.DataFrame:
    """geojson 폴리곤들의 중심좌표를 계산합니다 (shapely 없이 shoelace 공식 사용)."""

    def polygon_centroid_area(coords):
        x = [c[0] for c in coords]
        y = [c[1] for c in coords]
        if (x[0], y[0]) != (x[-1], y[-1]):
            x.append(x[0])
            y.append(y[0])
        n = len(x)
        A = Cx = Cy = 0.0
        for i in range(n - 1):
            cross = x[i] * y[i + 1] - x[i + 1] * y[i]
            A += cross
            Cx += (x[i] + x[i + 1]) * cross
            Cy += (y[i] + y[i + 1]) * cross
        A *= 0.5
        if abs(A) < 1e-12:
            return sum(x[:-1]) / (n - 1), sum(y[:-1]) / (n - 1), 0.0
        return Cx / (6 * A), Cy / (6 * A), abs(A)

    def geometry_centroid(geometry):
        gtype = geometry.get("type")
        if gtype == "Polygon":
            cx, cy, _ = polygon_centroid_area(geometry["coordinates"][0])
            return cx, cy
        if gtype == "MultiPolygon":
            total_area = wx = wy = 0.0
            for poly in geometry["coordinates"]:
                cx, cy, area = polygon_centroid_area(poly[0])
                wx += cx * area
                wy += cy * area
                total_area += area
            if total_area == 0:
                cx, cy, _ = polygon_centroid_area(geometry["coordinates"][0][0])
                return cx, cy
            return wx / total_area, wy / total_area
        return None, None

    rows = []
    for f in geojson_data["features"]:
        lon, lat = geometry_centroid(f["geometry"])
        rows.append({"시군구코드": str(f["properties"]["코드"]), "lon": lon, "lat": lat})
    return pd.DataFrame(rows)


def add_value_labels(fig, centroids_df: pd.DataFrame, data_df: pd.DataFrame, value_col: str, fmt: str = "{:.1f}"):
    """지도 위, 각 시군구 중심 좌표에 숫자를 텍스트로 얹습니다."""
    labeled = data_df.merge(centroids_df, on="시군구코드", how="left").dropna(subset=["lon", "lat"])
    fig.add_trace(
        go.Scattergeo(
            lon=labeled["lon"],
            lat=labeled["lat"],
            text=labeled[value_col].map(lambda v: fmt.format(v)),
            mode="text",
            textfont=dict(size=7, color="black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return fig


# ============================================================
# 공용 데이터 준비
# ============================================================

show_labels = st.checkbox("지도에 숫자 라벨 표시", value=True, help="지역이 많아 축소된 화면에서는 라벨이 겹칠 수 있습니다. 필요하면 꺼주세요.")

df_full = load_population()
geojson = load_geojson()
yearly = build_yearly_stats(df_full)
centroids = build_centroids(geojson)

names = pd.DataFrame([
    {
        "시군구코드": str(f["properties"]["코드"]),
        "시군구": f["properties"]["시군구"],
        "시도": f["properties"]["시도"],
    }
    for f in geojson["features"]
])

available_years = sorted(yearly["연도"].unique())
latest_year = available_years[-1]

BINS = [0, 19, 23, 28, 38, 100]
LABELS = ["19% 미만", "19~23%", "23~28%", "28~38%", "38% 이상"]
COLORS = {
    "19% 미만": "#fee6ce",
    "19~23%": "#fdc086",
    "23~28%": "#f79646",
    "28~38%": "#e8590c",
    "38% 이상": "#a63603",
}

# ============================================================
# 1. 현재 고령화율 지도
# ============================================================

merged = yearly[yearly["연도"] == latest_year].merge(names, on="시군구코드", how="left")
merged["단계"] = pd.cut(merged["고령화율"], bins=BINS, labels=LABELS, right=False)

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
if show_labels:
    add_value_labels(fig, centroids, merged, "고령화율", fmt="{:.1f}%")
st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
cols = ["시도", "시군구", "고령화율"]
with c1:
    st.subheader("🔴 고령화율 높은 곳 10")
    st.dataframe(merged.nlargest(10, "고령화율")[cols].reset_index(drop=True))
with c2:
    st.subheader("🟢 고령화율 낮은 곳 10")
    st.dataframe(merged.nsmallest(10, "고령화율")[cols].reset_index(drop=True))


# ============================================================
# 2. 🔮 미래 고령화율 예측
# ============================================================

st.markdown("---")
st.header("🔮 미래 고령화율 예측")
st.caption(
    "최근 연도별 추세를 시군구 단위로 선형 회귀하여 연장한 단순 예측입니다. "
    "실제 통계청 장래인구추계처럼 출생·사망·이동을 반영한 것이 아니므로 참고용으로만 봐주세요."
)

target_year = st.slider(
    "예측할 연도 선택 (고령화율)",
    min_value=latest_year + 1,
    max_value=latest_year + 30,
    value=latest_year + 10,
)


def linear_forecast(group: pd.DataFrame, year: int) -> float:
    """시군구별 (연도, 고령화율) 선형회귀로 특정 연도 값을 예측"""
    if len(group) < 2:
        return float(group["고령화율"].iloc[-1])
    slope, intercept = np.polyfit(group["연도"], group["고령화율"], 1)
    pred = slope * year + intercept
    return float(np.clip(pred, 0, 100))


forecast_rows = [
    {"시군구코드": code, "예측고령화율": round(linear_forecast(g, target_year), 2)}
    for code, g in yearly.groupby("시군구코드")
]
forecast_merged = pd.DataFrame(forecast_rows).merge(names, on="시군구코드", how="left")
forecast_merged["단계"] = pd.cut(forecast_merged["예측고령화율"], bins=BINS, labels=LABELS, right=False)

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
if show_labels:
    add_value_labels(fig_forecast, centroids, forecast_merged, "예측고령화율", fmt="{:.1f}%")
st.plotly_chart(fig_forecast, width="stretch")

# 전국 평균 추이 + 예측 라인 차트
national = yearly.groupby("연도")[["전체인구", "고령인구"]].sum().reset_index()
national["고령화율"] = (national["고령인구"] / national["전체인구"] * 100).round(2)

nat_slope, nat_intercept = np.polyfit(national["연도"], national["고령화율"], 1)
future_years = list(range(latest_year + 1, target_year + 1))
future_rates = [round(float(np.clip(nat_slope * y + nat_intercept, 0, 100)), 2) for y in future_years]

trend_df = pd.concat([
    pd.DataFrame({"연도": national["연도"], "고령화율": national["고령화율"], "구분": "실측"}),
    pd.DataFrame({"연도": future_years, "고령화율": future_rates, "구분": "예측"}),
])
fig_trend = px.line(trend_df, x="연도", y="고령화율", color="구분", markers=True, title="전국 평균 고령화율 추이 및 예측")
st.plotly_chart(fig_trend, width="stretch")

c3, c4 = st.columns(2)
fcols = ["시도", "시군구", "예측고령화율"]
with c3:
    st.subheader(f"🔴 {target_year}년 예측 고령화율 높은 곳 10")
    st.dataframe(forecast_merged.nlargest(10, "예측고령화율")[fcols].reset_index(drop=True))
with c4:
    st.subheader(f"🟢 {target_year}년 예측 고령화율 낮은 곳 10")
    st.dataframe(forecast_merged.nsmallest(10, "예측고령화율")[fcols].reset_index(drop=True))


# ============================================================
# 3. 📊 지역별 전체 인구수 변화
# ============================================================

st.markdown("---")
st.header("📊 지역별 전체 인구수 변화")
st.caption("고령 인구뿐 아니라 시군구 전체 인구가 어떻게 늘거나 줄었는지, 앞으로 어떻게 될지 살펴봅니다.")

wide = yearly.pivot(index="시군구코드", columns="연도", values="전체인구")

st.subheader("① 과거 인구 증감률")
col_a, col_b = st.columns(2)
with col_a:
    start_year = st.selectbox("기준 연도", available_years, index=0)
with col_b:
    end_year = st.selectbox("비교 연도", available_years, index=len(available_years) - 1)

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
        color_continuous_midpoint=0,
        hover_name="시군구",
        hover_data={"증감률": True, "증감인구": True, "시도": True, "시군구코드": False},
        labels={"증감률": f"{start_year}→{end_year} 인구 증감률(%)"},
    )
    fig_change.update_geos(fitbounds="locations", visible=False)
    fig_change.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=700, coloraxis_colorbar_title="증감률(%)")
    if show_labels:
        add_value_labels(fig_change, centroids, change_merged, "증감률", fmt="{:+.1f}%")
    st.plotly_chart(fig_change, width="stretch")

    cc1, cc2 = st.columns(2)
    ccols = ["시도", "시군구", "시작인구", "종료인구", "증감률"]
    with cc1:
        st.subheader("🔵 인구 늘어난 곳 10")
        st.dataframe(change_merged.nlargest(10, "증감률")[ccols].reset_index(drop=True))
    with cc2:
        st.subheader("🔴 인구 줄어든 곳 10")
        st.dataframe(change_merged.nsmallest(10, "증감률")[ccols].reset_index(drop=True))

st.subheader("② 전국 총인구 추이")
national_pop = yearly.groupby("연도")["전체인구"].sum().reset_index()
fig_national = px.line(national_pop, x="연도", y="전체인구", markers=True, title="전국 총인구 추이")
st.plotly_chart(fig_national, width="stretch")

st.subheader("③ 미래 인구수 예측")
st.caption(
    "시군구별 최근 연평균 증감률(CAGR)을 앞으로도 유지한다고 가정한 단순 추세 예측입니다. "
    "실제로는 출산율·이동 인구 등 다양한 변수에 따라 달라질 수 있습니다."
)

target_year_pop = st.slider(
    "예측할 연도 선택 (총인구)",
    min_value=latest_year + 1,
    max_value=latest_year + 30,
    value=latest_year + 10,
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
    return max(float(forecast), 0)


pop_forecast_rows = [
    {"시군구코드": code, "예측인구": round(cagr_forecast(g, target_year_pop))}
    for code, g in yearly.groupby("시군구코드")
]
pop_forecast_merged = pd.DataFrame(pop_forecast_rows).merge(names, on="시군구코드", how="left")

latest_pop = wide[latest_year].reset_index()
latest_pop.columns = ["시군구코드", "최신인구"]
pop_forecast_merged = pop_forecast_merged.merge(latest_pop, on="시군구코드", how="left")
pop_forecast_merged["예측증감률"] = ((pop_forecast_merged["예측인구"] / pop_forecast_merged["최신인구"] - 1) * 100).round(2)

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
    labels={"예측증감률": f"{latest_year}→{target_year_pop} 예측 증감률(%)"},
)
fig_pop_forecast.update_geos(fitbounds="locations", visible=False)
fig_pop_forecast.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=700, coloraxis_colorbar_title="예측 증감률(%)")
if show_labels:
    add_value_labels(fig_pop_forecast, centroids, pop_forecast_merged, "예측증감률", fmt="{:+.1f}%")
st.plotly_chart(fig_pop_forecast, width="stretch")

c5, c6 = st.columns(2)
pcols = ["시도", "시군구", "최신인구", "예측인구", "예측증감률"]
with c5:
    st.subheader(f"🔵 {target_year_pop}년 인구 증가 예상 10")
    st.dataframe(pop_forecast_merged.nlargest(10, "예측증감률")[pcols].reset_index(drop=True))
with c6:
    st.subheader(f"🔴 {target_year_pop}년 인구 감소 예상 10")
    st.dataframe(pop_forecast_merged.nsmallest(10, "예측증감률")[pcols].reset_index(drop=True))


# ============================================================
# 4. 🇰🇷 전국 총인구 비교 (2026년 기준)
# ============================================================

st.markdown("---")
st.header(f"🇰🇷 전국 총인구 비교 ({BASE_YEAR}년 기준)")


def national_total_for_year(year: int) -> float:
    """해당 연도의 전국 총인구. 실측 데이터가 있으면 실측값 합계, 없으면 시군구별 CAGR로 추정."""
    if year in available_years:
        return float(yearly.loc[yearly["연도"] == year, "전체인구"].sum())
    return float(sum(cagr_forecast(g, year) for _, g in yearly.groupby("시군구코드")))


baseline_total = national_total_for_year(BASE_YEAR)
target_total = national_total_for_year(target_year_pop)
growth_rate = (target_total / baseline_total - 1) * 100 if baseline_total else 0.0
baseline_note = "실측" if BASE_YEAR in available_years else "추정"

col_base, col_target = st.columns(2)
with col_base:
    st.metric(f"{BASE_YEAR}년 전국 총인구 ({baseline_note})", f"{baseline_total:,.0f}명")
with col_target:
    st.metric(
        f"{target_year_pop}년 전국 총인구 (예측)",
        f"{target_total:,.0f}명",
        delta=f"{growth_rate:+.2f}% ({BASE_YEAR}년 대비)",
    )
