import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

st.set_page_config(page_title="박스오피스 대시보드", layout="wide")
st.title("🎬 어제의 박스오피스")

# 비밀 금고에서 인증키 꺼내기 (코드에는 키를 적지 않는다)
KOBIS_KEY = st.secrets["KOBIS_KEY"]
TMDB_KEY = st.secrets.get("TMDB_KEY")  # 없으면 None → 설명/예고편 섹션에서 안내만 표시

# 한국 시간 기준 어제 날짜를 여덟 자리로 (배포 서버 시계는 외국 기준일 수 있다)
yesterday = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(days=1)
target_dt = yesterday.strftime("%Y%m%d")
st.caption(f"조회 기준일(어제): {yesterday.strftime('%Y-%m-%d')}")

url = "https://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"
res = requests.get(url, params={"key": KOBIS_KEY, "targetDt": target_dt}, timeout=10)

if res.status_code != 200:
    st.error(f"요청이 실패했습니다 (상태코드: {res.status_code})")
    st.stop()

data = res.json()

# KOBIS는 키가 틀려도 상태코드 200을 준다. 대신 faultInfo 상자가 온다.
if "faultInfo" in data:
    st.error("인증키가 올바르지 않습니다. 금고(Secrets)의 KOBIS_KEY를 확인해 주세요.")
    st.stop()

box_list = data.get("boxOfficeResult", {}).get("dailyBoxOfficeList", [])
if not box_list:
    st.warning("그날 자료가 없습니다. 날짜를 하루 더 앞으로 옮겨 보세요.")
    st.stop()

df = pd.DataFrame(box_list)

# 글자로 온 숫자들을 진짜 숫자로 바꾸기
for col in ["rank", "audiCnt", "audiAcc", "scrnCnt", "showCnt"]:
    df[col] = pd.to_numeric(df[col])

# 1위 영화 지표 카드 세 장
top = df.sort_values("rank").iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("어제 1위", top["movieNm"])
c2.metric("어제 관객수", f"{top['audiCnt']:,}명")
c3.metric("누적 관객", f"{top['audiAcc']:,}명")

# 표를 한국어 열 이름으로 정리
table = df[["rank", "movieNm", "openDt", "audiCnt", "audiAcc", "scrnCnt"]].copy()
table.columns = ["순위", "영화명", "개봉일", "관객수", "누적관객", "스크린수"]
table = table.sort_values("순위").reset_index(drop=True)

st.subheader("📋 박스오피스 TOP 10")
st.dataframe(
    table,
    column_config={
        "스크린수": st.column_config.NumberColumn(
            "스크린수",
            help="어제 하루 동안 이 영화가 상영된 스크린(상영관) 수입니다. "
                 "영화관 한 곳이 여러 스크린을 가질 수 있어, 숫자가 클수록 더 많은 상영관에서 동시에 걸렸다는 뜻입니다.",
            format="%d",
        ),
    },
)

st.subheader("📈 관객수 상위 10편")
top10 = table.sort_values("관객수", ascending=False).head(10).sort_values("관객수")  # 그래프 아래→위로 순위대로 보이게 오름차순 정렬

fig_top10 = px.bar(
    top10,
    x="관객수",
    y="영화명",
    orientation="h",
    color="관객수",
    color_continuous_scale="Blues",
    text="관객수",
)
fig_top10.update_traces(texttemplate="%{text:,}명", textposition="outside")
fig_top10.update_layout(
    xaxis_title="관객수(명)",
    yaxis_title="",
    coloraxis_showscale=False,
    margin=dict(l=10, r=70, t=10, b=10),
    height=520,
)
st.plotly_chart(fig_top10, width="stretch")


# ============================================================
# 🎥 TOP 10 영화 설명 + 예고편 (TMDB API)
# ============================================================

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)  # 하루 동안 캐싱 (같은 영화 반복 조회 방지)
def get_tmdb_info(movie_name: str, open_dt: str):
    """TMDB에서 영화를 검색해 줄거리·포스터·예고편(YouTube) 정보를 가져옵니다."""
    if not TMDB_KEY:
        return None
    try:
        year = open_dt[:4] if open_dt else None
        search_res = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={
                "api_key": TMDB_KEY,
                "query": movie_name,
                "language": "ko-KR",
                "region": "KR",
                **({"year": year} if year else {}),
            },
            timeout=10,
        ).json()

        results = search_res.get("results", [])
        if not results:
            return None
        movie = results[0]
        movie_id = movie["id"]

        # 예고편 조회 — 한국어 예고편이 없으면 영어로 한 번 더 시도
        trailer_key = None
        for lang in ["ko-KR", "en-US"]:
            videos = requests.get(
                f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
                params={"api_key": TMDB_KEY, "language": lang},
                timeout=10,
            ).json().get("results", [])
            trailers = [v for v in videos if v.get("type") == "Trailer" and v.get("site") == "YouTube"]
            if trailers:
                official = [v for v in trailers if v.get("official")]
                trailer_key = (official[0] if official else trailers[0])["key"]
                break

        return {
            "overview": movie.get("overview") or "줄거리 정보가 없습니다.",
            "poster_url": (
                f"https://image.tmdb.org/t/p/w342{movie['poster_path']}"
                if movie.get("poster_path") else None
            ),
            "trailer_url": f"https://www.youtube.com/watch?v={trailer_key}" if trailer_key else None,
        }
    except Exception:
        return None


st.subheader("🎥 TOP 10 영화 상세 정보")

if not TMDB_KEY:
    st.info(
        "영화 설명·예고편을 보려면 TMDB API 키가 필요합니다. "
        "themoviedb.org에서 무료로 발급받아 secrets.toml에 `TMDB_KEY`로 추가해주세요."
    )
else:
    for _, row in table.iterrows():
        info = get_tmdb_info(row["영화명"], row["개봉일"])
        with st.expander(f"{row['순위']}위 · {row['영화명']}"):
            col_poster, col_detail = st.columns([1, 3])

            if info and info["poster_url"]:
                col_poster.image(info["poster_url"], width=140)

            with col_detail:
                if info:
                    st.write(info["overview"])
                else:
                    st.write("TMDB에서 이 영화 정보를 찾지 못했습니다. (제목이 조금 다르게 등록되어 있을 수 있어요)")

                if info and info["trailer_url"]:
                    st.video(info["trailer_url"])
                else:
                    query = row["영화명"].replace(" ", "+")
                    st.markdown(
                        f"[🔎 유튜브에서 '{row['영화명']}' 예고편 검색하기]"
                        f"(https://www.youtube.com/results?search_query={query}+예고편)"
                    )
