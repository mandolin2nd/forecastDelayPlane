# -*- coding: utf-8 -*-
import os
import json
import platform
import requests
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from joblib import load

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

# .env 파일이 있다면 환경 변수를 로드합니다.
# 이 코드는 로컬 개발 환경에서 .env 파일을 사용하기 위함이며,
# Azure App Service와 같은 배포 환경에서는 서비스의 환경 변수 설정을 직접 사용합니다.
load_dotenv()

# =========================
# 0) LLM (Azure OpenAI: gpt-4o-mini) 설정 - 환경 변수에서 로드
# =========================
# 보안을 위해 민감한 정보는 코드에 직접 작성하지 않고 환경 변수에서 불러옵니다.
#
# [로컬 개발]
# 프로젝트 루트에 .env 파일을 만들고 아래 내용을 추가하세요. (이 파일은 .gitignore에 포함되어야 합니다)
# AOAI_ENDPOINT="YOUR_AZURE_OPENAI_ENDPOINT"
# AOAI_API_KEY="YOUR_AZURE_OPENAI_KEY"
AOAI_ENDPOINT    = os.getenv("AOAI_ENDPOINT")
AOAI_DEPLOYMENT  = os.getenv("AOAI_DEPLOYMENT")
AOAI_API_KEY     = os.getenv("AOAI_API_KEY")
AOAI_API_VERSION = os.getenv("AOAI_API_VERSION")

# =========================
# 1) Matplotlib 한글 폰트
# =========================
try:
    font_manager.fontManager.addfont("fonts/NanumGothic-Regular.ttf")
    font_manager.fontManager.addfont("fonts/NanumGothic-Bold.ttf")
    rcParams["font.family"] = "NanumGothic"
except Exception:
    sysname = platform.system()
    if sysname == "Darwin":
        rcParams["font.family"] = "AppleGothic"
    elif sysname == "Windows":
        rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False


# =========================
# 2) 기본 설정
# =========================
st.set_page_config(page_title="항공편 예측 (서로 다른 1h/2h/3h 입력) + gpt-4o-mini", layout="centered")

FEATURE_COLS = ["풍속(m/s)", "풍향(deg)", "시정(m)", "강수량(mm)", "순간풍속(m/s)"]
DEFAULT_MODEL = "./model/flight.joblib"


# =========================
# 3) 유틸 함수(큰 기능 단위)
# =========================
@st.cache_resource
def load_model(path: str):
    """학습된 scikit-learn 파이프라인(.joblib) 로드"""
    return load(path)

def predict_one(pipe, row: list[float]) -> dict:
    """단일 입력에 대해 예측 확률/레이블 반환"""
    X = pd.DataFrame([row], columns=FEATURE_COLS)
    proba = pipe.predict_proba(X)[0]
    label = pipe.predict(X)[0]
    return {
        "label": str(label),
        "proba": {cls: float(p) for cls, p in zip(pipe.classes_, proba)}
    }

def build_prompt_json(r1: dict, r2: dict, r3: dict) -> dict:
    """gpt-4o-mini에 넘길 메시지 바디 구성(간결 JSON + 지시문)"""
    data = {
        "horizons": [
            {"name": "1h", "label": r1["label"], "proba": r1["proba"]},
            {"name": "2h", "label": r2["label"], "proba": r2["proba"]},
            {"name": "3h", "label": r3["label"], "proba": r3["proba"]},
        ],
        "labels_desc": {"출발":"정상 출발 가능성", "지연":"지연 가능성", "취소":"취소 가능성"}
    }
    system_msg = (
        "너는 항공 지연 예측 결과를 간결하게 해석하는 보조 멘토다. "
        "원칙: 1) 확률은 %단위, 소수점 둘째 자리. 2) 과도한 단정 금지. "
        "3) 1h→2h→3h 순서 비교 후, 친절한 말투의 한줄 권고." 
        "4) 항공기 출발이라는 표현을 한줄 권고에 명확하게 주어로 표시"
        "5) 즐거운 여행이 되길 기원하는 한줄 추가"
    )
    user_msg = "다음 예측 결과를 한국어로 요약해줘.\n" + f"DATA={json.dumps(data, ensure_ascii=False)}"
    return {
        "temperature": 0.3,
        "max_tokens": 400,
        "top_p": 1.0,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    }

def call_chat_completions(body: dict) -> dict:
    """Azure OpenAI Chat Completions 호출(상수 설정 사용)"""
    if not AOAI_ENDPOINT or not AOAI_DEPLOYMENT or not AOAI_API_KEY:
        raise RuntimeError("LLM 설정이 비어 있습니다. AOAI_ENDPOINT / AOAI_DEPLOYMENT / AOAI_API_KEY 확인 필요")

    url = f"{AOAI_ENDPOINT}/openai/deployments/{AOAI_DEPLOYMENT}/chat/completions?api-version={AOAI_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AOAI_API_KEY}
    resp = requests.post(url, headers=headers, json=body, timeout=40)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:800]}") from e
    return resp.json()

def to_table_row(name: str, r: dict) -> dict:
    """표 출력을 위한 라인 변환"""
    p = r["proba"]
    return {
        "구간": name, "예측": r["label"],
        "출발(%)": round(p.get("출발", 0.0)*100, 2),
        "지연(%)": round(p.get("지연", 0.0)*100, 2),
        "취소(%)": round(p.get("취소", 0.0)*100, 2),
    }


# =========================
# 4) 모델 로드
# =========================
if not os.path.exists(DEFAULT_MODEL):
    st.error(f"모델 파일을 찾을 수 없습니다: {DEFAULT_MODEL}")
    st.stop()
pipe = load_model(DEFAULT_MODEL)


# =========================
# 5) 입력 폼 (서로 다른 1h/2h/3h 입력)
# =========================
st.title("✈️ 항공편 상태 예측 (서로 다른 1h/2h/3h 입력) + gpt-4o-mini")
st.markdown("**1시간/2시간/3시간에 대해 서로 다른 입력을 지정하고, 결과를 표와 한 장 차트로 확인합니다.**")

tabs = st.tabs(["1시간 입력", "2시간 입력", "3시간 입력"])

with tabs[0]:
    st.subheader("1시간 입력")
    c1, c2, c3 = st.columns(3)
    with c1: wind_speed_1h = st.number_input("풍속(m/s)_1h", value=3.0, key="ws1")
    with c2: wind_dir_1h   = st.number_input("풍향(deg)_1h", value=180, key="wd1")
    with c3: vis_1h        = st.number_input("시정(m)_1h", value=2000, key="vi1")
    c4, c5 = st.columns(2)
    with c4: rain_1h = st.number_input("강수량(mm)_1h", value=0.0, key="rn1")
    with c5: gust_1h = st.number_input("순간풍속(m/s)_1h", value=4.0, key="gs1")

with tabs[1]:
    st.subheader("2시간 입력")
    c1, c2, c3 = st.columns(3)
    with c1: wind_speed_2h = st.number_input("풍속(m/s)_2h", value=3.0, key="ws2")
    with c2: wind_dir_2h   = st.number_input("풍향(deg)_2h", value=180, key="wd2")
    with c3: vis_2h        = st.number_input("시정(m)_2h", value=2000, key="vi2")
    c4, c5 = st.columns(2)
    with c4: rain_2h = st.number_input("강수량(mm)_2h", value=0.0, key="rn2")
    with c5: gust_2h = st.number_input("순간풍속(m/s)_2h", value=4.0, key="gs2")

with tabs[2]:
    st.subheader("3시간 입력")
    c1, c2, c3 = st.columns(3)
    with c1: wind_speed_3h = st.number_input("풍속(m/s)_3h", value=3.0, key="ws3")
    with c2: wind_dir_3h   = st.number_input("풍향(deg)_3h", value=180, key="wd3")
    with c3: vis_3h        = st.number_input("시정(m)_3h", value=2000, key="vi3")
    c4, c5 = st.columns(2)
    with c4: rain_3h = st.number_input("강수량(mm)_3h", value=0.0, key="rn3")
    with c5: gust_3h = st.number_input("순간풍속(m/s)_3h", value=4.0, key="gs3")

st.divider()


# =========================
# 6) 예측(1h/2h/3h) → 표 + 한 장 차트
# =========================
row_1h = [wind_speed_1h, wind_dir_1h, vis_1h, rain_1h, gust_1h]
row_2h = [wind_speed_2h, wind_dir_2h, vis_2h, rain_2h, gust_2h]
row_3h = [wind_speed_3h, wind_dir_3h, vis_3h, rain_3h, gust_3h]

res_1h = predict_one(pipe, row_1h)
res_2h = predict_one(pipe, row_2h)
res_3h = predict_one(pipe, row_3h)

# 표
table_df = pd.DataFrame([
    to_table_row("1시간", res_1h),
    to_table_row("2시간", res_2h),
    to_table_row("3시간", res_3h),
])
st.subheader("예측 요약표")
st.dataframe(table_df, use_container_width=True)

# 한 장 차트(그룹 막대)
labels = ["출발", "지연", "취소"]
vals_1h = [table_df.loc[0, f"{lb}(%)"] for lb in labels]
vals_2h = [table_df.loc[1, f"{lb}(%)"] for lb in labels]
vals_3h = [table_df.loc[2, f"{lb}(%)"] for lb in labels]

x = np.arange(len(labels))
width = 0.25

st.subheader("1h/2h/3h 확률 비교 (한 장 차트)")
fig, ax = plt.subplots()
bars1 = ax.bar(x - width, vals_1h, width, label="1h")
bars2 = ax.bar(x,         vals_2h, width, label="2h")
bars3 = ax.bar(x + width, vals_3h, width, label="3h")

ax.set_xticks(x, labels)
ax.set_ylabel("확률(%)")
ax.set_title("출발/지연/취소별 1h·2h·3h 확률 비교")
ax.legend()

for bars in (bars1, bars2, bars3):
    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2, h+0.5, f"{h:.2f}", ha="center", va="bottom", fontweight="bold")

st.pyplot(fig)

st.divider()


# =========================
# 7) AI로 예측 조언 (버튼 클릭 시 즉시 LLM 호출)
# =========================
st.subheader("LLM 설명 요청 (gpt-4o-mini)")
st.caption("표/차트의 예측 결과를 gpt-4o-mini에 전달하여 간단 요약/권고를 받습니다.")

if st.button("AI로 예측 조언"):
    try:
        body = build_prompt_json(res_1h, res_2h, res_3h)
        data = call_chat_completions(body)
        msg = (data.get("choices", [{}])[0].get("message", {}).get("content", "")) or ""
        if not msg:
            st.warning("LLM 응답이 비어 있습니다. 원문(JSON)을 아래에 표시합니다.")
            st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")
        else:
            st.success("AI 조언")
            st.write(msg)
    except Exception as e:
        st.error(f"LLM 호출 실패: {e}")
        st.info("AOAI_ENDPOINT / AOAI_DEPLOYMENT / AOAI_API_KEY / AOAI_API_VERSION 값을 확인하세요.")


# =========================
# 8) 디버그 (선택)
# =========================
with st.expander("디버그 정보"):
    st.write("Model path:", DEFAULT_MODEL)
    st.write("Classes:", list(getattr(pipe, "classes_", [])))
    st.write("AOAI Endpoint:", AOAI_ENDPOINT or "(빈 값)")
    st.write("AOAI Deployment:", AOAI_DEPLOYMENT or "(빈 값)")
    st.write("AOAI API key 설정 여부:", "O" if bool(AOAI_API_KEY) else "X")
    st.write("AOAI API version:", AOAI_API_VERSION)
