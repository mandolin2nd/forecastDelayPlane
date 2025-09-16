import os
import json
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
AOAI_DEPLOYMENT  = os.getenv("AOAI_DEPLOYMENT", "gpt-4o-mini")
AOAI_API_KEY     = os.getenv("AOAI_API_KEY")
AOAI_API_VERSION = os.getenv("AOAI_API_VERSION", "2024-08-01-preview")

# --- 추가: 필수 환경 변수 확인 ---
# Azure App Service에 배포 시, 아래 변수들이 설정되지 않으면 앱이 시작되지 않고 명확한 오류를 표시합니다.
# 이렇게 하면 디버깅이 훨씬 쉬워집니다.
missing_vars = []
if not AOAI_ENDPOINT:
    missing_vars.append("AOAI_ENDPOINT")
if not AOAI_API_KEY:
    missing_vars.append("AOAI_API_KEY")

if missing_vars:
    st.error(f"오류: 필수 환경 변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
    st.info("Azure Portal > App Service > '구성' > '응용 프로그램 설정'에서 해당 변수들이 올바르게 추가되었는지 확인해주세요. 이름에 오타가 없는지 다시 한번 확인하는 것이 좋습니다.")
    st.stop()

# =========================
# 1) Matplotlib 한글 폰트
# =========================
font_manager.fontManager.addfont("fonts/NanumGothic-Regular.ttf")
font_manager.fontManager.addfont("fonts/NanumGothic-Bold.ttf")
rcParams["font.family"] = "NanumGothic"
rcParams["axes.unicode_minus"] = False


# =========================
# 2) 상수 및 모델 로드
# =========================
FEATURE_COLS = ["풍속(m/s)", "풍향(deg)", "시정(m)", "강수량(mm)", "순간풍속(m/s)"]
DEFAULT_MODEL = "./model/flight.joblib"


# =========================
# 3) 유틸리티 함수
# =========================
@st.cache_resource
def load_model(path: str):
    """학습된 scikit-learn 파이프라인(.joblib) 로드"""
    return load(path)

def build_vector_from_inputs(input_map: dict, feature_cols: list[str]) -> tuple[list | None, str | None]:
    """st.text_input 맵에서 예측용 벡터 생성. 에러 시 메시지 반환."""
    vals = []
    provided_count = 0
    for col in feature_cols:
        if col in input_map:
            raw = (input_map[col] or "").strip()
            if raw == "":
                vals.append(None)
            else:
                try:
                    vals.append(float(raw))
                    provided_count += 1
                except (ValueError, TypeError):
                    return None, f"'{col}'에 유효한 숫자를 입력하세요."
        else:
            # 선택되지 않은 변수는 결측치로 전달
            vals.append(None)
    if provided_count == 0:
        return None, "최소 1개 이상의 변수에 값을 입력해야 합니다."
    return vals, None

def predict_one(pipe, row: list[float]) -> dict:
    """단일 입력에 대해 예측 확률/레이블 반환"""
    X = pd.DataFrame([row], columns=FEATURE_COLS)
    proba = pipe.predict_proba(X)[0]
    return {
        "label": pipe.classes_[np.argmax(proba)],
        "proba": {cls: float(p) for cls, p in zip(pipe.classes_, proba)}
    }

def to_table_row(name: str, res: dict) -> dict:
    """예측 결과를 표의 한 행으로 변환"""
    row = {"구분": name, "예측": res["label"]}
    for k, v in res["proba"].items():
        row[f"{k}(%)"] = v * 100
    return row

def build_prompt_json(r1: dict, r2: dict, r3: dict, user_prompt: str) -> dict:
    """gpt-4o-mini에 넘길 메시지 바디 구성(간결 JSON + 사용자 지시문)"""
    data = {
        "horizons": [
            {"name": "1h", "label": r1["label"], "proba": r1["proba"]},
            {"name": "2h", "label": r2["label"], "proba": r2["proba"]},
            {"name": "3h", "label": r3["label"], "proba": r3["proba"]},
        ]
    }
    system_msg = (
        "당신은 항공편 이륙 지연이나, 취소 가능성을 안내하는 한국어 비서입니다."
        "1) 출발/지연/취소 확률을 고려해서 승객에게 지연이나 취소 가능성이 숫자가 아닌 표현으로 알려줘."
        "2) 과도한 확신 표현은 피하고 확률 기반의 조심스러운 어조를 유지해주세요."
        "3) 항공기 출발이라는 표현을 한줄 권고에 명확하게 주어로 표시"
        "4) 지연이나 취소 확률에 대해서는 기상때문이라는 점을 명확하게 알려줘"
        "5) 즐거운 여행이 되길 기원하는 한줄 추가"
    )
    user_msg = user_prompt + "\n\n" + "참고 데이터:\n" + f"DATA={json.dumps(data, ensure_ascii=False)}"
    return {
        "temperature": 0.3,
        "max_tokens": 400,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    }

def call_chat_completions(body: dict) -> dict:
    """Azure OpenAI Chat Completions API 호출"""
    url = f"{AOAI_ENDPOINT}/openai/deployments/{AOAI_DEPLOYMENT}/chat/completions?api-version={AOAI_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AOAI_API_KEY}
    resp = requests.post(url, headers=headers, json=body, timeout=40)
    resp.raise_for_status()
    return resp.json()


# =========================
# 4) 모델 로드 및 에러 처리
# =========================
pipe = None
try:
    if not os.path.exists(DEFAULT_MODEL):
        st.error(f"모델 파일을 찾을 수 없습니다: {DEFAULT_MODEL}")
    else:
        pipe = load_model(DEFAULT_MODEL)
except Exception as e:
    st.error(f"모델 로딩 중 오류: {e}")

if pipe is None:
    st.stop()


# =========================
# 5) UI 구성 및 예측 실행
# =========================
st.title("✈️ 항공편 상태 예측 + LLM 요약")
st.caption("공통 '입력 변수 선택' → 시간대별(1/2/3시간후) 입력 → 한 번에 예측 및 LLM 조언 요청")

# ----- 공통: 입력 변수 선택 -----
with st.container():
    st.markdown("### 🔧 입력 변수 선택 (공통)")
    selected_features = st.multiselect(
        "예측에 사용할 변수를 선택하세요.",
        options=FEATURE_COLS,
        default=FEATURE_COLS,
        help="선택된 변수만 아래 1/2/3시간후 입력 영역에 노출됩니다. 미선택/미입력 변수는 결측으로 전달되어 중앙값으로 보간됩니다."
    )
    if not selected_features:
        st.info("최소 1개 이상의 변수를 선택하세요.")
    st.markdown("---")

# ----- 입력 영역: 1/2/3시간후 -----
col1, col2, col3 = st.columns(3)

def input_section(col, horizon_label, features_to_show):
    with col:
        st.markdown(f"### ⏱ {horizon_label} 값 입력")
        st.caption("선택한 변수만 입력하세요.")
        inputs = {}
        if not features_to_show:
            st.write("위에서 변수를 선택하세요.")
        for feat in features_to_show:
            ph = {
                "풍속(m/s)": "예: 6.5", "풍향(deg)": "예: 270", "시정(m)": "예: 9000",
                "강수량(mm)": "예: 1.2", "순간풍속(m/s)": "예: 10.3",
            }.get(feat, "")
            inputs[feat] = st.text_input(f"{feat}", value="", placeholder=ph, key=f"{horizon_label}_{feat}")
    return inputs

inputs1 = input_section(col1, "1시간후", selected_features)
inputs2 = input_section(col2, "2시간후", selected_features)
inputs3 = input_section(col3, "3시간후", selected_features)

st.divider()

# ----- LLM 프롬프트 입력 (선택) -----
st.markdown("### 🧾 LLM 프롬프트")
default_prompt = (
    "다음 확률을 바탕으로 비행기 이륙 예상 안내문을 한국어로 3~4문장으로 작성해줘. "
    "출발/지연/취소 확률을 고려해서 승객에게 출발 지연이나 취소 가능성이 얼만큼인지를 알려주고, 즐거운 여행이 되라는 좋은 얘기도 한줄 남겨줘. "
    "과도한 확신 표현은 피하고 확률 기반의 조심스러운 어조를 유지해줘."
)
user_prompt = st.text_area("프롬프트", value=default_prompt, height=120, help="gpt-4o-mini에 전달할 지시사항")

# ----- 단일 통합 버튼 -----
if st.button("🚀 예측 및 AI 조언 요청", disabled=not selected_features):
    row_1h, err1 = build_vector_from_inputs(inputs1, FEATURE_COLS)
    row_2h, err2 = build_vector_from_inputs(inputs2, FEATURE_COLS)
    row_3h, err3 = build_vector_from_inputs(inputs3, FEATURE_COLS)

    has_err = any((err1, err2, err3))
    if has_err:
        for e in (err1, err2, err3):
            if e: st.warning(e)
    else:
        with st.spinner("모델 예측 및 AI 조언을 생성하는 중..."):
            res_1h = predict_one(pipe, row_1h)
            res_2h = predict_one(pipe, row_2h)
            res_3h = predict_one(pipe, row_3h)

            table_df = pd.DataFrame([
                to_table_row("1시간", res_1h), to_table_row("2시간", res_2h), to_table_row("3시간", res_3h),
            ])
            st.subheader("예측 요약표")
            st.dataframe(table_df, use_container_width=True)

            labels = ["출발", "지연", "취소"]
            vals_1h = [table_df.loc[0, f"{lb}(%)"] for lb in labels]
            vals_2h = [table_df.loc[1, f"{lb}(%)"] for lb in labels]
            vals_3h = [table_df.loc[2, f"{lb}(%)"] for lb in labels]

            x = np.arange(len(labels)); width = 0.25
            st.subheader("1h/2h/3h 확률 비교 (한 장 차트)")
            fig, ax = plt.subplots()
            bars1 = ax.bar(x - width, vals_1h, width, label="1h")
            bars2 = ax.bar(x,         vals_2h, width, label="2h")
            bars3 = ax.bar(x + width, vals_3h, width, label="3h")
            ax.set_xticks(x, labels); ax.set_ylabel("확률(%)")
            ax.set_title("출발/지연/취소별 1h·2h·3h 확률 비교"); ax.legend()
            for bars in (bars1, bars2, bars3):
                for rect in bars:
                    h = rect.get_height()
                    ax.text(rect.get_x()+rect.get_width()/2, h+0.5, f"{h:.1f}", ha="center", va="bottom", size=9)
            st.pyplot(fig)
            st.divider()

            st.subheader("LLM 설명 (gpt-4o-mini)")
            try:
                body = build_prompt_json(res_1h, res_2h, res_3h, user_prompt)
                data = call_chat_completions(body)
                msg = (data.get("choices", [{}])[0].get("message", {}).get("content", "")) or ""
                if not msg:
                    st.warning("LLM 응답이 비어 있습니다. 원문(JSON)을 아래에 표시합니다.")
                    st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")
                else:
                    st.success("✅ AI 조언")
                    st.write(msg)
            except Exception as e:
                st.error(f"LLM 호출 실패: {e}")
                st.info("AOAI_ENDPOINT / AOAI_DEPLOYMENT / AOAI_API_KEY / AOAI_API_VERSION 값을 확인하세요.")


# =========================
# 6) 푸터
# =========================
st.divider()
st.caption("🧠 모델: RandomForest(class_weight='balanced') | 전처리: SimpleImputer(median) | 폰트: NanumGothic")
