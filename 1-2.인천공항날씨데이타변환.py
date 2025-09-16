#########################
# Input데이타 
#      24년도의 인천공항날씨 데이타  
#      데이타 출처 (기상철 깅상정보개발포털 : https://data.kma.go.kr/data/air/selectAmosRltmList.do?pgmNo=575)
# Output데이타
#      아래 조건에 따라 필터링 처리한 후 새로운 csv (인천공항기상데이타2024.csv)생성
#      - "지점","지점명","일기현상" 데이타 제거 (일기현상의 경우 데이타 필드는 있으나 내용이 안보여서 제거함)
#      - "풍속","순간풍속" 데이타가 knot단위(KT)여서 m/s로 변경 저장
#      - "일시" 데이타를 "비교시간"으로 변경하여 추가 
#########################
# 
# #### 사전준비
# pip install pandas
# pip install path
#
# #### 실행방법
# python 1-2.인천공항날씨데이타변환.py
# 



import pandas as pd
from pathlib import Path

# ===== 경로 설정 =====
in_path = Path("./source_data/weathinfo_inchon_airport2024.csv")
out_path = Path("./middle_data/인천공항기상데이타2024.csv")

# ===== 1) 원본 로드 (문자열로 일단 읽기) =====
df = pd.read_csv(in_path, dtype=str)

# ===== 2) 불필요 컬럼 제거 =====
# 없으면 무시(errors="ignore")
# df = df.drop(columns=["지점", "지점명", "일기현상"], errors="ignore")
df = df.drop(columns=["지점", "지점명"], errors="ignore")

# ===== 3) '일시' -> '비교시간'(YYYY.MM.DD HH) =====
# 예: '2024.1.1 1:00' -> '2024.01.01 01'
if "일시" not in df.columns:
    raise KeyError("'일시' 컬럼이 없습니다.")
dt = pd.to_datetime(df["일시"], errors="coerce")
df["비교시간"] = dt.dt.strftime("%Y.%m.%d %H")  # Series.dt.strftime 사용!

# ===== 4) 풍속 단위 변환 (knots -> m/s) =====
# 참고: 1 knot = 0.514444 m/s
KNOT_TO_MS = 0.514444
convert_map = {
    "풍속(KT)": "풍속(m/s)",
    "순간풍속(KT)": "순간풍속(m/s)",
}
for src_col, new_col in convert_map.items():
    if src_col in df.columns:
        vals = pd.to_numeric(df[src_col], errors="coerce")
        df[new_col] = (vals * KNOT_TO_MS).round(3)

# ===== 5) 결측값 0으로 채우기 =====
# 요구사항: 비어있는 데이터는 0으로
# df = df.fillna(0)

# ===== 6) 저장 =====
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"저장 완료: {out_path}")


