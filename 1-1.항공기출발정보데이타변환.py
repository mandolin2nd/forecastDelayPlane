#########################
# Input데이타 
#      24년도의 항공기출발도착 정보 
#      데이타 출처 (에어포탈 : https://www.airportal.go.kr/airport/aircraftInfo.do)
# Output데이타
#      아래 조건에 따라 필터링 처리한 후 새로운 csv (항공기출발정보2024.csv)생성
#      - 지연, 취소 사례가 "기상에 의한 지연"이 아닌 경우는 항목 제외 
#      - 일자, 계획시간(일정에 등록된 출발 예쩡 시간), 상태를 제외한 "출발/도착","공항명,"항공사","편명"등의 불필요한 항목 제외
#      - 일자, 계획시간, 상태에 데이타가 하나라도 없으면 항목 제거
#      - 일자, 계획시간, 상태 데이타가 "" ㄸ는 ":"로 들어있는 경우도 항목 제거"
#      - 일자(20241201), 계획시간(02:35) 데이타를 가공해서 일관된 형태인 "비교시간" (2024.12.01 02) 필드를 생성하여 추가 (이후 기술할 날씨 데이타는 "시간"단위 데이타여서 '분'이하는 제외처리)
#########################
# 
# #### 사전준비
# pip install pandas
# pip install openpyxl
#
# #### 실행방법
# python 1-1.항공기출발정보데이타변환.py


import pandas as pd
import os

# 입력 폴더와 출력 파일 경로
INPUT_FOLDER = "./source_data/"  # 엑셀 파일이 있는 폴더
OUTPUT_FOLDER = "./middle_data/"
OUTPUT_FILE = "항공기출발정보2024.csv"

# 필터링 함수
def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    조건:
    - 상태 != "출발" 이고 지연원인 != "기상에 의한 지연"인 경우는 제외
    - '일자', '계획시간', '상태' 중 빈값("") 또는 ":"만 있는 행 제외
    """
    # 제외 조건
    exclude_mask = (df["상태"] != "출발") & (~df["지연원인"].str.contains("기상에 의한", na=False))
    df = df[~exclude_mask]

    # 필요한 컬럼만 선택
    df = df[["일자", "계획시간", "상태"]]

    # NaN 제거
    df = df.dropna(subset=["일자", "계획시간", "상태"])

    # 빈 문자열 또는 ":"만 있는 값 제거
    mask = df.apply(lambda col: col.map(lambda x: str(x).strip() not in ["", ":"])).all(axis=1)
    df = df[mask]

    # 일자와 계획시간을 합치고 분을 제거한후 다시 정리 (2024.01.01 02와 같은 식으로 표현되도록 )
    combo = df["일자"].astype(str) + " " + df["계획시간"].astype(str)
    dt = pd.to_datetime(combo, format="%Y%m%d %H:%M", errors="coerce")
    df["비교시간"] = dt.dt.strftime("%Y.%m.%d %H")
    
    ## 비교시간이 맨 앞의 칼럼으로 가도록 조정
    if "비교시간" in df.columns:
        df.insert(0, "비교시간", df.pop("비교시간"))

     # 비교시간이 NaN 이면 제거
    df = df.dropna(subset=["비교시간"])

    return df


#### 프로그램 시작부
merged_df = pd.DataFrame()

for month in range(1, 13):
    filename = f"항공기출도착2024{month:02}.xlsx"
    file_path = os.path.join(INPUT_FOLDER, filename)

    try:
        df = pd.read_excel(file_path, dtype=str)
    except FileNotFoundError:
        print(f"⚠️ 파일 없음: {filename}")
        continue

    original_count = len(df)
    filtered_df = filter_data(df)

    after_count = len(filtered_df)
    excluded_count = original_count - after_count

    print(f"📅 {month:02}월: 원본 {original_count}건 → 변환 {after_count}건, 제외 {excluded_count}건")

    merged_df = pd.concat([merged_df, filtered_df], ignore_index=True)

output_file = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)
merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n✅ 완료: {output_file} 생성됨, 총 {len(merged_df)}행")
