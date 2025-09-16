#########################
# Input데이타 
#      1-1, 1-2에서 생성된 "인천공항기상데이타2024.csv", "항공기출발정보2024.csv"
# Output데이타
#      최종.merge.csv생성
#      - 항공기출발정보 데이타의 오른쪽에 인천공항기상데이타의 정보를 추가 ("비교시간"이 같은 데이타 기준) 
#########################
# 
# #### 사전준비
# pip install pandas
# pip install path
#
# #### 실행방법
# python 2.데이타병합.py
# 

import pandas as pd
from pathlib import Path


base_path = Path("./middle_data/항공기출발정보2024.csv")         # 기준 파일
wx_path   = Path("./middle_data/인천공항기상데이타2024.csv")     # 기상 파일
out_path  = Path("./middle_data/최종merge.csv")

# 1) 읽기: '비교시간'을 문자열로 강제(자동 날짜 파싱 방지)
base = pd.read_csv(base_path, dtype={"비교시간": "string"})
wx   = pd.read_csv(wx_path,   dtype={"비교시간": "string", "일시": "string"})

# 2) 공백 제거(같은 값인데 공백 때문에 병합 실패 방지)
base["비교시간"] = base["비교시간"].str.strip()
wx["비교시간"]   = wx["비교시간"].str.strip()

# 3) 컬럼명 변경 (요청사항)
base = base.rename(columns={
    "일자": " (항공기)일자".strip(),          # 안전하게 공백 제거
    "계획시간": "(항공기)계획일자"
})
wx = wx.rename(columns={
    "일시": "(날씨)일시"
})

# 4) '비교시간' (문자열) 기준으로 병합 (Left: 항공기 기준)
merged = pd.merge(base, wx, on="비교시간", how="left")

# 5) '비교시간'을 맨 앞으로
cols = ["비교시간"] + [c for c in merged.columns if c != "비교시간"]
merged = merged[cols]

# 6) 저장
out_path.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(out_path, index=False)
