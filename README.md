
아래와 같은 과제를 진행하려고 해.
제목 : 항공기 지연, 취소 예측 프로젝트
목적 : 인천국제공항 근처에 갔을때에 로밍 안내 문자에, 현재 기상정보에 따른 항공기 이륙 지연이나 취소 가능성이 있는지를 안내하고 싶어

방법 :

(소스데이타 - 첨부한 파일)
- airplane_takeoff_2024.xlsx (2024년 항공기 출발 정보)
- weatherinfo_inchon_airport2024.csv (2024년도 인천공항 기상정보)
- weatherCode.xlsx (위 기상정보에서 사용하는 "일기현상"에 대한 설명 코드표)

(추론 방법)
- 기상정보, 항공기 출발 정보를 입력시에 아래와 같은 항목을 조정
   - 이륙 정보에서 기상으로 인한 이륙을 제외한 다른 지연,취소 사유는 배제
   - 풍속 (KT - knot)정보를 km/s 또는 m/s로 보정하여 입력 
   - 기타 데이타가 형식에 맞지 않은 경우는 배제 
-  입력데이타는 "출발 예상 날짜 및 시간"이며, 해당 시간부터 1시간 단위로 3시간까지의 기상 예보를  인천 공항 날씨 예보(https://amo.kma.go.kr/weather/airport.do?icaoCode=RKSI)에서 찾아, 기존의 기상 데이타와 비교하여 적절한 항공기 취소 또는 지연될 확률을 예측
- 해당 방법을 Azure App Service (Python사용)와 Azure AI Search를 사용하여 데모 서비스를 구축해보자고 함

(테스트 방법)
- pythone streamlit으로 구축한 페이지에서
* 입력 : 날짜, 시간 입력
* 처리 : 해당 시간으로 부터의 3시간 이내의 기상 예보(https://amo.kma.go.kr/weather/airport.do?icaoCode=RKSI)를 참고하여, 비행기 출발 지연, 취소 가능성을 예측
* 결과 : 예측 내용을, 사용자가 이해하기 쉬운 문구로 글 정리하여 노출

(요청사항)
- 해당 내용의 기획을 조금 다듬어서 기획문서로 정리
- 해당 내용을 Azure App Service (Python사용)와 Azure AI Search로 구축하기 위한 How-To를 단계별로 정리 하여 별도 문서로 제공
- 각각 인덱싱과 실제 사용을 위한 python코드를 풀소스로 정리하여 제공




-------------- 배포 -------------------

Azure App Service 배포 (West US 3)
1. App Service Plan 생성
az appservice plan create \
  -n appserviceplan-mandolin-flight-delay \
  -g rg-mandolin-0915 \
  --sku B1 \
  --is-linux \
  -l westus3
이름: appserviceplan-mandolin-flight-delay
위치: West US 3
SKU: B1 (테스트/개발용, 필요 시 S1/P1v3 등으로 업그레이드)
2. Web App 생성
az webapp create \
  -n appservice-mandolin-flight-delay \
  -g rg-mandolin-0915 \
  --plan appserviceplan-mandolin-flight-delay \
  --runtime "PYTHON:3.10" \
  -l westus3
이름: appservice-mandolin-flight-delay
런타임: Python 3.10
위치: West US 3
3. 환경 변수 설정 (Azure OpenAI 접속 정보)
코드에 포함된 민감한 정보(API 키 등)를 Azure에 안전하게 설정합니다.
아래 명령어를 터미널에서 실행하세요. `<...>` 부분은 실제 값으로 반드시 교체해야 합니다.

**참고:** Azure Portal에서 직접 설정할 수도 있습니다.
`App Service` -> `구성(Configuration)` -> `응용 프로그램 설정(Application settings)`에서 동일한 키-값 쌍을 추가할 수 있습니다.

az webapp config appsettings set \
  -g rg-mandolin-0915 \
  -n appservice-mandolin-flight-delay \
  --settings SCM_DO_BUILD_DURING_DEPLOYMENT=1 WEBSITES_PORT=8000 \
  AOAI_ENDPOINT="<YOUR_AZURE_OPENAI_ENDPOINT>" \
  AOAI_DEPLOYMENT="gpt-4o-mini" \
  AOAI_API_KEY="<YOUR_AZURE_OPENAI_KEY>" \
  AOAI_API_VERSION="2024-08-01-preview"

4. Startup Command 등록
az webapp config set \
  -g rg-mandolin-0915 \
  -n appservice-mandolin-flight-delay \
  --startup-file "python -m streamlit run predict_all.py --server.port 8000 --server.address 0.0.0.0"
5. 코드 배포 (Zip 방식 예시)
./deploy.sh
🔹 리소스 이름 정리 (West US 3)
Resource Group: rg-mandolin-0915 (기존, 지역은 그대로)
App Service Plan: appserviceplan-mandolin-flight-delay (West US 3)
Web App: appservice-mandolin-flight-delay (West US 3)
배포 후 URL:
https://appservice-mandolin-flight-delay.azurewebsites.net
👉 이렇게 하면 rg-mandolin-0915 그룹은 한국에 있고, Web App은 West US 3에 존재하게 돼요.
Azure는 리소스 그룹 안에 지역이 다른 리소스를 넣을 수 있으니 문제는 없습니다.