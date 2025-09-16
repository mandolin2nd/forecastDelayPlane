#!/bin/bash

# 스크립트 실행 중 오류가 발생하면 즉시 중단을 위해 set -e 사용
set -e

# --- 변수 설정 ---
RESOURCE_GROUP="rg-mandolin-0915"
APP_NAME="appservice-mandolin-flight-delay"
ZIP_FILE="deploy.zip"

# 압축할 파일 및 폴더 목록
SOURCE_FILES_AND_DIRS="./fonts ./model ./predict_all.py ./requirements.txt ./.env"

echo "배포 스크립트를 시작합니다."

# --- 1. deploy.zip 생성 ---
echo "1. 기존 배포 파일($ZIP_FILE)을 삭제합니다."
rm -f "$ZIP_FILE"

echo "2. 새 배포 파일($ZIP_FILE)을 생성합니다."
zip -r "$ZIP_FILE" $SOURCE_FILES_AND_DIRS
echo "압축 완료: $ZIP_FILE"

# --- 2. Azure App Service에 배포 ---
echo "3. Azure App Service에 배포를 시작합니다..."
az webapp deployment source config-zip \
  --resource-group "$RESOURCE_GROUP" \
  --name "$APP_NAME" \
  --src "$ZIP_FILE"

echo "✅ 배포가 성공적으로 완료되었습니다."
echo "애플리케이션 URL: https://$APP_NAME.azurewebsites.net"