# Python 3.8 기반 이미지 사용
FROM python:3.8-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 및 캐시 정리
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 필요한 Python 패키지 설치를 위한 requirements.txt 복사
COPY requirements.txt .

# Python 패키지 설치 (캐시 사용하지 않음)
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip/*

# 애플리케이션 코드 복사
COPY . .

# 데이터 디렉토리 생성
RUN mkdir -p data logs

# 포트 설정
EXPOSE 8000

# 애플리케이션 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]