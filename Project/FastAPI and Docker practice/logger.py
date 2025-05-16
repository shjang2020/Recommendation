# 로깅 시스템:
# 1. 로그 디렉토리 설정
#    - `logs` 디렉토리 자동 생성
#    - `exist_ok=True`: 이미 디렉토리가 있어도 에러 발생하지 않음
# 2. 로거 설정 함수 (setup_logger)
#    - 로그 레벨: INFO (정보성 메시지)
#    - 로그 포맷: "시간 - 로거이름 - 로그레벨 - 메시지"
#    - 두 가지 핸들러 사용:
#      a. 콘솔 핸들러: 터미널에 출력
#      b. 파일 핸들러: 로그 파일에 저장
#        - 최대 파일 크기: 10MB
#        - 백업 파일 수: 5개
#        - 파일명: app.log, app.log.1, app.log.2, ...
# 3. 전역 로거 인스턴스
#    - 이름: "movie_recommender"
#    - 애플리케이션 전체에서 사용

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

# 로그 디렉토리 생성
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# 로거 설정
def setup_logger(name: str) -> logging.Logger:
    """로거 설정 함수"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 로그 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (최대 10MB, 최대 5개 파일)
    file_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# 애플리케이션 로거 생성
app_logger = setup_logger("movie_recommender") 