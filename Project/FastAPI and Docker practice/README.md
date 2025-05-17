# Movie Recommendation API with FastAPI and Docker

영화 추천 시스템을 FastAPI와 Docker를 사용하여 구현한 프로젝트입니다. DistilBERT 기반의 추천 시스템과 Redis 캐싱을 통해 효율적인 영화 추천 서비스를 제공합니다.

## 주요 기능

- 장르, 키워드 기반 영화 추천
- DistilBERT 모델을 활용한 영화 플롯 분석
- Redis를 이용한 추천 결과 캐싱
- TMDB API 연동을 통한 영화 포스터 및 배경 이미지 제공
- Docker 컨테이너화를 통한 쉬운 배포

## DistilBERT 선택 이유

이 프로젝트에서는 실시간 영화 추천을 위해 DistilBERT 모델을 선택했습니다. DistilBERT는 BERT의 경량화 버전으로, 다음과 같은 장점을 제공합니다:

- **빠른 추론 속도**: 원본 BERT 대비 40% 더 빠른 추론 속도로 실시간 추천이 가능합니다.
- **메모리 효율성**: 원본 BERT 대비 60% 더 적은 파라미터로 동일한 성능을 유지합니다.
- **실시간 처리**: 적은 리소스로도 빠른 텍스트 처리와 의미 분석이 가능합니다.
- **높은 정확도**: BERT의 97% 수준의 성능을 유지하면서도 가벼운 모델 구조를 제공합니다.
- **배포 용이성**: 작은 모델 크기로 Docker 컨테이너 배포가 용이합니다.

이러한 장점들로 인해 실무에서도 실시간 추천 시스템에 DistilBERT가 널리 사용되고 있습니다.

## 기술 스택

- **Backend**: FastAPI, Uvicorn
- **ML/DL**: PyTorch, Transformers, scikit-learn
- **Data Processing**: Pandas, NumPy
- **Caching**: Redis
- **Containerization**: Docker, Docker Compose

## 프로젝트 구조

```
.
├── app.py              # FastAPI 애플리케이션 메인 파일
├── recommender.py      # BERT 기반 추천 시스템 구현
├── data_loader.py      # MovieLens 데이터 로딩 및 전처리
├── cache.py           # Redis 캐시 관리
├── logger.py          # 로깅 설정
├── exceptions.py      # 커스텀 예외 처리
├── Dockerfile         # Docker 이미지 빌드 설정
├── docker-compose.yml # Docker 서비스 구성
├── requirements.txt   # 프로젝트 의존성
├── data/             # 데이터 저장 디렉토리
├── logs/             # 로그 파일 저장 디렉토리
└── model/            # 학습된 모델 저장 디렉토리
```

## 시작하기

### 사전 요구사항

- Docker와 Docker Compose
- TMDB API 키

### 설치 및 실행

1. 저장소 클론
```bash
git clone [repository-url]
cd "FastAPI and Docker practice"
```

2. 환경 변수 설정
`.env` 파일을 생성하고 TMDB API 키를 설정합니다:
```
TMDB_API_KEY=your_tmdb_api_key
```

3. Docker 컨테이너 실행
```bash
docker-compose up --build
```

4. API 접속
- API 문서: http://localhost:8000/docs
- API 엔드포인트: http://localhost:8000

## API 엔드포인트

### GET /
- API 기본 정보 및 사용 가능한 엔드포인트 목록 제공

### POST /recommend
- 영화 추천 요청 처리
- 요청 파라미터:
  - genres: 영화 장르 리스트 (선택)
  - keywords: 검색 키워드 (선택)
  - num_recommendations: 추천 영화 수 (1-10, 기본값: 1)
  - start_year: 시작 연도 (기본값: 1900)
  - end_year: 종료 연도 (기본값: 2020)

### DELETE /cache
- Redis 캐시 초기화

## 지원하는 영화 장르

- Action
- Adventure
- Animation
- Comedy
- Crime
- Documentary
- Drama
- Fantasy
- Horror
- Mystery
- Romance
- Science Fiction
- Thriller
- War
- Western

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 