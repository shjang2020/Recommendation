# 📘 Project

실제 추천 시스템 프로젝트를 진행하며 정리하는 공간입니다.

## 📚 프로젝트 목록

| 제목 | 링크 | 설명 |
|:---:|:---:|:---:|
| FastAPI and Docker practice with DistilBert | [링크](./FastAPI%20and%20Docker%20practice) | DistilBERT 기반 영화 추천 시스템 구현 및 FastAPI, Docker를 활용한 배포 |

## 🎬 현재 구현 프로젝트: DistilBERT 기반 영화 추천 시스템

### 프로젝트 개요
- **목적**: 실시간 영화 추천 API 서비스 구현 및 배포
- **기술 스택**: 
  - Backend: FastAPI
  - ML: DistilBERT, PyTorch
  - Infrastructure: Docker, Redis
  - Data: MovieLens, TMDB API

### 주요 기능
1. **영화 추천**
   - 장르 기반 추천
   - 키워드 기반 추천
   - 연도 범위 지정
   - 추천 수 조절 (1-10개)

2. **성능 최적화**
   - Redis 캐싱
   - DistilBERT 모델 경량화
   - 비동기 처리

3. **운영 기능**
   - API 문서화 (Swagger UI)
   - 로깅 시스템
   - 에러 처리
   - 모니터링

### 시스템 아키텍처
```
[Client] <-> [FastAPI Server] <-> [Redis Cache]
                    |
                    v
            [DistilBERT Model]
                    |
                    v
            [MovieLens Data]
```

### 배포 환경
- Docker 컨테이너화
- Redis 캐시 서버
- 환경 변수 관리
- 볼륨 마운트

## 🚀 향후 진행할 프로젝트

### 1. 실시간 추천 시스템
- Apache Kafka를 활용한 실시간 데이터 처리
- 증분 학습 구현
- 실시간 성능 모니터링

### 2. 대규모 추천 시스템
- 분산 처리 구현 (PySpark)
- 마이크로서비스 아키텍처
- 수평적 확장성 확보

### 3. 개인화 추천 시스템
- 사용자 피드백 반영
- A/B 테스트 프레임워크
- 동적 가중치 조정

### 4. 멀티모달 추천
- 이미지 기반 추천
- 음성 기반 추천
- 하이브리드 모달 추천
