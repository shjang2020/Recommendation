# 📕 Practice

추천 시스템 관련 코드 구현과 실험을 정리하는 공간입니다.

## 📚 구현 목록

| 제목 | 링크 |
|:---:|:---:|
| Finding Users Who Act Alike : Transfer Learning for Expanding Advertiser Audiences | [링크](./Transfer%20Learning%20for%20Expanding%20Advertiser%20Audiences) |

## 🚀 구현할 것들

### 1. 기본 추천 알고리즘 구현
- **협업 필터링 (Collaborative Filtering)**
  - 사용자 기반 협업 필터링
  - 아이템 기반 협업 필터링
  - 행렬 분해 (Matrix Factorization)
  - SVD (Singular Value Decomposition)

- **콘텐츠 기반 필터링 (Content-based Filtering)**
  - TF-IDF를 이용한 콘텐츠 분석
  - 코사인 유사도 계산
  - 장르 기반 추천

### 2. 하이브리드 추천 시스템
- **LightFM 활용**
  - 사용자-아이템 상호작용 데이터 처리
  - 사용자/아이템 특성 활용
  - 하이브리드 모델 학습 및 평가

- **커스텀 하이브리드 모델**
  - 협업 필터링 + 콘텐츠 기반 필터링
  - 가중치 조정 실험
  - 앙상블 방법론 적용

### 3. 딥러닝 기반 추천
- **Neural Collaborative Filtering**
  - PyTorch 구현
  - 다양한 손실 함수 실험
  - 하이퍼파라미터 튜닝

- **BERT 기반 추천**
  - 텍스트 데이터 전처리
  - BERT 임베딩 생성
  - 유사도 기반 추천

### 4. 실시간 추천 시스템
- **스트리밍 데이터 처리**
  - Apache Kafka 연동
  - 실시간 데이터 처리 파이프라인
  - 증분 학습 구현

- **캐싱 전략**
  - Redis를 이용한 추천 결과 캐싱
  - 캐시 무효화 전략
  - 성능 최적화

### 5. 평가 및 최적화
- **다양한 평가 지표 구현**
  - Precision@K, Recall@K
  - MAP, NDCG
  - 다양성, 신규성 지표

- **A/B 테스트 프레임워크**
  - 실험 설계
  - 통계적 유의성 검정
  - 결과 분석 및 시각화

### 6. 대규모 데이터 처리
- **분산 처리 구현**
  - PySpark를 이용한 데이터 처리
  - 분산 학습 구현
  - 성능 벤치마킹

- **효율적인 데이터 구조**
  - 희소 행렬 최적화
  - 메모리 사용량 최적화
  - 처리 속도 개선

## 📊 평가 지표

일반적으로 사용되는 추천 시스템 평가 지표:

1. **정확도 기반**
   - Precision@K
   - Recall@K
   - MAP (Mean Average Precision)
   - NDCG (Normalized Discounted Cumulative Gain)

2. **다양성 기반**
   - Coverage
   - Diversity
   - Novelty

3. **사용자 경험**
   - Response Time
   - User Satisfaction
   - Click-through Rate