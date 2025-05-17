# 📕 Practice

추천 시스템 관련 코드 구현과 실험을 정리하는 공간입니다.

## 구현 목록

| 제목 | 링크 | 상태 |
|:---:|:---:|:---:|
| (기초) 추천시스템_협업필터링 | [링크](./\(기초\)%20추천시스템_협업필터링.ipynb) | ✅ 완료 |
| (기초) 추천시스템- 아이템 기반 협업 필터링 | [링크](./\(기초\)%20추천시스템-%20아이템%20기반%20협업%20필터링.ipynb) | ✅ 완료 |
| (기초) 추천시스템_개인화 추천 | [링크](./\(기초\)%20추천시스템_개인화%20추천.ipynb) | ✅ 완료 |
| 1. 추천시스템 구현 및 성능평가(CF, MF) | [링크](./1.%20추천시스템%20구현%20및%20성능평가\(CF%2C%20MF\).ipynb) | ✅ 완료 |
| 2. LightFM 구현 | [링크](./2.%20LightFM%20구현.ipynb) | ✅ 완료 |

## 구현할 것들

### 1. 기본 추천 알고리즘 심화
- **협업 필터링 (Collaborative Filtering)**
  - SVD (Singular Value Decomposition) 구현
  - ALS (Alternating Least Squares) 구현
  - 평가 지표 구현 (Precision@K, Recall@K, NDCG)

- **콘텐츠 기반 필터링 (Content-based Filtering)**
  - TF-IDF를 이용한 콘텐츠 분석
  - 코사인 유사도 계산
  - 장르 기반 추천

### 2. 하이브리드 추천 시스템
- **LightFM 심화**
  - 다양한 손실 함수 실험
  - 하이퍼파라미터 튜닝
  - 성능 평가 및 비교

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

### 4. 평가 및 최적화
- **다양한 평가 지표 구현**
  - MAP (Mean Average Precision)
  - NDCG (Normalized Discounted Cumulative Gain)
  - 다양성, 신규성 지표

- **A/B 테스트 프레임워크**
  - 실험 설계
  - 통계적 유의성 검정
  - 결과 분석 및 시각화

### 5. 효율적인 데이터 처리
- **메모리 최적화**
  - 희소 행렬 최적화
  - 메모리 사용량 최적화
  - 처리 속도 개선

- **데이터 구조 개선**
  - 효율적인 데이터 구조 설계
  - 캐싱 전략 구현
  - 배치 처리 최적화

## 평가 지표

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