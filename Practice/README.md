# 📕 Practice

추천 시스템 관련 코드 구현과 실험을 정리하는 공간입니다.

## 📚 구현 목록

| 제목 | 링크 |
|:---:|:---:|
| Finding Users Who Act Alike : Transfer Learning for Expanding Advertiser Audiences | [링크](./Transfer%20Learning%20for%20Expanding%20Advertiser%20Audiences) |

## 📝 코드 구현 가이드라인

각 구현 프로젝트는 다음 구조를 따릅니다:

```
project_name/
├── README.md           # 프로젝트 설명
├── requirements.txt    # 의존성 패키지
├── data/              # 데이터셋
├── src/               # 소스 코드
│   ├── data/         # 데이터 처리
│   ├── models/       # 모델 구현
│   ├── utils/        # 유틸리티 함수
│   └── evaluation/   # 평가 코드
├── notebooks/         # 실험 노트북
└── results/          # 실험 결과
```

## 🔧 구현 시 고려사항

1. **코드 품질**
   - PEP 8 스타일 가이드 준수
   - 적절한 주석과 문서화
   - 모듈화된 구조

2. **실험 관리**
   - 실험 설정 기록
   - 결과 재현 가능성
   - 성능 지표 측정

3. **성능 최적화**
   - 메모리 사용량
   - 실행 시간
   - 확장성

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

## 📌 참고 자료

- [Recommender Systems with Python](https://github.com/lsjsj92/recommender_system_with_Python)
- [Surprise Library](https://github.com/NicolasHug/Surprise)
- [LightFM](https://github.com/lyst/lightfm)
- [Implicit](https://github.com/benfred/implicit)
