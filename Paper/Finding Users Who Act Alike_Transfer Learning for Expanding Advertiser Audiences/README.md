# Finding Users Who Act Alike : Transfer Learning for Expanding Advertiser Audiences 리뷰

- 논문 리뷰 : [PDF](https://github.com/shjang2020/Recommendation/blob/master/Paper/Finding%20Users%20Who%20Act%20Alike%20%3A%20Transfer%20Learning%20for%20Expanding%20Advertiser%20Audiences/Finding%20Users%20Who%20Act%20Alike_%EB%A6%AC%EB%B7%B0.pdf) 및 [Notion](https://roasted-rake-be8.notion.site/Finding-Users-Who-Act-Alike-Transfer-Learning-for-Expanding-Advertiser-Audiences-1dc818aea60f80c0a738e856a4b1dfb2?pvs=4) 참고 부탁드립니다.

- Code 구현 : [Github](https://github.com/shjang2020/Recommendation/tree/master/Practice/Transfer%20Learning%20for%20Expanding%20Advertiser%20Audiences) 참고 부탁드립니다.

---
## Abstract | 논문 한눈에 보기

온라인 광고에서 광고주는 기존 고객과 유사한 신규 고객을 찾는 **Audience Expansion** 기술을 활용합니다. 본 논문에서는 Pinterest가 **실제 서비스에 적용한 임베딩 기반 Audience Expansion 모델**을 소개합니다.

핵심 아이디어는 다음과 같습니다.

- Pinterest의 모든 사용자 데이터를 활용해 **전역 사용자 임베딩 모델**을 학습
- 광고주가 제공한 소규모 고객 리스트(Seed)를 전역 임베딩 공간에서 효율적으로 표현하고, 이를 통해 신규 고객의 유사도를 측정
- 기존 광고주별 분류기(Classifier) 모델과 **앙상블(Ensemble)** 하여 성능을 극대화

실험 결과, 제안된 모델은 특히 **소규모 Seed 리스트**에서 기존 분류기 모델의 한계를 크게 극복했으며, 실제 서비스에서 높은 성과를 나타냈습니다.
