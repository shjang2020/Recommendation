# Finding Users Who Act Alike: Transfer Learning for Expanding Advertiser Audiences

[📄 논문 원문 (KDD 2019)](https://dl.acm.org/doi/10.1145/3292500.3330742)  
[📖 ArXiv 버전](https://arxiv.org/abs/1903.01625)  
[🔗 GitHub 코드](https://github.com/shjang2020/Recommendation/tree/master/Practice/Transfer%20Learning%20for%20Expanding%20Advertiser%20Audiences)

---

## 프로젝트 소개

본 리포지토리는 “Finding Users Who Act Alike: Transfer Learning for Expanding Advertiser Audiences” 논문에서 제안된 방법을 **PyTorch** 기반으로 구현한 코드와 실험 스크립트를 제공합니다.  
- **유저 임베딩 학습**: StarSpace 기반 대조 학습 모델  
- **시드 유저 기반 추천**: 평균 벡터 + 코사인 유사도  
- **확장 단계**: LSH(Locality‐Sensitive Hashing) + Affinity Scoring MLP  
- **평가 지표**: Precision@K, Recall@K

---

## 설치 및 환경 구성

```bash
# 1. 리포지토리 클론
git clone https://github.com/shjang2020/Recommendation.git

# 2. 해당 디렉터리로 이동
cd Recommendation/Practice/"Transfer Learning for Expanding Advertiser Audiences"

# 3. 가상환경 생성 및 활성화 (optional)
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

# 4. 의존성 설치
pip install -r requirements.txt

