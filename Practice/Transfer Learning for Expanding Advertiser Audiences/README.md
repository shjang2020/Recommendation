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
```
---

## 주요 스크립트
1. train_starspace.py
- 유저 임베딩 학습용 학습 스크립트
- positive/negative pair 생성
2. extract_embeddings.py
- 학습된 모델에서 유저 임베딩 추출
3. seed_recommendation.py
- 시드 유저 리스트를 받아 평균 벡터 기반 기본 추천
4. expand_seed_users.py
- LSH를 이용한 후보군 검색 후 Affinity MLP로 점수 매김
5, evaluate.py
- 추천 결과에 대한 Precision@K, Recall@K 계산

---

## 사용 예시
1) 임베딩 학습
```bash
python train_starspace.py \
  --interactions data/interactions.csv \
  --epochs 10 \
  --batch-size 1024 \
  --embedding-dim 128 \
  --output models/starspace.pth
```
2) 임베딩 추출
```bash
python extract_embeddings.py \
  --model-path models/starspace.pth \
  --output embeddings/user_embeddings.csv
```
3) 시드 기반 기본 추천
```bash
python seed_recommendation.py \
  --embeddings embeddings/user_embeddings.csv \
  --seed-list seeds.txt \
  --top-k 100 \
  --output recommendations/seed_basic.csv
```
4) LSH + Affinity 확장 추천
```bash
python expand_seed_users.py \
  --embeddings embeddings/user_embeddings.csv \
  --seed-list seeds.txt \
  --lsh-index-path indices/user_lsh.index \
  --affinity-model models/affinity.pth \
  --top-k 100 \
  --output recommendations/expanded.csv
```
5) 평가
```bash
python evaluate.py \
  --recommendations recommendations/expanded.csv \
  --ground-truth data/ground_truth.csv \
  --metrics precision recall \
  --k 100
```
---
## 디렉터리 구조
```bash
├── data/
│   ├── interactions.csv
│   ├── user_meta.csv
│   └── item_meta.csv
├── indices/
│   └── user_lsh.index
├── models/
│   ├── starspace.pth
│   └── affinity.pth
├── embeddings/
│   └── user_embeddings.csv
├── recommendations/
│   ├── seed_basic.csv
│   └── expanded.csv
├── train_starspace.py
├── extract_embeddings.py
├── seed_recommendation.py
├── expand_seed_users.py
├── evaluate.py
├── requirements.txt
└── README.md
```
