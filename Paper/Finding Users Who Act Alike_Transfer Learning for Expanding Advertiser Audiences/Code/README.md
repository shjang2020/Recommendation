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

## 환경 설정

```bash
# 1. 리포지토리 클론
git clone https://github.com/shjang2020/Recommendation.git
cd Recommendation/Practice/"Transfer Learning for Expanding Advertiser Audiences"

# 2. Conda 환경 생성 & 활성화
conda env create -f environment.yml
conda activate transfer-learning-ad-audiences
```
---

## 주요 스크립트 개요
1. generate_mock_data.py
- 사용자-토픽 triplet 형태의 모의 데이터를 생성합니다.
2. dataset.py
- mock_user_topic_triplets.csv를 로드해 MockTripletDataset과 collate_fn을 제공합니다.
3. model.py
- UserEncoder, TopicEmbedding, StarSpaceModel 아키텍처를 정의합니다.
4. train.py
- StarSpace-style margin-ranking loss로 유저 임베딩을 학습하고 최적 체크포인트를 저장합니다.
5. embed.py
-  학습된 user_encoder_best.pth를 로드해 전체 사용자 벡터를 추출, Parquet/CSV로 저장합니다.
6. build_ann_index.py
- 추출된 사용자 임베딩을 Annoy 인덱스로 빌드하고 .idx 및 .idmap.npy를 저장합니다.
7. expand_seed_users.py
- 시드 유저 리스트를 받아 Annoy 또는 cosine 유사도로 후보군을 확장·저장합니다.
8. train_affinity.py
- LSH로 생성된 시드-후보 쌍으로부터 positive/negative pair를 샘플링해 Affinity Scoring MLP를 학습합니다.
9. lsh_mapreduce.py & mapreduce_framework.py
- 대규모 임베딩을 분산 처리하는 LSH MapReduce 예시를 제공합니다.
10. region_seed_ensemble_expansion.py
- LSH, Affinity MLP, 그리고 LogisticClassifier를 결합한 앙상블 확장 파이프라인의 메인 스크립트입니다.

---

## 사용 예시
1) 모의 데이터 생성
```bash
python generate_mock_data.py \
  --num-users 10000 \
  --num-topics 100 \
  --interactions-per-user 50
```
2) 임베딩 학습
```bash
python train.py \
  --csv_path data/mock_user_topic_triplets.csv \
  --epochs 10 \
  --batch_size 256 \
  --dim 32 \
  --lr 1e-3 \
  --margin 0.2 \
  --es_patience 3
```
3) 임베딩 추출
```bash
python embed.py \
  --csv_path data/mock_user_topic_triplets.csv \
  --model_path runs/<timestamp>/user_encoder_best.pth \
  --out_path data/user_embeddings.parquet \
  --format parquet
```
4) Annoy 인덱스 생성
```bash
python build_ann_index.py \
  --embed_path data/user_embeddings.parquet \
  --index_path data/annoy_user.idx \
  --metric angular \
  --n_trees 50
```
5) 시드 기반 후보 확장
```bash
python expand_seed_users.py \
  1234 5678 9012 \
  --top_k 200 \
  --index_path data/annoy_user.idx \
  --search_k 500 \
  --pairs_out data/seed_to_cands.npy
```
6) Affinity MLP 학습
```bash
python train_affinity.py \
  --embed_path data/user_embeddings.parquet \
  --lsh_pairs data/seed_to_cands.npy \
  --out_dir models \
  --dim 32 \
  --epochs 10 \
  --batch 512
```
7) 앙상블 확장 실행
```bash
python region_seed_ensemble_expansion.py \
  --embed_path data/user_embeddings.parquet \
  --seed_ids 1234 5678 9012 \
  --n_workers 4 \
  --n_trees 10 \
  --top_k_lsh 200 \
  --top_k_final 100 \
  --out_path data/final_expanded.npy
```

---
## 디렉터리 구조
```bash
├── data/
│   ├── mock_user_topic_triplets.csv
│   ├── user_embeddings.parquet
│   ├── annoy_user.idx
│   ├── annoy_user.idmap.npy
│   ├── seed_to_cands.npy
│   └── final_expanded.npy
├── models/
│   └── affinity_mlp.pth
├── runs/
│   └── user_encoder_best.pth
├── generate_mock_data.py
├── dataset.py
├── model.py
├── train.py
├── embed.py
├── build_ann_index.py
├── expand_seed_users.py
├── train_affinity.py
├── lsh_mapreduce.py
├── mapreduce_framework.py
├── region_seed_ensemble_expansion.py
├── environment.yml 
└── README.md
```
