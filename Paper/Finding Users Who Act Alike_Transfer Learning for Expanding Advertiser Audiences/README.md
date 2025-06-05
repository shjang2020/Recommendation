# Finding Users Who Act Alike: Transfer Learning for Expanding Advertiser Audiences

- **원문**: [Finding Users Who Act Alike : Transfer Learning for Expanding Advertiser Audiences](https://www.pinterestlabs.com/media/phkg2uau/transferlearning-kdd2019.pdf)
- **발행연도**: 2019
- **분야**: 추천시스템

## 논문 리뷰

- [논문 리뷰 PDF](./Finding%20Users%20Who%20Act%20Alike(Transfer%20Learning%20for%20Expanding%20Advertiser%20Audiences)%20논문%20리뷰.pdf)
- [Notion](https://roasted-rake-be8.notion.site/Finding-Users-Who-Act-Alike-Transfer-Learning-for-Expanding-Advertiser-Audiences-1dc818aea60f80c0a738e856a4b1dfb2)

## 구현

- **코드 위치**: [`Code/`](./Code)
- **주요 구현 내용**:
  - StarSpace 기반 유저 임베딩 학습
  - 시드 유저 기반 후보 확장 (Annoy, LSH, 코사인 유사도)
  - Affinity MLP 및 앙상블 확장
  - 대규모 임베딩 분산 처리 (MapReduce 예시)
  - 실험용 mock 데이터 생성 및 전체 파이프라인 제공

### 주요 스크립트
- `generate_mock_data.py`: 모의 사용자-토픽 데이터 생성
- `train.py`: StarSpace-style margin-ranking loss로 임베딩 학습
- `embed.py`: 임베딩 추출 및 저장
- `build_ann_index.py`: Annoy 인덱스 생성
- `expand_seed_users.py`: 시드 기반 후보 확장
- `train_affinity.py`: Affinity MLP 학습
- `region_seed_ensemble_expansion.py`: 앙상블 확장 파이프라인
- `lsh_mapreduce.py`, `mapreduce_framework.py`: 분산 LSH MapReduce 예시

### 데이터 예시
- `data/mock_user_topic_triplets.csv`: 사용자-토픽 상호작용 데이터
- `data/user_embeddings.parquet`: 학습된 유저 임베딩
- `data/annoy_user.idx`, `data/annoy_user.idmap.npy`: Annoy 인덱스 및 매핑
- `data/seed_to_cands.npy`, `data/final_expanded.npy`: 확장 결과

### 모델 예시
- `models/affinity_mlp.pth`: 학습된 Affinity MLP 모델

---

## 실행 예시

```bash
# 1. 환경 세팅
conda env create -f environment.yml
conda activate transfer-learning-ad-audiences

# 2. 데이터 생성
python generate_mock_data.py --num-users 10000 --num-topics 100 --interactions-per-user 50

# 3. 임베딩 학습
python train.py --csv_path data/mock_user_topic_triplets.csv --epochs 10 --batch_size 256 --dim 32 --lr 1e-3 --margin 0.2 --es_patience 3

# 4. 임베딩 추출
python embed.py --csv_path data/mock_user_topic_triplets.csv --model_path runs/<timestamp>/user_encoder_best.pth --out_path data/user_embeddings.parquet --format parquet

# 5. Annoy 인덱스 생성
python build_ann_index.py --embed_path data/user_embeddings.parquet --index_path data/annoy_user.idx --metric angular --n_trees 50

# 6. 시드 기반 후보 확장
python expand_seed_users.py 1234 5678 9012 --top_k 200 --index_path data/annoy_user.idx --search_k 500 --pairs_out data/seed_to_cands.npy

# 7. Affinity MLP 학습
python train_affinity.py --embed_path data/user_embeddings.parquet --lsh_pairs data/seed_to_cands.npy --out_dir models --dim 32 --epochs 10 --batch 512

# 8. 앙상블 확장 실행
python region_seed_ensemble_expansion.py --embed_path data/user_embeddings.parquet --seed_ids 1234 5678 9012 --n_workers 4 --n_trees 10 --top_k_lsh 200 --top_k_final 100 --out_path data/final_expanded.npy
```

---

## 📂 폴더 구조

```
Finding Users Who Act Alike_Transfer Learning for Expanding Advertiser Audiences/
├── Finding Users Who Act Alike_리뷰.pdf
├── README.md
└── Code/
    ├── data/
    ├── models/
    ├── runs/
    ├── *.py
    ├── environment.yml
    └── README.md
```
