# 7. expand_seed_users.py - 광고주가 들어왔을 때 후보군 추출
# CLI 인자(seed_ids, top_k 등) 수신
# AnnoyIndex 로드 -> per-seed & mean-vector 기반 후보 추출
# AnnoyIndex가 없는 경우 Cosine fallback
# optional로 seed_to_cands.npy에 저장
# LSH로 후보군을 선별

"""
expand_seed_users.py
──────────────────────────────────────────────────────────
1) 사용자 임베딩  : embed.py → data/user_embeddings.parquet
   ├─ user_id
   └─ dim_0 … dim_31 (L2-norm)
   
2) Annoy 인덱스   : build_ann_index.py → data/annoy_user.idx
   └─ .idmap.npy  : idx ↔ user_id 매핑

3) 후보군 저장    : --pairs_out 옵션 사용 시
   numpy .npy (pickle) { "<seed_ids>": [candidate_id, …] }

사용 예)
$ python expand_seed_users.py \
        --top_k 200 \
        --index_path data/annoy_user.idx \
        --search_k 1000 \
        --pairs_out data/seed_to_cands.npy
"""

import argparse, os
from pathlib import Path
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity     # fallback 용

# ---------- Annoy 유틸 --------------------------------------------
def load_annoy(index_path: str, embed_path: str):
    schema = pq.read_schema(embed_path)
    dim = len(schema) - 1

    idmap = np.load(Path(index_path).with_suffix(".idmap.npy"))
    ann = AnnoyIndex(dim, 'angular')
    ann.load(index_path)
    return ann, idmap

def get_candidates_annoy(ann, idmap, seed_ids, top_k, search_k):
    """Annoy 기반 후보군 추출 (seed 개별 + 평균벡터)"""
    seed_idx = []
    for sid in seed_ids:
        idx_arr = np.where(idmap == sid)[0]
        if len(idx_arr) == 0:
            raise ValueError(f"seed_id {sid}는 인덱스에 없습니다.")
        seed_idx.append(int(idx_arr[0]))
    # 1. 각 seed 개별 이웃 top-k 추출
    neighbor_sets = [ann.get_nns_by_item(i, top_k, search_k=search_k) for i in seed_idx]
    # 2. seed 평균 벡터의 이웃
    mean_vec = np.mean([ann.get_item_vector(i) for i in seed_idx], axis=0) # seed 벡터들의 평균
    mean_cand = ann.get_nns_by_vector(mean_vec.tolist(), top_k, search_k=search_k) # 평균벡터 기준 이웃 top-k 추출
    # 3. 인덱스 집합(번호 중복 제거, 시드 인덱스 제외)
    cand_idx = set(mean_cand).union(*neighbor_sets) - set(seed_idx)
    # 4. 번호 → user_id 매핑
    cand_uids = idmap[list(cand_idx)]
    # 5. user_id 기준 중복 제거 + 시드 제거 + 순서 고정
    seen = set(seed_ids)    # 시드는 미리 넣어 제외
    rec_ids_unique = []
    for uid in cand_uids:
        if uid not in seen:
            seen.add(uid)
            rec_ids_unique.append(uid)
        if len(rec_ids_unique) >= top_k:   # early stop
            break
    return rec_ids_unique  # list[int] 길이 ≤ top_k

# ---------- 기존 Cosine fallback ----------------------------------
def load_embeddings(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

def recommend_cosine(seed_ids, top_k, emb_path):
    df = load_embeddings(emb_path)
    seed_vecs = df[df["user_id"].isin(seed_ids)].drop(columns=["user_id"]).values
    seed_mean = np.mean(seed_vecs, axis=0, keepdims=True)
    sims = cosine_similarity(seed_mean, df.drop(columns=["user_id"]).values).flatten()
    result = (pd.DataFrame({"user_id": df["user_id"], "sim": sims})
                .query("user_id not in @seed_ids")
                .sort_values("sim", ascending=False)
                .head(top_k))
    return result["user_id"].values

# ---------- main ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Seed user 기반 후보군 추출 (Annoy + Cosine fallback)")
    parser.add_argument("seed_ids", nargs="+", type=int, help="seed user id 리스트")
    parser.add_argument("--top_k", type=int, default=100, help="후보군 크기")
    parser.add_argument("--index_path", type=str, default="data/annoy_user.idx", help="Annoy .idx 경로")
    parser.add_argument("--search_k", type=int, default=-1, help="Annoy search_k (기본 n_trees*10)")
    parser.add_argument("--embedding_path", type=str, default="data/user_embeddings.parquet",
                        help="코사인 fallback 시 임베딩 경로")
    parser.add_argument("--pairs_out", type=str, default="data/seed_to_cands.npy",
                        help="후보군 딕셔너리를 .npy 로 저장")
    args = parser.parse_args()

    if args.index_path and Path(args.index_path).exists():
        ann, idmap = load_annoy(args.index_path, embed_path=args.embedding_path)
        if args.search_k == -1:
            args.search_k = ann.get_n_trees() * 10
        rec_ids = get_candidates_annoy(ann, idmap, args.seed_ids, args.top_k, args.search_k)
        print(f"📌 Annoy 결과 Top-{args.top_k}  (search_k={args.search_k})")
    else:
        rec_ids = recommend_cosine(args.seed_ids, args.top_k, args.embedding_path)
        print(f"📌 Cosine(fallback) 결과 Top-{args.top_k}")
    print(rec_ids)
    
    # ── 후보군 저장
    if args.pairs_out:
        Path(args.pairs_out).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(args.pairs_out):
            mapping = np.load(args.pairs_out, allow_pickle=True).item()
        else:
            mapping = {}
        mapping[str(args.seed_ids)] = rec_ids
        np.save(args.pairs_out, mapping)
        print(f"💾 pairs saved → {args.pairs_out}")

if __name__ == "__main__":
    main()