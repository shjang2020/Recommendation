# 6. build_ann_index.py
# 사용자 임베딩 데이터(parquet)을 로드하여 AnnoyInDdex(angular)빌드 및 저장(.idx)
# .idmap.npy로 index<->user_id 매핑 저장
# Approximate NN 인덱스 구축

import argparse, time, numpy as np, pandas as pd
from pathlib import Path
from annoy import AnnoyIndex

def build(index_path: Path, embed_path: Path, metric: str, n_trees: int):
    t0 = time.time()
    df = pd.read_parquet(embed_path)        # 1. 사용자 임베딩 불러오기기
    vecs = df.drop(columns=["user_id"]).values.astype("float32")
    dim  = vecs.shape[1]                    # 임베딩 차원

    index = AnnoyIndex(dim, metric)         # 2. Annoy 인덱스 생성
    for i, v in enumerate(vecs):            # 3. 모든 벡터를 트리에 추가
        index.add_item(i, v)
    index.build(n_trees)                    # 4. n_trees 만큼 트리 빌드
    index.save(str(index_path))             # 5. .idx 파일로 저장
    print(f"✅ Annoy index saved → {index_path}  "
          f"[{time.time()-t0:.1f}s | {n_trees} trees | {len(df):,} items]")

    # id ↔ idx 매핑 테이블도 저장
    idmap_path = index_path.with_suffix(".idmap.npy")   # 6. idx -> user_id 매핑 별도 저장
    np.save(idmap_path, df["user_id"].values.astype("int64"))
    print(f"↪ id-map saved → {idmap_path}")

if __name__ == "__main__":                  # 7. CLI 인자 정의
    p = argparse.ArgumentParser()
    p.add_argument("--embed_path", default="data/user_embeddings.parquet")
    p.add_argument("--index_path", default="data/annoy_user.idx")
    p.add_argument("--metric",     choices=["angular", "euclidean"], default="angular")
    p.add_argument("--n_trees",    type=int, default=50,
                   help="트리가 많을수록 recall↑, build time↑")
    cfg = p.parse_args()
    Path(cfg.index_path).parent.mkdir(parents=True, exist_ok=True)
    build(Path(cfg.index_path), Path(cfg.embed_path), cfg.metric, cfg.n_trees)