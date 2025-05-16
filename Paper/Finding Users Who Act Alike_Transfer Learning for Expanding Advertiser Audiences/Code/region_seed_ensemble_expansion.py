# 9. region_seed_ensemble_expansion.py: 메인 시스템
# LSH, Affinity MLP, 분류기를 결합한 앙상블 방식
# 분산 처리 지원
# 최종 사용자 확장 로직 구현
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression

from mapreduce_framework import MapReduceFramework, save_fragments, load_fragments
from lsh_mapreduce import LSHProcessor
from mapreduce_framework import FragmentedJoin, BucketSort

# 1) Affinity MLP 정의
class AffinityMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 4, 256),  # input_dim = 32, total = 32 * 4 = 128
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# 2) Pairwise feature 생성
def make_pairwise_features(seed_vec: np.ndarray, cand_vecs: np.ndarray) -> torch.FloatTensor:
    u = torch.from_numpy(seed_vec).float()  # [32]
    v = torch.from_numpy(cand_vecs).float()  # [B, 32]
    
    # 시드 벡터를 2D로 변환 (마지막 차원 유지)
    if u.dim() > 1:
        u = u.reshape(-1, u.size(-1))  # [..., 32] → [N, 32]
        u = u.mean(dim=0, keepdim=True)  # [N, 32] → [1, 32]
    else:
        u = u.unsqueeze(0)  # [32] → [1, 32]
    
    # 배치 크기만큼 확장
    u = u.expand(v.size(0), -1)  # [1, 32] → [B, 32]
    
    feat = torch.cat([
        u * v,          # 요소별 곱 [B, 32]
        u - v,          # 차이 [B, 32]
        torch.abs(u - v),  # 절대 차이 [B, 32]
        u + v,          # 합 [B, 32]
    ], dim=1)  # 최종: [B, 128]
    return feat

def process_embeddings(args):
    """임베딩 데이터 처리 및 LSH 기반 후보 추출"""
    # 1. 임베딩 로드 및 전처리
    print("Loading embeddings...")
    emb_df = pd.read_parquet(args.embed_path).set_index('user_id')
    if emb_df.index.duplicated().any():
        print("Warning: Found duplicate user IDs in embeddings. Using first occurrence...")
        emb_df = emb_df[~emb_df.index.duplicated(keep='first')]
    print(f"Loaded embeddings: {len(emb_df)} users, {emb_df.shape[1]} dimensions")
    
    # 2. LSH 처리 (MapReduce)
    print("Running distributed LSH processing...")
    lsh_processor = LSHProcessor(
        n_workers=args.n_workers,
        n_trees=args.n_trees,
        top_k=args.top_k_lsh
    )
    valid_seeds = [sid for sid in args.seed_ids if sid in emb_df.index]
    if len(valid_seeds) != len(args.seed_ids):
        print(f"Warning: Only {len(valid_seeds)} out of {len(args.seed_ids)} seed users found in embeddings")
    
    seed_indices = [emb_df.index.get_loc(sid) for sid in valid_seeds]
    lsh_candidates = lsh_processor.process(emb_df.values, seed_indices)
    
    # 인덱스를 실제 user_id로 변환
    user_ids = emb_df.index.values
    lsh_counts = {
        valid_seeds[seed_idx]: [user_ids[cand_idx] for cand_idx in cands]
        for seed_idx, cands in lsh_candidates.items()
    }
    
    return emb_df, valid_seeds, lsh_counts

def score_candidates(args, emb_df, valid_seeds, lsh_counts):
    """Affinity MLP와 Classifier로 후보 점수 계산"""
    # 1. Affinity MLP 스코어링
    print("Running Affinity MLP scoring...")
    affinity_model = AffinityMLP(input_dim=emb_df.shape[1])
    affinity_model.load_state_dict(
        torch.load('models/affinity_mlp.pth', map_location='cpu', weights_only=True)
    )
    affinity_model.eval()
    
    # Fragment-and-Replicate Join 사용
    join_processor = FragmentedJoin(n_fragments=args.n_workers)
    bucket_sorter = BucketSort(n_buckets=20)
    
    embed_sorted = {}
    for sid in valid_seeds:
        seed_vec = emb_df.loc[sid].values
        cands = list(lsh_counts[sid])
        cand_vecs = emb_df.loc[cands].values
        
        # 큰 후보 데이터셋을 fragment하여 처리
        fragments = join_processor.fragment_and_replicate(
            cand_vecs,
            seed_vec.reshape(1, -1),
            lambda x: 0,  # 모든 후보는 같은 시드와 조인
            lambda x: 0
        )
        
        all_scores = []
        for cand_frag, seed_frag in fragments:
            feats = make_pairwise_features(seed_frag[0], cand_frag)
            with torch.no_grad():
                scores = affinity_model(feats).squeeze().numpy()
            all_scores.extend(scores)
        
        # 버킷 소팅으로 점수 기반 정렬
        scored_cands = list(zip(all_scores, cands))
        sorted_cands = bucket_sorter.sort(scored_cands)
        embed_sorted[sid] = [c for _, c in sorted_cands]
    
    # 2. Classifier 학습 및 스코어링
    print("Training and applying classifier...")
    all_users = emb_df.index.unique()
    non_seed = np.setdiff1d(all_users, valid_seeds)
    n_seeds = len(valid_seeds)
    neg_ids = np.random.choice(non_seed, size=n_seeds, replace=False)
    
    # 학습 데이터 준비
    X_train = np.vstack([
        emb_df.loc[valid_seeds].values,
        emb_df.loc[neg_ids].values
    ])
    y_train = np.array([1] * n_seeds + [0] * n_seeds)
    
    # 분류기 학습
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Fragment-and-Replicate Join으로 예측
    clf_sorted = {}
    for sid in valid_seeds:
        cands = list(lsh_counts[sid])
        cand_vecs = emb_df.loc[cands].values
        
        fragments = join_processor.fragment_and_replicate(
            cand_vecs,
            np.array([]),  # classifier는 right 데이터셋 불필요
            lambda x: 0,
            lambda x: 0
        )
        
        all_probs = []
        for cand_frag, _ in fragments:
            probs = clf.predict_proba(cand_frag)[:,1]
            all_probs.extend(probs)
        
        # 버킷 소팅으로 확률 기반 정렬
        scored_cands = list(zip(all_probs, cands))
        sorted_cands = bucket_sorter.sort(scored_cands)
        clf_sorted[sid] = [c for _, c in sorted_cands]
    
    return embed_sorted, clf_sorted

def merge_results(embed_sorted, clf_sorted, final_k):
    """앙상블 결과 병합"""
    print("Merging ensemble results...")
    final_mapping = {}
    for sid in embed_sorted.keys():
        # 1. 교집합으로 후보를 우선 축소
        inter = [u for u in embed_sorted[sid] if u in clf_sorted[sid]]
        
        # 2. 양쪽에서 제외된 나머지 후보
        rem_embed = [u for u in embed_sorted[sid] if u not in inter]
        rem_clf = [u for u in clf_sorted[sid] if u not in inter]
        
        # 3. 교집합 + 나머지 후보 병합
        merged = inter.copy()
        merged.extend(rem_embed)
        merged.extend(rem_clf)
        
        final_mapping[sid] = merged[:final_k]
    
    return final_mapping

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_path', type=str, default='data/user_embeddings.parquet',
                        help='user embeddings 경로')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='분산 처리 워커 수')
    parser.add_argument('--n_trees', type=int, default=10,
                        help='LSH 트리 개수')
    parser.add_argument('--top_k_lsh', type=int, default=200,
                        help='LSH 후보 Top-K')
    parser.add_argument('--top_k_final', type=int, default=100,
                        help='최종 추천 사용자 수')
    parser.add_argument('--seed_ids', nargs='+', type=int, required=True,
                        help='시드 사용자 ID 리스트')
    parser.add_argument('--out_path', type=str, default='data/final_expanded.npy',
                        help='최종 확장 사용자 저장 경로')
    args = parser.parse_args()
    
    # 1. 임베딩 처리 및 LSH 후보 추출
    emb_df, valid_seeds, lsh_counts = process_embeddings(args)
    
    # 2. 후보 점수 계산 (Affinity MLP + Classifier)
    embed_sorted, clf_sorted = score_candidates(args, emb_df, valid_seeds, lsh_counts)
    
    # 3. 결과 병합
    final_mapping = merge_results(embed_sorted, clf_sorted, args.top_k_final)
    
    # 4. 저장
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_path, final_mapping)
    print(f"Saved final expanded audiences → {args.out_path}")