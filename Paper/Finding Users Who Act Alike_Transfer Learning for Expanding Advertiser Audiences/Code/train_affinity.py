# 8. train_affinity.py
# parquet 형태의 embedding + seed_to_cands.npy을 로드해서 positive/negative pair 생성
# u, v, |u–v|, u*v + side-features(cosine 등) → MLP 학습 → affinity_mlp.pth 저장
# Affinity Scoring MLP 학습

import argparse, json, random, time, ast
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, numpy as np

# ---------- 1. MLP 정의 --------------------------------------------
class AffinityMLP(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ---------- 2. Pair 데이터셋 ---------------------------------------
class PairDataset(torch.utils.data.Dataset):
    def __init__(self, embed_path, lsh_path, pos_top=5, neg_per_pos=2):
        df = pd.read_parquet(embed_path)
        vec = torch.tensor(df.drop(columns=["user_id"]).values)
        ids = df["user_id"].values
        id2idx = {u:i for i,u in enumerate(ids)}

        lsh = np.load(lsh_path, allow_pickle=True).item()   # {seed:[cand...]}
        pairs, labels = [], []
        for seed_str, cands in lsh.items():
            # 문자열로 저장된 시드 ID 리스트를 파싱
            try:
                seeds = [int(seed_str)]  # 단일 ID인 경우
            except ValueError:
                # 리스트 형태의 문자열인 경우
                seeds = ast.literal_eval(seed_str)  # 모든 시드 ID 사용
            
            print(f"seeds: {seeds}")
            for seed in seeds:
                seed_idx = id2idx[seed]
                pos_idx = [id2idx[c] for c in cands[:pos_top]]
                for p in pos_idx:
                    pairs.append((seed_idx, p))
                    labels.append(1.)
                    for _ in range(neg_per_pos):
                        n = random.choice(cands[pos_top:])
                        pairs.append((seed_idx, id2idx[n]))
                        labels.append(0.)
        
        print(f"총 {len(set(p[0] for p in pairs))}개의 시드로부터 {len(pairs)}개의 학습 쌍 생성")
        self.vecs, self.pairs, self.labels = vec, pairs, torch.tensor(labels)

    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        a,b = self.pairs[i]
        u,v = self.vecs[a], self.vecs[b]
        # 논문의 4가지 기본 연산 적용
        feat = torch.cat([
            u * v,          # 요소별 곱
            u - v,          # 차이
            torch.abs(u - v),  # 절대 차이
            u + v,          # 합
        ])
        return feat, self.labels[i]

# ---------- 3. Train Loop ------------------------------------------
def train(cfg):
    ds = PairDataset(cfg.embed_path, cfg.lsh_pairs, cfg.pos_top, cfg.neg_per_pos)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch, shuffle=True)
    model = AffinityMLP(dim=cfg.dim)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best, patience = 0,0
    for epoch in range(cfg.epochs):
        model.train(); loss_sum=0
        for feat,y in dl:
            pred = model(feat).squeeze()  # [B,1] → [B]
            loss = F.binary_cross_entropy(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()*len(y)
        auroc = evaluate(model, dl)       # 간단 내부 AUROC
        print(f"[{epoch}] loss {loss_sum/len(ds):.4f}  AUROC {auroc:.3f}")
        if auroc>best: best,patience=auroc,0
        else: patience+=1
        if patience>=cfg.es: break
    Path(cfg.out_dir).mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), Path(cfg.out_dir)/"affinity_mlp.pth")

@torch.no_grad()
def evaluate(model, dl):
    model.eval(); y_true,y_pred=[],[]
    for feat,y in dl:
        y_true += y.tolist()
        y_pred += model(feat).squeeze().tolist()  # [B,1] → [B]
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true,y_pred)

# ---------- 4. CLI -------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--embed_path", default="data/user_embeddings.parquet")
    p.add_argument("--lsh_pairs",  default="data/seed_to_cands.npy")  # seed_id→list(cands)
    p.add_argument("--out_dir",    default="models")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--pos_top", type=int, default=5)
    p.add_argument("--neg_per_pos", type=int, default=2)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr",  type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--es", type=int, default=2)
    cfg = p.parse_args()
    train(cfg)