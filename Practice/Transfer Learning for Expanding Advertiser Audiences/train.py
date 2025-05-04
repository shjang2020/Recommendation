# 4. train.py
# dataset.py와 model.py를 불러와서 margin-ranking loss로 학습
# transfer error 기준 user encoder 최적 checkpoint 저장 - user_encoder_best.pth
# StarSpace - user encoder 학습 완료

import argparse, json, math, random, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm                       # 진행 막대

from dataset import (MockTripletDataset, collate_fn, NUM_TOPICS, PAD_ID)

from model import UserEncoder, TopicEmbedding, StarSpaceModel

# ----------------- utils -----------------
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():          # GPU가 있을 때만 CUDA 시드 고정
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True   # 계산결과 고정
        torch.backends.cudnn.benchmark     = False  # 변동 최적화 비활성화

# ----------------- train loop ------------
def train_one_epoch(model, loader, opt, margin, device, epoch):
    model.train()
    running, start = 0.0, time.time()
    for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch:02d}")):
        # 배치 데이터를 device로 이동
        gender, country, interests, ctr, saves, pos_t, neg_t, vl = [t.to(device) for t in batch]
        # 모델 forward pass
        sim_p, sim_n = model(gender, country, interests, ctr, saves, pos_t, neg_t, valid_lens=vl)
        # margin ranking loss 계산
        loss = F.margin_ranking_loss(sim_p, sim_n, target=torch.ones_like(sim_p), margin=margin)
        # 역전파 및 최적화
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item() * sim_p.size(0)

        if (step + 1) % 100 == 0:
            avg = running / ((step + 1) * loader.batch_size)
            print(f" ├─ step {step+1}/{len(loader)}  curr_loss {loss.item():.4f}  avg_loss {avg:.4f}")

    print(f" └─ Epoch {epoch:02d} done  avg_loss {running/len(loader.dataset):.4f}  time {(time.time()-start)/60:.1f} min")
    return running / len(loader.dataset)
    
# ------------- transfer eval  -------------
@torch.no_grad()
def build_transfer_set(u_idx, hold_ratio=.02):         # hold-out 2 %
    ids = list(u_idx); random.shuffle(ids)
    cut = max(2, int(len(ids) * hold_ratio))
    hold = ids[:cut]; known = ids[cut:]

    pairs = []
    for us in hold:
        un = random.choice(ids[cut:])      # 비시드 확보
        pairs.append((us, un))
    return known, pairs

@torch.no_grad()
def transfer_eval(model, hold_pairs, user_tensors, known_embs, device):
    model.eval()  # 평가 모드로 설정 (BatchNorm을 eval 모드로)
    known = known_embs.to(device)
    errs = 0
    for idx_s, idx_n in hold_pairs:
        us_emb = model.user_encoder(*[t.to(device) for t in user_tensors[idx_s]]).squeeze(0)
        un_emb = model.user_encoder(*[t.to(device) for t in user_tensors[idx_n]]).squeeze(0)
        if (known @ us_emb).sum() < (known @ un_emb).sum():
            errs += 1
    model.train()  # 다시 학습 모드로 복귀
    return errs / len(hold_pairs)

# ---------- 전체 파이프라인 --------------------------------------
def main(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"▶ using {device}")

    # 1. dataset
    ds = MockTripletDataset(cfg.csv_path)
    train_dl = DataLoader(ds, cfg.batch_size, shuffle=True, collate_fn=collate_fn)

    # 2. model
    ue = UserEncoder(num_interest_topics=NUM_TOPICS, pad_id=PAD_ID)
    te = TopicEmbedding(num_topics=NUM_TOPICS, embedding_dim=cfg.dim)
    model = StarSpaceModel(ue, te).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    
    # -------- transfer-eval 세트 준비 (epoch 1 직후 최초 1회) --------
    user_tensors = None
    known_set = hold_pairs = known_embs = None
    SAMPLE_K = 1000

    best_err, patience = 1.0, 0
    save_dir = Path("runs") / time.strftime("%Y-%m-%d-%H%M%S")
    save_dir.mkdir(parents=True)

    # 3. training loop
    for epoch in range(1, cfg.epochs + 1):
        print(f"\n▶▶ Epoch {epoch:02d} started")
        train_one_epoch(model, train_dl, opt, cfg.margin, device, epoch)
        
        # ---- transfer-eval 준비 (첫 epoch 끝난 뒤 1회) ----
        if epoch == 1:
            user_tensors = ds.get_all_user_tensors()
            known_set, hold_pairs = build_transfer_set(range(len(user_tensors)))
        
        # known seed 임베딩 (K개 샘플)
        k_sample = random.sample(known_set, min(SAMPLE_K, len(known_set)))
        model.eval()  # BatchNorm을 eval 모드로
        with torch.no_grad():
            known_embs = torch.stack([
                model.user_encoder(*user_tensors[i]).squeeze(0)
                for i in k_sample
            ]).cpu()
        
        # transfer evaluation
        val_st = time.time()
        trans_err = transfer_eval(model, hold_pairs, user_tensors, known_embs, device)
        print(f"▲ validation  TransferErr {trans_err:.4f}  ({time.time()-val_st:.1f}s)")

        # ---- early stopping ----
        if trans_err < best_err:
            best_err = trans_err
            patience = 0
            torch.save(model.user_encoder.state_dict(), save_dir / "user_encoder_best.pth")
            print("  ✔ checkpoint saved (TransferErr ↓)")
        else:
            patience += 1
            if patience >= cfg.es_patience:
                print("Early stopping")
                break

    # 4. meta info
    (save_dir / "meta.json").write_text(json.dumps(vars(cfg), indent=2))
    print(f"✦ best TransferErr={best_err:.4f}  ckpt={save_dir}")
    
# ---------- CLI 엔트리포인트 ----------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", default="data/mock_user_topic_triplets.csv")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--es_patience", type=int, default=3)
    ap.add_argument("--device", default="cuda")   # cpu / cuda
    ap.add_argument("--seed", type=int, default=42)
    cfg = ap.parse_args()
    main(cfg)
