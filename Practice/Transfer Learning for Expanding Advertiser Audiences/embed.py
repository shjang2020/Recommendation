# 5. embed.py
# train.py를 통해 만들어진 최적의 모델인 user_encoder_best.pth를 로드하여
# 전체 사용자 벡터 추출, parquet 저장 - data/user_embeddings.parquet 

import argparse, time
import pandas as pd, torch
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import MockTripletDataset, collate_fn, NUM_TOPICS, PAD_ID
from model   import UserEncoder

@torch.inference_mode()
def extract_embeddings(model, loader, device, l2_norm=True):
    model.eval()
    rows = []
    for batch in loader:
        gender, country, interests, ctr, saves, _, _, vl = [t.to(device) for t in batch]
        vecs = model(gender, country, interests, ctr, saves, valid_lens=vl)  # (B, D)
        if l2_norm:
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
        rows.append(vecs.cpu())
    return torch.cat(rows, 0)  # (N, D)

def main(cfg):
    t0 = time.time()
    device = torch.device("cuda" if cfg.device=="cuda" and torch.cuda.is_available() else "cpu")
    print("▶ using", device)

    # 1. 데이터
    ds = MockTripletDataset(cfg.csv_path)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                    collate_fn=collate_fn, num_workers=cfg.workers)

    # 2. 모델 로드
    ue = UserEncoder(num_interest_topics=NUM_TOPICS, pad_id=PAD_ID).to(device)
    ue.load_state_dict(torch.load(cfg.model_path, map_location=device))

    # 3. 임베딩 추출
    vecs = extract_embeddings(ue, dl, device, l2_norm=cfg.normalize)  # (N,32)

    # 4. user_id 컬럼 결합
    user_ids = ds.df["user_id"].values
    df_out = pd.DataFrame(vecs.numpy(), columns=[f"dim_{i}" for i in range(vecs.size(1))])
    df_out.insert(0, "user_id", user_ids)

    # 5. 저장
    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)   # 폴더가 없으면 만든다
    if cfg.format == "parquet":
        df_out.to_parquet(out_path, index=False)
    else:
        df_out.to_csv(out_path, index=False)
    print(f"✅ Saved {len(df_out):,} embeddings → {out_path} ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path",   default="mock_user_topic_triplets.csv")
    p.add_argument("--model_path", default="runs/2025-04-25-1732/user_encoder_best.pth")
    p.add_argument("--out_path",   default="data/user_embeddings.parquet")
    p.add_argument("--format",     choices=["parquet", "csv"], default="parquet")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--normalize",  action="store_true")
    p.add_argument("--device",     default="cpu")
    cfg = p.parse_args()
    main(cfg)
