# 2. dataset.py
# interests 컬럼 파싱, padding 및 collate_fn 정의
# StarSpace 학습용 배치 처리

import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
import torch.nn.functional as F

NUM_TOPICS = 100
PAD_ID = NUM_TOPICS

class MockTripletDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df['interests'] = self.df['interests'].apply(ast.literal_eval)  # 문자열을 리스트로

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gender = torch.tensor(row['gender'], dtype=torch.long)
        country = torch.tensor(row['country'], dtype=torch.long)
        interests = torch.tensor(row['interests'], dtype=torch.long)  # variable length
        ctr = torch.tensor([row['ctr']], dtype=torch.float32)
        saves = torch.tensor([row['saves']], dtype=torch.float32)
        pos_topic = torch.tensor(row['positive_topic'], dtype=torch.long)
        neg_topic = torch.tensor(row['negative_topic'], dtype=torch.long)

        return gender, country, interests, ctr, saves, pos_topic, neg_topic

    # ● 모든 사용자 특징을 리스트로 반환
    def get_all_user_tensors(self):
        users = []
        for idx in range(len(self.df)):
            g, c, intr, ctr, sav, *_ = self.__getitem__(idx)
            # 배치 차원(1) 추가
            g_batch    = g.unsqueeze(0)        # [1]
            c_batch    = c.unsqueeze(0)        # [1]
            intr_batch = intr.unsqueeze(0)     # [1, L]
            ctr_batch  = ctr.unsqueeze(0)      # [1, 1]
            sav_batch  = sav.unsqueeze(0)      # [1, 1]
            # valid_len도 1차원 텐서로
            valid_len  = torch.tensor([intr.size(0)], dtype=torch.long)  # [1]

            users.append((g_batch, c_batch,intr_batch, ctr_batch,sav_batch, valid_len))
        return users
    
def collate_fn(batch):
    genders, countries, interests, ctrs, saves, pos_topics, neg_topics = zip(*batch)

    genders = torch.stack(genders)  # [B, 1]
    countries = torch.stack(countries)  # [B, 1]

    # 가변 길이 interests → padding
    max_len = max([i.size(0) for i in interests])   # .size(0) : 첫번쨰 차원의 크기 반환
    padded, valid_lens = [], []
    for i in interests:
        valid_lens.append(i.size(0))
        pad_len = max_len - i.size(0)
        padded.append(F.pad(i, (0, pad_len), value=PAD_ID))
    interests = torch.stack(padded)  # [B, L]
    valid_lens = torch.tensor(valid_lens, dtype=torch.long) 
    
    ctrs = torch.stack(ctrs)
    saves = torch.stack(saves)
    pos_topics = torch.stack(pos_topics)
    neg_topics = torch.stack(neg_topics)

    return genders, countries, interests, ctrs, saves, pos_topics, neg_topics, valid_lens

