# 3 .model.py
# UserEncoder : 카테고리 + 연속형 -> 32차원 임베딩
# TopicEmbedding : 토픽ID -> 32차원
# StarSpaceModel : user x (pos, neg) dot-product 학습 구조
# StarSpace 모델 아키텍처 정의

import torch
import torch.nn as nn
import torch.nn.functional as F

# 사용자 피처 인코더
class UserEncoder(nn.Module):
    def __init__(self, num_genders=2, num_countries=5, 
                 num_interest_topics=10, embedding_dim=32, pad_id=None):
        super().__init__()
        self.pad_id = pad_id if pad_id is not None else num_interest_topics
        self.gender_embed = nn.Embedding(num_genders, embedding_dim)
        self.country_embed = nn.Embedding(num_countries, embedding_dim)
        self.interest_embed = nn.Embedding(num_interest_topics + 1, 
                                           embedding_dim, 
                                           padding_idx=self.pad_id)

        # continuous 2-dim -> BN -> FC
        self.bn_cont = nn.BatchNorm1d(2)        # (ctr, saves)
        self.fc = nn.Linear(embedding_dim * 3 + 2, embedding_dim)  # 3 categorical + 2 continuous = 98
        self.activation = nn.ReLU()

    def forward(self, gender, country, interests, ctr, saves, valid_lens):
        gender_emb = self.gender_embed(gender)           # (B, 32)
        country_emb = self.country_embed(country)         # (B, 32)
        
        # interests mean pooling (padding은 제외하고 임베딩을 추출하기 위함)
        i_emb = self.interest_embed(interests)     # (B, K, 32)
        mask = (interests != self.pad_id).unsqueeze(-1)  # pad id와 같은건 mask함
        interest_emb = (i_emb * mask).sum(dim=1) / valid_lens.unsqueeze(1)
        
        # continuous features : log1p & BN
        cont = torch.cat([ctr.log1p(), saves.log1p()], dim=1) # (B,2)
        cont = self.bn_cont(cont)
        
        x = torch.cat([gender_emb, country_emb, interest_emb, cont], dim=1) # (B, 32 + 32 + 32 + 1 + 1 = 98)
        x = self.activation(self.fc(x))
        return F.normalize(x, p=2, dim=1)  # final user embedding / shape : (B, 32)

# 토픽 임베딩
class TopicEmbedding(nn.Module):
    def __init__(self, num_topics, embedding_dim=32):
        super(TopicEmbedding, self).__init__()
        self.topic_embed = nn.Embedding(num_topics, embedding_dim)

    def forward(self, topic_ids):
        return self.topic_embed(topic_ids)

# 전체 모델
class StarSpaceModel(nn.Module):
    def __init__(self, user_encoder, topic_embed):
        super(StarSpaceModel, self).__init__()
        self.user_encoder = user_encoder
        self.topic_embed = topic_embed

    def forward(self, gender, country, interests, ctr, saves, pos_topic, neg_topic, valid_lens):
        user_vec = self.user_encoder(gender, country, interests, ctr, saves, valid_lens)  # (B, D)
        t_pos = self.topic_embed(pos_topic)  # (B, D)
        t_neg = self.topic_embed(neg_topic)  # (B, D)

        # 유사도: 내적(dot product)
        sim_pos = (user_vec * t_pos).sum(dim=1)  # (B,)
        sim_neg = (user_vec * t_neg).sum(dim=1)  # (B,)

        return sim_pos, sim_neg

# 전체 흐름 요약

# Batch Input
# ↓
# UserEncoder
#   ├─ gender/country → embedding (B, 32)
#   ├─ interests → (B, K, 32) → mean → (B, 32)
#   └─ ctr, saves → (B, 1)
#   → concat → (B, 98)
#   → fc layer → user_vec (B, 32)

# TopicEmbedding
#   ├─ pos_topic → (B, 32)
#   ├─ neg_topic → (B, 32)

# 유사도 계산
#   → dot(user_vec, t_pos) → sim_pos (B,)
#   → dot(user_vec, t_neg) → sim_neg (B,)