# 1. generate_nock_data.py
# 사용자 수 및 토픽 수 등 설정에 따라 사용자 주제 triplet(MockTriplet)을 랜덤으로 생성
# 학습용 사용자 positive, negative 예제 준비

import random
import pandas as pd

# 설정
NUM_USERS = 10000
NUM_TOPICS = 100
MAX_INTERESTS = 10
MAX_POSITIVE = 5

# 사용자 1명의 정보 생성
def generate_mock_user(user_id):
    gender = random.randint(0, 1)
    country = random.randint(0, 4)
    interest_topics = random.sample(range(NUM_TOPICS), random.randint(1, MAX_INTERESTS))
    ctr = round(random.uniform(0.01, 0.5), 3) # 클릭률
    saves = random.randint(0, 100) # 저장 수
    positive_topics = random.sample(interest_topics, min(len(interest_topics), MAX_POSITIVE))
    
    return {
        "user_id": user_id,
        "gender": gender,
        "country": country,
        "interests": interest_topics,
        "ctr": ctr,
        "saves": saves,
        "positive_topics": positive_topics
    }

# 전체 사용자 생성
mock_users = [generate_mock_user(i) for i in range(NUM_USERS)]

# Triplet (user, positive topic, negative topic) 생성
triplets = []
for user in mock_users:
    for pt in user["positive_topics"]:
        nt_candidates = [t for t in range(NUM_TOPICS) if t not in user["positive_topics"]]
        if not nt_candidates:
            continue
        nt = random.choice(nt_candidates)
        triplets.append({
            "user_id": user["user_id"],
            "gender": user["gender"],
            "country": user["country"],
            "interests": user["interests"],
            "ctr": user["ctr"],
            "saves": user["saves"],
            "positive_topic": pt,
            "negative_topic": nt
        })

# 데이터프레임으로 변환
df_triplets = pd.DataFrame(triplets)

# ✅ CSV로 저장
df_triplets.to_csv("data/mock_user_topic_triplets.csv", index=False)
print("✅ CSV 저장 완료: mock_user_topic_triplets.csv")
