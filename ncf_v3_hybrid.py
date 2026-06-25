"""
KKBox v3 — NCF 임베딩 + LightGBM 하이브리드
============================================
전략:
  1단계: NCF를 학습시켜 유저/곡 임베딩을 만듦 (딥러닝 = 표현 학습)
  2단계: 그 임베딩을 LightGBM의 피처로 넣음 (트리 = 일반화된 예측)
  → "딥러닝이 잘하는 것만 딥러닝에게 맡기자"
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import time
import gc
import warnings
warnings.filterwarnings('ignore')

import sys, io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def log(msg):
    print(msg, flush=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cpu')

log("=" * 65)
log("  KKBox v3: NCF Embedding + LightGBM Hybrid")
log("=" * 65)

# ================================================================
# 1. 데이터 로드
# ================================================================
log("\n[1/7] 데이터 로드...")
t0 = time.time()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
songs = pd.read_csv('songs.csv')
members = pd.read_csv('members.csv')
extra = pd.read_csv('song_extra_info.csv', usecols=['song_id', 'isrc'])

log(f"  train: {len(train):,}, test: {len(test):,} ({time.time()-t0:.1f}s)")

# ================================================================
# 2. ID 매핑
# ================================================================
log("\n[2/7] ID 매핑...")
all_users = pd.concat([train['msno'], test['msno']]).unique()
all_songs = pd.concat([train['song_id'], test['song_id']]).unique()
user2idx = {uid: i for i, uid in enumerate(all_users)}
song2idx = {sid: i for i, sid in enumerate(all_songs)}
n_users = len(user2idx)
n_songs = len(song2idx)

train['user_idx'] = train['msno'].map(user2idx).astype(np.int32)
train['song_idx'] = train['song_id'].map(song2idx).astype(np.int32)
test['user_idx'] = test['msno'].map(user2idx).astype(np.int32)
test['song_idx'] = test['song_id'].map(song2idx).astype(np.int32)
log(f"  유저: {n_users:,}, 곡: {n_songs:,}")

# ================================================================
# 3. NCF 임베딩 학습 (딥러닝 파트)
# ================================================================
log("\n[3/7] NCF 임베딩 학습...")
log("  (유저/곡 ID → 의미 있는 벡터로 변환하는 과정)")

NCF_DIM = 32
NCF_EPOCHS = 2
NCF_BATCH = 16384

class SimpleNCF(nn.Module):
    """임베딩 추출용 경량 NCF"""
    def __init__(self, n_users, n_songs, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.song_emb = nn.Embedding(n_songs, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.song_emb.weight, std=0.01)

    def forward(self, users, songs):
        u = self.user_emb(users)
        s = self.song_emb(songs)
        x = torch.cat([u, s], dim=1)
        return self.mlp(x).squeeze()

class IDDataset(Dataset):
    def __init__(self, users, songs, targets):
        self.u = torch.LongTensor(users)
        self.s = torch.LongTensor(songs)
        self.t = torch.FloatTensor(targets)
    def __len__(self): return len(self.u)
    def __getitem__(self, i): return self.u[i], self.s[i], self.t[i]

# 전체 train으로 학습 (CV 없이, 임베딩 추출 목적)
ds = IDDataset(train['user_idx'].values, train['song_idx'].values, train['target'].values)
dl = DataLoader(ds, batch_size=NCF_BATCH, shuffle=True, num_workers=0)

model = SimpleNCF(n_users, n_songs, NCF_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(NCF_EPOCHS):
    model.train()
    total_loss = 0
    n_batch = 0
    t_ep = time.time()
    for bu, bs, bt in dl:
        optimizer.zero_grad()
        loss = criterion(model(bu, bs), bt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batch += 1
    log(f"  Epoch {epoch+1}/{NCF_EPOCHS}: loss={total_loss/n_batch:.4f} ({time.time()-t_ep:.0f}s)")

# 임베딩 추출
log("  임베딩 추출 중...")
user_vectors = model.user_emb.weight.detach().numpy()  # (n_users, 32)
song_vectors = model.song_emb.weight.detach().numpy()  # (n_songs, 32)
log(f"  유저 임베딩: {user_vectors.shape}, 곡 임베딩: {song_vectors.shape}")

# train/test에 NCF 임베딩 피처 추가
for i in range(NCF_DIM):
    train[f'ncf_user_{i}'] = user_vectors[train['user_idx'].values, i].astype('float32')
    test[f'ncf_user_{i}'] = user_vectors[test['user_idx'].values, i].astype('float32')
    train[f'ncf_song_{i}'] = song_vectors[train['song_idx'].values, i].astype('float32')
    test[f'ncf_song_{i}'] = song_vectors[test['song_idx'].values, i].astype('float32')

# NCF 궁합 점수 (유저 벡터 · 곡 벡터)
u_tr = user_vectors[train['user_idx'].values]
s_tr = song_vectors[train['song_idx'].values]
train['ncf_dot'] = np.sum(u_tr * s_tr, axis=1).astype('float32')
u_te = user_vectors[test['user_idx'].values]
s_te = song_vectors[test['song_idx'].values]
test['ncf_dot'] = np.sum(u_te * s_te, axis=1).astype('float32')

del model, ds, dl, u_tr, s_tr, u_te, s_te, user_vectors, song_vectors
gc.collect()
log("  NCF 임베딩 피처 추가 완료")

# ================================================================
# 4. LightGBM 피처 (v8과 동일)
# ================================================================
log("\n[4/7] LightGBM 피처 엔지니어링...")
t0 = time.time()

# Song features
extra['isrc_country'] = extra['isrc'].str[:2]
extra['isrc_year'] = pd.to_numeric(extra['isrc'].str[5:7], errors='coerce')
extra['isrc_year'] = extra['isrc_year'].apply(
    lambda x: 1900 + x if x > 30 else 2000 + x if pd.notna(x) else np.nan
).astype('float32')
extra.drop('isrc', axis=1, inplace=True)
songs = songs.merge(extra, on='song_id', how='left')
del extra; gc.collect()

songs['first_genre'] = pd.to_numeric(songs['genre_ids'].str.split('|').str[0], errors='coerce').astype('float32')
songs['genre_count'] = songs['genre_ids'].str.count(r'\|') + 1
songs.loc[songs['genre_ids'].isna(), 'genre_count'] = 0
songs['song_length_min'] = (songs['song_length'].fillna(0) / 60000.0).astype('float32')
songs.drop(['genre_ids', 'song_length'], axis=1, inplace=True)

# Member features
members['bd_clean'] = members['bd'].copy()
members.loc[(members['bd_clean'] <= 5) | (members['bd_clean'] >= 80), 'bd_clean'] = 0
members['bd_clean'] = members['bd_clean'].astype('int8')
members['gender'] = members['gender'].astype('category').cat.codes.astype('int8')
members['registration_init_time'] = pd.to_datetime(members['registration_init_time'], format='%Y%m%d', errors='coerce')
members['expiration_date'] = pd.to_datetime(members['expiration_date'], format='%Y%m%d', errors='coerce')
members['membership_days'] = (members['expiration_date'] - members['registration_init_time']).dt.days.astype('float32')
members['registration_year'] = members['registration_init_time'].dt.year.astype('float32')
members.drop(['bd', 'registration_init_time', 'expiration_date'], axis=1, inplace=True)

# Merge
train = train.merge(songs, on='song_id', how='left')
train = train.merge(members, on='msno', how='left')
test = test.merge(songs, on='song_id', how='left')
test = test.merge(members, on='msno', how='left')
del songs, members; gc.collect()

# Label encoding
for col in ['source_system_tab', 'source_screen_name', 'source_type', 'isrc_country']:
    if col in train.columns:
        combined = pd.concat([train[col], test[col]], ignore_index=True)
        codes, _ = pd.factorize(combined)
        train[col] = codes[:len(train)].astype('int16')
        test[col] = codes[len(train):].astype('int16')

combined = pd.concat([train['artist_name'], test['artist_name']])
codes, _ = pd.factorize(combined)
train['artist_code'] = codes[:len(train)].astype('int32')
test['artist_code'] = codes[len(train):].astype('int32')

for col in ['msno', 'song_id']:
    combined = pd.concat([train[col], test[col]])
    codes, _ = pd.factorize(combined)
    train[f'{col}_code'] = codes[:len(train)].astype('int32')
    test[f'{col}_code'] = codes[len(train):].astype('int32')

# Count features
def add_count(tr, te, col, prefix):
    counts = tr[col].value_counts().to_dict()
    tr[f'{prefix}_count'] = tr[col].map(counts).fillna(0).astype('int32')
    te[f'{prefix}_count'] = te[col].map(counts).fillna(0).astype('int32')
    return tr, te

train, test = add_count(train, test, 'msno_code', 'user')
train, test = add_count(train, test, 'song_id_code', 'song')
train, test = add_count(train, test, 'artist_code', 'artist')

# Ratio features
for grp, name in [(['msno_code', 'artist_code'], 'user_artist_count'),
                  (['msno_code', 'first_genre'], 'user_genre_count')]:
    cnt = train.groupby(grp).size().reset_index(name=name)
    train = train.merge(cnt, on=grp, how='left')
    test = test.merge(cnt, on=grp, how='left')
    test[name] = test[name].fillna(0).astype('int16')

for df in [train, test]:
    df['user_artist_ratio'] = (df['user_artist_count'] / (df['user_count'] + 1)).astype('float32')
    df['user_genre_ratio'] = (df['user_genre_count'] / (df['user_count'] + 1)).astype('float32')

log(f"  LightGBM 피처 완료 ({time.time()-t0:.1f}s)")

# ================================================================
# 5. SVD 임베딩 (기존 v8과 동일, 비교용)
# ================================================================
log("\n[5/7] SVD 임베딩 생성...")
t0 = time.time()

all_msno = pd.concat([train['msno_code'], test['msno_code']])
all_song = pd.concat([train['song_id_code'], test['song_id_code']])
n_u = all_msno.max() + 1
n_s = all_song.max() + 1

interaction = csr_matrix(
    (np.ones(len(all_msno), dtype=np.float32), (all_msno.values, all_song.values)),
    shape=(n_u, n_s)
)
interaction.data = np.minimum(interaction.data, 1.0)

svd = TruncatedSVD(n_components=32, random_state=SEED, n_iter=10)
user_factors = svd.fit_transform(interaction)
song_factors = svd.components_.T

for i in range(32):
    train[f'svd_user_{i}'] = user_factors[train['msno_code'].values, i].astype('float32')
    test[f'svd_user_{i}'] = user_factors[test['msno_code'].values, i].astype('float32')
    train[f'svd_song_{i}'] = song_factors[train['song_id_code'].values, i].astype('float32')
    test[f'svd_song_{i}'] = song_factors[test['song_id_code'].values, i].astype('float32')

u_tr = user_factors[train['msno_code'].values]
s_tr = song_factors[train['song_id_code'].values]
train['svd_dot'] = np.sum(u_tr * s_tr, axis=1).astype('float32')
u_te = user_factors[test['msno_code'].values]
s_te = song_factors[test['song_id_code'].values]
test['svd_dot'] = np.sum(u_te * s_te, axis=1).astype('float32')

del interaction, svd, user_factors, song_factors, u_tr, s_tr, u_te, s_te
gc.collect()
log(f"  SVD 완료 ({time.time()-t0:.1f}s)")

# ================================================================
# 6. LightGBM 학습 (NCF 임베딩 + SVD + 피처)
# ================================================================
log("\n[6/7] LightGBM 학습...")

drop_cols = ['msno', 'song_id', 'target', 'artist_name', 'id',
             'user_idx', 'song_idx', 'composer', 'lyricist']
feature_cols = [c for c in train.columns if c not in drop_cols]

ncf_count = len([c for c in feature_cols if 'ncf' in c])
svd_count = len([c for c in feature_cols if 'svd' in c])
other_count = len(feature_cols) - ncf_count - svd_count
log(f"  피처 총 {len(feature_cols)}개:")
log(f"    NCF 임베딩: {ncf_count}개 (딥러닝이 만든 것)")
log(f"    SVD 임베딩: {svd_count}개 (행렬분해가 만든 것)")
log(f"    기타 피처:  {other_count}개 (사람이 만든 것)")

X = train[feature_cols]
y = train['target']
X_test = test[feature_cols]

params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.1, 'num_leaves': 255, 'max_depth': -1,
    'min_child_samples': 100, 'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 1.0, 'reg_lambda': 5.0, 'n_estimators': 2000,
    'random_state': SEED, 'verbose': -1, 'n_jobs': -1,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_scores = []
feature_importance = pd.DataFrame()

for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y)):
    log(f"\n  Fold {fold+1}/5...")
    t_f = time.time()
    model = lgb.LGBMClassifier(**params)
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[vl_idx], y.iloc[vl_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(500)])

    val_pred = model.predict_proba(X.iloc[vl_idx])[:, 1]
    oof_preds[vl_idx] = val_pred
    fold_auc = roc_auc_score(y.iloc[vl_idx], val_pred)
    fold_scores.append(fold_auc)
    test_preds += model.predict_proba(X_test)[:, 1] / 5

    fi = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
    feature_importance = pd.concat([feature_importance, fi])

    log(f"  Fold {fold+1} AUC: {fold_auc:.6f} ({time.time()-t_f:.0f}s)")
    del model; gc.collect()

overall_auc = roc_auc_score(y, oof_preds)

# ================================================================
# 7. 결과
# ================================================================
log(f"\n{'='*65}")
log(f"  v3 결과: NCF Embedding + LightGBM Hybrid")
log(f"{'='*65}")
log(f"  OOF AUC: {overall_auc:.6f}")
log(f"  Fold별:  {[f'{s:.4f}' for s in fold_scores]}")
log(f"")
log(f"  비교 (Kaggle AUC):")
log(f"    v8  LightGBM + SVD only     : 0.70")
log(f"    v9  LightGBM + SVD tuned    : 0.71")
log(f"    NCF v2 단독                 : 0.67")
log(f"    v3  LightGBM + SVD + NCF    : OOF {overall_auc:.4f} (제출 필요)")

# 피처 중요도 Top 15
fi_avg = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
log(f"\n  피처 중요도 Top 15:")
for i, (feat, imp) in enumerate(fi_avg.head(15).items()):
    tag = ""
    if 'ncf' in feat: tag = " [NCF]"
    elif 'svd' in feat: tag = " [SVD]"
    log(f"    {i+1:2d}. {feat:30s} {imp:8.0f}{tag}")

ncf_total = fi_avg[fi_avg.index.str.contains('ncf')].sum()
svd_total = fi_avg[fi_avg.index.str.contains('svd')].sum()
log(f"\n  NCF 기여도: {ncf_total/fi_avg.sum()*100:.1f}%")
log(f"  SVD 기여도: {svd_total/fi_avg.sum()*100:.1f}%")
log(f"{'='*65}")

submission = pd.DataFrame({'id': test['id'], 'target': test_preds})
submission.to_csv('submission_v3_hybrid.csv', index=False)
log(f"\n  submission_v3_hybrid.csv 저장 완료")
