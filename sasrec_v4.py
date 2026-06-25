"""
KKBox v4 — SASRec (Transformer 시퀀스 추천) + LightGBM 하이브리드
================================================================
v3까지의 교훈:
  - NCF/SVD 임베딩은 "유저-곡 행렬"에서 같은 정보를 뽑음 → 중복
  - 진짜 새로운 정보 = "순서" (어떤 순서로 들었는가)
  - Transformer로 시퀀스를 이해 → 시퀀스 기반 유저 벡터 생성
  - 이 벡터를 LightGBM에 넣어서 일반화된 예측

SASRec 원리:
  GPT가 [단어1, 단어2, 단어3] → 다음 단어? 를 예측하듯
  SASRec는 [곡1, 곡2, 곡3] → 다음 곡 재청취? 를 예측
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

MAX_SEQ_LEN = 50     # 유저당 최근 50곡 사용
EMBED_DIM = 32
N_HEADS = 4
N_LAYERS = 2
SEQ_EPOCHS = 2
SEQ_BATCH = 2048
LR = 0.001

log("=" * 65)
log("  KKBox v4: SASRec + LightGBM Hybrid")
log("=" * 65)

# ================================================================
# 1. 데이터 로드
# ================================================================
log("\n[1/8] 데이터 로드...")
t0 = time.time()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
songs = pd.read_csv('songs.csv')
members = pd.read_csv('members.csv')
extra = pd.read_csv('song_extra_info.csv', usecols=['song_id', 'isrc'])
log(f"  train: {len(train):,}, test: {len(test):,} ({time.time()-t0:.1f}s)")

# ================================================================
# 2. ID 매핑 + 시퀀스 구축
# ================================================================
log("\n[2/8] 시퀀스 구축...")
t0 = time.time()

all_users = pd.concat([train['msno'], test['msno']]).unique()
all_songs = pd.concat([train['song_id'], test['song_id']]).unique()
user2idx = {uid: i for i, uid in enumerate(all_users)}
song2idx = {sid: i + 1 for i, sid in enumerate(all_songs)}  # 0은 padding
n_users = len(user2idx)
n_songs = len(song2idx) + 1  # +1 for padding

train['user_idx'] = train['msno'].map(user2idx).astype(np.int32)
train['song_idx'] = train['song_id'].map(song2idx).astype(np.int32)
test['user_idx'] = test['msno'].map(user2idx).astype(np.int32)
test['song_idx'] = test['song_id'].map(song2idx).astype(np.int32)

# 유저별 곡 시퀀스 구축 (행 순서 = 시간 순서)
log("  유저별 시퀀스 구축 중...")
user_histories = {}
for uid, group in train.groupby('user_idx'):
    user_histories[uid] = group['song_idx'].values

# 각 train 행에 대해 "이 시점까지의 시퀀스" 생성
log("  Train 시퀀스 생성 중...")
train_seqs = []
train_positions = {}  # user_idx → current position
for i in range(len(train)):
    uid = train.iloc[i]['user_idx']
    if uid not in train_positions:
        train_positions[uid] = 0
    pos = train_positions[uid]

    # 이 시점까지의 히스토리 (현재 곡 제외)
    history = user_histories[uid][:pos]
    # 최근 MAX_SEQ_LEN개만
    if len(history) > MAX_SEQ_LEN:
        history = history[-MAX_SEQ_LEN:]
    # padding
    padded = np.zeros(MAX_SEQ_LEN, dtype=np.int32)
    if len(history) > 0:
        padded[-len(history):] = history

    train_seqs.append(padded)
    train_positions[uid] = pos + 1

train_seqs = np.array(train_seqs)
log(f"  Train 시퀀스: {train_seqs.shape}")

# Test 시퀀스: 유저의 전체 train 히스토리 사용
log("  Test 시퀀스 생성 중...")
test_seqs = []
for i in range(len(test)):
    uid = test.iloc[i]['user_idx']
    history = user_histories.get(uid, np.array([], dtype=np.int32))
    if len(history) > MAX_SEQ_LEN:
        history = history[-MAX_SEQ_LEN:]
    padded = np.zeros(MAX_SEQ_LEN, dtype=np.int32)
    if len(history) > 0:
        padded[-len(history):] = history
    test_seqs.append(padded)

test_seqs = np.array(test_seqs)
log(f"  Test 시퀀스: {test_seqs.shape}")
log(f"  완료 ({time.time()-t0:.1f}s)")

del user_histories, train_positions; gc.collect()

# ================================================================
# 3. SASRec 모델 정의
# ================================================================
log("\n[3/8] SASRec 모델 정의...")

class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation
    = Transformer Decoder를 곡 시퀀스에 적용

    입력: [곡1, 곡2, ..., 곡50] (유저의 최근 청취 기록)
    출력: 유저의 현재 상태 벡터 (32차원)
    """
    def __init__(self, n_songs, embed_dim, n_heads, n_layers, max_len):
        super().__init__()
        self.song_emb = nn.Embedding(n_songs, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim*4,
            dropout=0.2, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # 예측: 시퀀스 상태 + 타겟 곡 → 재청취 확률
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.song_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def encode_sequence(self, seq):
        """시퀀스 → 유저 상태 벡터"""
        batch_size, seq_len = seq.shape

        # 곡 임베딩 + 위치 임베딩
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0)
        x = self.song_emb(seq) + self.pos_emb(positions)

        # 패딩 마스크 (0인 위치는 무시)
        padding_mask = (seq == 0)

        # Causal 마스크 (미래 곡을 보지 못하게)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()

        # Transformer
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)

        # 마지막 유효 위치의 출력 = 유저의 현재 상태
        # 간단하게 마지막 위치 사용
        user_state = x[:, -1, :]  # (batch, embed_dim)
        return user_state

    def forward(self, seq, target_song):
        user_state = self.encode_sequence(seq)       # (batch, embed_dim)
        song_vec = self.song_emb(target_song)        # (batch, embed_dim)
        combined = torch.cat([user_state, song_vec], dim=1)
        return self.predictor(combined).squeeze()


class SeqDataset(Dataset):
    def __init__(self, seqs, songs, targets=None):
        self.seqs = torch.LongTensor(seqs)
        self.songs = torch.LongTensor(songs)
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self): return len(self.seqs)

    def __getitem__(self, i):
        if self.targets is not None:
            return self.seqs[i], self.songs[i], self.targets[i]
        return self.seqs[i], self.songs[i]


# ================================================================
# 4. SASRec 학습
# ================================================================
log("\n[4/8] SASRec 학습...")
log(f"  Dim={EMBED_DIM}, Heads={N_HEADS}, Layers={N_LAYERS}, MaxLen={MAX_SEQ_LEN}")

ds = SeqDataset(train_seqs, train['song_idx'].values, train['target'].values)
dl = DataLoader(ds, batch_size=SEQ_BATCH, shuffle=True, num_workers=0)

model = SASRec(n_songs, EMBED_DIM, N_HEADS, N_LAYERS, MAX_SEQ_LEN).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

MODEL_PATH = 'sasrec_model.pt'
import os
if os.path.exists(MODEL_PATH):
    log("  저장된 SASRec 가중치 로드 중... (학습 생략)")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    for epoch in range(SEQ_EPOCHS):
        model.train()
        total_loss = 0
        n_batch = 0
        t_ep = time.time()
        for batch_seq, batch_song, batch_target in dl:
            optimizer.zero_grad()
            logits = model(batch_seq, batch_song)
            loss = criterion(logits, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
            if n_batch % 500 == 0:
                log(f"    batch {n_batch}/{len(dl)}: loss={total_loss/n_batch:.4f}")
        log(f"  Epoch {epoch+1}/{SEQ_EPOCHS}: loss={total_loss/n_batch:.4f} ({time.time()-t_ep:.0f}s)")
    torch.save(model.state_dict(), MODEL_PATH)
    log(f"  SASRec 모델 가중치 저장 완료: {MODEL_PATH}")

del ds, dl; gc.collect()

# ================================================================
# 5. 시퀀스 임베딩 추출
# ================================================================
log("\n[5/8] 시퀀스 임베딩 추출...")
t0 = time.time()

model.eval()

# 피처를 원활하게 할당하기 위해 Numpy 배열 사용
log("  Numpy 배열 할당 중...")
seq_user_train = np.zeros((len(train), EMBED_DIM), dtype=np.float32)
seq_dot_train = np.zeros(len(train), dtype=np.float32)
seq_user_test = np.zeros((len(test), EMBED_DIM), dtype=np.float32)
seq_dot_test = np.zeros(len(test), dtype=np.float32)

song_vectors = model.song_emb.weight.detach().numpy()
seq_song_train = song_vectors[train['song_idx'].values].astype(np.float32)
seq_song_test = song_vectors[test['song_idx'].values].astype(np.float32)

log("  Train 임베딩 추출 & 할당 중...")
with torch.no_grad():
    for i in range(0, len(train_seqs), SEQ_BATCH * 2):
        end = min(i + SEQ_BATCH * 2, len(train_seqs))
        batch = torch.LongTensor(train_seqs[i:end])
        embeds = model.encode_sequence(batch).numpy()
        
        seq_user_train[i:end] = embeds
        seq_dot_train[i:end] = np.sum(embeds * seq_song_train[i:end], axis=1)

# 메모리 해제
del train_seqs; gc.collect()

log("  Test 임베딩 추출 & 할당 중...")
with torch.no_grad():
    for i in range(0, len(test_seqs), SEQ_BATCH * 2):
        end = min(i + SEQ_BATCH * 2, len(test_seqs))
        batch = torch.LongTensor(test_seqs[i:end])
        embeds = model.encode_sequence(batch).numpy()
        
        seq_user_test[i:end] = embeds
        seq_dot_test[i:end] = np.sum(embeds * seq_song_test[i:end], axis=1)

del test_seqs; gc.collect()

log("  DataFrame 컬럼 생성...")
for i in range(EMBED_DIM):
    train[f'seq_user_{i}'] = seq_user_train[:, i]
    test[f'seq_user_{i}'] = seq_user_test[:, i]
    train[f'seq_song_{i}'] = seq_song_train[:, i]
    test[f'seq_song_{i}'] = seq_song_test[:, i]

train['seq_dot'] = seq_dot_train
test['seq_dot'] = seq_dot_test

del seq_user_train, seq_user_test, seq_song_train, seq_song_test, seq_dot_train, seq_dot_test, song_vectors; gc.collect()
log(f"  완료 ({time.time()-t0:.1f}s)")

# ================================================================
# 6. LightGBM 피처 (기존 v8 피처 + SVD)
# ================================================================
log("\n[6/8] LightGBM 피처...")
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

members['bd_clean'] = members['bd'].copy()
members.loc[(members['bd_clean'] <= 5) | (members['bd_clean'] >= 80), 'bd_clean'] = 0
members['gender'] = members['gender'].astype('category').cat.codes.astype('int8')
members['registration_init_time'] = pd.to_datetime(members['registration_init_time'], format='%Y%m%d', errors='coerce')
members['expiration_date'] = pd.to_datetime(members['expiration_date'], format='%Y%m%d', errors='coerce')
members['membership_days'] = (members['expiration_date'] - members['registration_init_time']).dt.days.astype('float32')
members['registration_year'] = members['registration_init_time'].dt.year.astype('float32')
members.drop(['bd', 'registration_init_time', 'expiration_date'], axis=1, inplace=True)

train = train.merge(songs, on='song_id', how='left')
train = train.merge(members, on='msno', how='left')
test = test.merge(songs, on='song_id', how='left')
test = test.merge(members, on='msno', how='left')
del songs, members; gc.collect()

for col in ['source_system_tab', 'source_screen_name', 'source_type', 'isrc_country']:
    if col in train.columns:
        combined = pd.concat([train[col], test[col]])
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

def add_count(tr, te, col, prefix):
    counts = tr[col].value_counts().to_dict()
    tr[f'{prefix}_count'] = tr[col].map(counts).fillna(0).astype('int32')
    te[f'{prefix}_count'] = te[col].map(counts).fillna(0).astype('int32')
    return tr, te

train, test = add_count(train, test, 'msno_code', 'user')
train, test = add_count(train, test, 'song_id_code', 'song')
train, test = add_count(train, test, 'artist_code', 'artist')

for grp, name in [(['msno_code', 'artist_code'], 'user_artist_count'),
                  (['msno_code', 'first_genre'], 'user_genre_count')]:
    cnt = train.groupby(grp).size().reset_index(name=name)
    train = train.merge(cnt, on=grp, how='left')
    test = test.merge(cnt, on=grp, how='left')
    test[name] = test[name].fillna(0).astype('int16')

for df in [train, test]:
    df['user_artist_ratio'] = (df['user_artist_count'] / (df['user_count'] + 1)).astype('float32')
    df['user_genre_ratio'] = (df['user_genre_count'] / (df['user_count'] + 1)).astype('float32')

# SVD
log("  SVD 임베딩...")
all_msno = pd.concat([train['msno_code'], test['msno_code']])
all_song = pd.concat([train['song_id_code'], test['song_id_code']])
interaction = csr_matrix(
    (np.ones(len(all_msno), dtype=np.float32), (all_msno.values, all_song.values)),
    shape=(all_msno.max()+1, all_song.max()+1)
)
interaction.data = np.minimum(interaction.data, 1.0)
svd = TruncatedSVD(n_components=32, random_state=SEED, n_iter=10)
uf = svd.fit_transform(interaction)
sf = svd.components_.T
for i in range(32):
    train[f'svd_user_{i}'] = uf[train['msno_code'].values, i].astype('float32')
    test[f'svd_user_{i}'] = uf[test['msno_code'].values, i].astype('float32')
    train[f'svd_song_{i}'] = sf[train['song_id_code'].values, i].astype('float32')
    test[f'svd_song_{i}'] = sf[test['song_id_code'].values, i].astype('float32')
train['svd_dot'] = np.sum(uf[train['msno_code'].values] * sf[train['song_id_code'].values], axis=1).astype('float32')
test['svd_dot'] = np.sum(uf[test['msno_code'].values] * sf[test['song_id_code'].values], axis=1).astype('float32')
del interaction, svd, uf, sf; gc.collect()

log(f"  완료 ({time.time()-t0:.1f}s)")

# ================================================================
# 7. LightGBM 학습
# ================================================================
log("\n[7/8] LightGBM 학습...")

drop_cols = ['msno', 'song_id', 'target', 'artist_name', 'id',
             'user_idx', 'song_idx', 'composer', 'lyricist']
feature_cols = [c for c in train.columns if c not in drop_cols]

seq_count = len([c for c in feature_cols if 'seq_' in c])
svd_count = len([c for c in feature_cols if 'svd_' in c])
other = len(feature_cols) - seq_count - svd_count
log(f"  피처 총 {len(feature_cols)}개:")
log(f"    SASRec 시퀀스: {seq_count}개 (Transformer가 만든 것)")
log(f"    SVD 임베딩:    {svd_count}개 (행렬분해)")
log(f"    기타 피처:     {other}개 (사람이 만든 것)")

X, y = train[feature_cols], train['target']
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
    m = lgb.LGBMClassifier(**params)
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
          eval_set=[(X.iloc[vl_idx], y.iloc[vl_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(500)])
    vp = m.predict_proba(X.iloc[vl_idx])[:, 1]
    oof_preds[vl_idx] = vp
    auc = roc_auc_score(y.iloc[vl_idx], vp)
    fold_scores.append(auc)
    test_preds += m.predict_proba(X_test)[:, 1] / 5
    fi = pd.DataFrame({'feature': feature_cols, 'importance': m.feature_importances_})
    feature_importance = pd.concat([feature_importance, fi])
    log(f"  Fold {fold+1} AUC: {auc:.6f} ({time.time()-t_f:.0f}s)")
    del m; gc.collect()

overall_auc = roc_auc_score(y, oof_preds)

# ================================================================
# 8. 결과
# ================================================================
log(f"\n{'='*65}")
log(f"  v4 결과: SASRec + LightGBM Hybrid")
log(f"{'='*65}")
log(f"  OOF AUC: {overall_auc:.6f}")
log(f"  Fold별:  {[f'{s:.4f}' for s in fold_scores]}")
log(f"")
log(f"  비교:")
log(f"    v9  LightGBM + SVD          : Kaggle 0.71")
log(f"    v3  LightGBM + SVD + NCF    : Kaggle ~0.71")
log(f"    v4  LightGBM + SVD + SASRec : OOF {overall_auc:.4f}")

fi_avg = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
log(f"\n  피처 중요도 Top 15:")
for i, (feat, imp) in enumerate(fi_avg.head(15).items()):
    tag = ""
    if 'seq_' in feat: tag = " [SASRec]"
    elif 'svd_' in feat: tag = " [SVD]"
    log(f"    {i+1:2d}. {feat:30s} {imp:8.0f}{tag}")

seq_total = fi_avg[fi_avg.index.str.contains('seq_')].sum()
svd_total = fi_avg[fi_avg.index.str.contains('svd_')].sum()
log(f"\n  SASRec 기여도: {seq_total/fi_avg.sum()*100:.1f}%")
log(f"  SVD 기여도:    {svd_total/fi_avg.sum()*100:.1f}%")
log(f"{'='*65}")

submission = pd.DataFrame({'id': test['id'], 'target': test_preds})
submission.to_csv('submission_v4_sasrec.csv', index=False)
log(f"\n  submission_v4_sasrec.csv 저장 완료")
