"""
KKBox NCF v2 — Side Feature 포함 + 전체 데이터
================================================
v1 문제점:
  - 100만 행 샘플 → test의 유저/곡을 못 봄 → Cold Start → Kaggle 0.61
  
v2 개선:
  - 전체 730만 행 학습
  - 유저/곡 ID 임베딩 + Side Feature(장르, 언어, 나이 등) 결합
  - Cold Start 유저도 Side Feature로 어느 정도 예측 가능
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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

EMBED_DIM = 32
MLP_DIMS = [128, 64]
EPOCHS = 3
BATCH_SIZE = 16384
LR = 0.001
N_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log("=" * 65)
log("  KKBox NCF v2 (Full Data + Side Features)")
log(f"  Device: {DEVICE}")
log("=" * 65)

# ================================================================
# 1. 데이터 로드 + 전처리
# ================================================================
log("\n[1/5] 데이터 로드...")
t0 = time.time()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
songs = pd.read_csv('songs.csv')
members = pd.read_csv('members.csv')
extra = pd.read_csv('song_extra_info.csv', usecols=['song_id', 'isrc'])

log(f"  train: {len(train):,}, test: {len(test):,} ({time.time()-t0:.1f}s)")

# ================================================================
# 2. 피처 엔지니어링
# ================================================================
log("\n[2/5] 피처 엔지니어링...")
t0 = time.time()

# --- Song features ---
songs['first_genre'] = pd.to_numeric(
    songs['genre_ids'].str.split('|').str[0], errors='coerce'
).fillna(-1).astype(int)
songs['genre_count'] = songs['genre_ids'].str.count(r'\|').fillna(0).astype(int) + 1
songs.loc[songs['genre_ids'].isna(), 'genre_count'] = 0
songs['song_length_min'] = (songs['song_length'].fillna(0) / 60000.0).astype('float32')
songs['language'] = songs['language'].fillna(-1).astype(int)

# artist code
artist_codes, artist_uniques = pd.factorize(songs['artist_name'])
songs['artist_code'] = artist_codes
n_artists = len(artist_uniques)

# ISRC 기반 연도/국가
extra['isrc_year'] = pd.to_numeric(extra['isrc'].str[5:7], errors='coerce')
extra['isrc_year'] = extra['isrc_year'].apply(
    lambda x: 1900 + x if x > 30 else 2000 + x if pd.notna(x) else 0
).fillna(0).astype(int)
songs = songs.merge(extra[['song_id', 'isrc_year']], on='song_id', how='left')
songs['isrc_year'] = songs['isrc_year'].fillna(0).astype(int)
del extra; gc.collect()

# first_genre factorize
genre_codes, _ = pd.factorize(songs['first_genre'])
songs['genre_code'] = genre_codes
n_genres = int(songs['genre_code'].max()) + 1

# language factorize
lang_codes, _ = pd.factorize(songs['language'])
songs['lang_code'] = lang_codes
n_langs = int(songs['lang_code'].max()) + 1

song_features = songs[['song_id', 'song_length_min', 'genre_code', 'lang_code',
                        'artist_code', 'genre_count', 'isrc_year']].copy()

# --- Member features ---
members['bd_clean'] = members['bd'].clip(0, 80).fillna(0).astype(int)
members.loc[(members['bd_clean'] <= 5) | (members['bd_clean'] >= 75), 'bd_clean'] = 0
members['gender_code'] = members['gender'].astype('category').cat.codes.astype(int) + 1  # 0=missing
members['registered_via'] = members['registered_via'].fillna(0).astype(int)
members['city'] = members['city'].fillna(0).astype(int)

member_features = members[['msno', 'bd_clean', 'gender_code', 'registered_via', 'city']].copy()

# --- Merge ---
train = train.merge(song_features, on='song_id', how='left')
train = train.merge(member_features, on='msno', how='left')
test = test.merge(song_features, on='song_id', how='left')
test = test.merge(member_features, on='msno', how='left')
del songs, members; gc.collect()

# --- Context features ---
for col in ['source_system_tab', 'source_screen_name', 'source_type']:
    combined = pd.concat([train[col], test[col]])
    codes, _ = pd.factorize(combined)
    train[col + '_code'] = codes[:len(train)]
    test[col + '_code'] = codes[len(train):]

n_source_tab = int(train['source_system_tab_code'].max()) + 1
n_source_screen = int(train['source_screen_name_code'].max()) + 1
n_source_type = int(train['source_type_code'].max()) + 1

# --- User/Song ID mapping ---
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

# Fill NaN
for col in ['song_length_min', 'genre_code', 'lang_code', 'artist_code',
            'genre_count', 'isrc_year', 'bd_clean', 'gender_code',
            'registered_via', 'city', 'source_system_tab_code',
            'source_screen_name_code', 'source_type_code']:
    train[col] = train[col].fillna(0).astype(np.int32 if col != 'song_length_min' else np.float32)
    test[col] = test[col].fillna(0).astype(np.int32 if col != 'song_length_min' else np.float32)

log(f"  유저: {n_users:,}, 곡: {n_songs:,}, 아티스트: {n_artists:,}")
log(f"  장르: {n_genres}, 언어: {n_langs}")
log(f"  완료 ({time.time()-t0:.1f}s)")

# ================================================================
# 3. 모델 정의
# ================================================================
log("\n[3/5] 모델 정의...")

class NeuMF_v2(nn.Module):
    """
    NeuMF + Side Features
    
    v1과의 차이:
      - 유저/곡 임베딩 외에 장르/언어/아티스트 등의 임베딩도 학습
      - 연속 피처(곡 길이, 나이)는 직접 MLP에 입력
      - Cold Start 유저/곡도 side feature로 어느 정도 예측 가능
    """
    def __init__(self, n_users, n_songs, n_artists, n_genres, n_langs,
                 n_src_tab, n_src_screen, n_src_type, embed_dim, mlp_dims):
        super().__init__()
        side_embed = 8  # side feature 임베딩 차원
        
        # GMF 임베딩
        self.gmf_user = nn.Embedding(n_users, embed_dim)
        self.gmf_song = nn.Embedding(n_songs, embed_dim)
        
        # MLP 유저/곡 임베딩
        self.mlp_user = nn.Embedding(n_users, embed_dim)
        self.mlp_song = nn.Embedding(n_songs, embed_dim)
        
        # Side Feature 임베딩
        self.emb_artist = nn.Embedding(n_artists + 2, side_embed, padding_idx=0)
        self.emb_genre = nn.Embedding(n_genres + 2, side_embed, padding_idx=0)
        self.emb_lang = nn.Embedding(n_langs + 2, side_embed, padding_idx=0)
        self.emb_src_tab = nn.Embedding(n_src_tab + 2, side_embed, padding_idx=0)
        self.emb_src_screen = nn.Embedding(n_src_screen + 2, side_embed, padding_idx=0)
        self.emb_src_type = nn.Embedding(n_src_type + 2, side_embed, padding_idx=0)
        self.emb_gender = nn.Embedding(10, side_embed, padding_idx=0)
        self.emb_city = nn.Embedding(50, side_embed, padding_idx=0)
        self.emb_reg_via = nn.Embedding(30, side_embed, padding_idx=0)
        
        # 각 임베딩의 크기 저장 (clamp용)
        self.n_artist = n_artists + 1
        self.n_genre = n_genres + 1
        self.n_lang = n_langs + 1
        self.n_src_tab = n_src_tab + 1
        self.n_src_screen = n_src_screen + 1
        self.n_src_type = n_src_type + 1
        
        # MLP 입력 크기:
        #   유저 embed + 곡 embed + side embeds(9개*8) + 연속피처(3개)
        n_side_embeds = 9 * side_embed  # artist, genre, lang, 3x source, gender, city, reg_via
        n_continuous = 3  # song_length_min, bd_clean, isrc_year
        mlp_input = embed_dim * 2 + n_side_embeds + n_continuous
        
        mlp_layers = []
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(mlp_input, dim))
            mlp_layers.append(nn.BatchNorm1d(dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.3))
            mlp_input = dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # 최종: GMF(embed_dim) + MLP(last dim) → 1
        self.output = nn.Linear(embed_dim + mlp_dims[-1], 1)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, user_ids, song_ids, artist, genre, lang,
                src_tab, src_screen, src_type, gender, city, reg_via,
                song_len, age, year):
        # GMF
        gmf_out = self.gmf_user(user_ids) * self.gmf_song(song_ids)
        
        # MLP: 임베딩 + side features + 연속 피처
        mlp_u = self.mlp_user(user_ids)
        mlp_s = self.mlp_song(song_ids)
        
        # clamp로 범위 초과 방지
        side = torch.cat([
            self.emb_artist(artist.clamp(0, self.n_artist)),
            self.emb_genre(genre.clamp(0, self.n_genre)),
            self.emb_lang(lang.clamp(0, self.n_lang)),
            self.emb_src_tab(src_tab.clamp(0, self.n_src_tab)),
            self.emb_src_screen(src_screen.clamp(0, self.n_src_screen)),
            self.emb_src_type(src_type.clamp(0, self.n_src_type)),
            self.emb_gender(gender.clamp(0, 9)),
            self.emb_city(city.clamp(0, 49)),
            self.emb_reg_via(reg_via.clamp(0, 29)),
        ], dim=1)
        
        continuous = torch.stack([song_len, age, year], dim=1)
        
        mlp_input = torch.cat([mlp_u, mlp_s, side, continuous], dim=1)
        mlp_out = self.mlp(mlp_input)
        
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        return self.output(combined).squeeze()


class KKBoxDataset_v2(Dataset):
    def __init__(self, df, has_target=True):
        self.user_ids = torch.LongTensor(df['user_idx'].values)
        self.song_ids = torch.LongTensor(df['song_idx'].values)
        self.artists = torch.LongTensor(df['artist_code'].values)
        self.genres = torch.LongTensor(df['genre_code'].values)
        self.langs = torch.LongTensor(df['lang_code'].values)
        self.src_tabs = torch.LongTensor(df['source_system_tab_code'].values)
        self.src_screens = torch.LongTensor(df['source_screen_name_code'].values)
        self.src_types = torch.LongTensor(df['source_type_code'].values)
        self.genders = torch.LongTensor(df['gender_code'].values.clip(0, 3))
        self.cities = torch.LongTensor(df['city'].values.clip(0, 24))
        self.reg_vias = torch.LongTensor(df['registered_via'].values.clip(0, 19))
        self.song_lens = torch.FloatTensor(df['song_length_min'].values)
        self.ages = torch.FloatTensor(df['bd_clean'].values.astype(float) / 80.0)
        self.years = torch.FloatTensor(
            ((df['isrc_year'].values.astype(float) - 1990) / 40.0).clip(-1, 1)
        )
        if has_target:
            self.targets = torch.FloatTensor(df['target'].values)
        else:
            self.targets = None
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        items = (self.user_ids[idx], self.song_ids[idx],
                 self.artists[idx], self.genres[idx], self.langs[idx],
                 self.src_tabs[idx], self.src_screens[idx], self.src_types[idx],
                 self.genders[idx], self.cities[idx], self.reg_vias[idx],
                 self.song_lens[idx], self.ages[idx], self.years[idx])
        if self.targets is not None:
            return items + (self.targets[idx],)
        return items


# ================================================================
# 4. 학습
# ================================================================
log("\n[4/5] 학습 시작 (전체 데이터)...")
log(f"  Data: {len(train):,}행")
log(f"  Embed: {EMBED_DIM}, MLP: {MLP_DIMS}, Batch: {BATCH_SIZE}")

y = train['target'].values
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_scores = []

test_ds = KKBoxDataset_v2(test, has_target=False)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
    log(f"\n  --- Fold {fold+1}/{N_FOLDS} ---")
    t_fold = time.time()
    
    train_ds = KKBoxDataset_v2(train.iloc[train_idx])
    val_ds = KKBoxDataset_v2(train.iloc[val_idx])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
    
    model = NeuMF_v2(n_users, n_songs, n_artists, n_genres, n_langs,
                     n_source_tab, n_source_screen, n_source_type,
                     EMBED_DIM, MLP_DIMS).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_auc = 0
    best_state = None
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss = 0
        n_batch = 0
        
        for batch in train_dl:
            *features, targets = [x.to(DEVICE) for x in batch]
            optimizer.zero_grad()
            logits = model(*features)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_preds = []
        val_tgts = []
        with torch.no_grad():
            for batch in val_dl:
                *features, targets = batch
                features = [x.to(DEVICE) for x in features]
                logits = model(*features)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_tgts.append(targets.numpy())
        
        val_preds = np.concatenate(val_preds)
        val_tgts = np.concatenate(val_tgts)
        val_auc = roc_auc_score(val_tgts, val_preds)
        
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        log(f"    Epoch {epoch+1}/{EPOCHS}: loss={total_loss/n_batch:.4f}, "
            f"AUC={val_auc:.6f}{'  *best*' if is_best else ''}")
    
    # Best model로 OOF + Test 예측
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        # OOF
        oof = []
        for batch in val_dl:
            *features, _ = batch
            features = [x.to(DEVICE) for x in features]
            oof.append(torch.sigmoid(model(*features)).cpu().numpy())
        oof_preds[val_idx] = np.concatenate(oof)
        
        # Test
        test_fold = []
        for batch in test_dl:
            features = [x.to(DEVICE) for x in batch]
            test_fold.append(torch.sigmoid(model(*features)).cpu().numpy())
        test_preds += np.concatenate(test_fold) / N_FOLDS
    
    fold_scores.append(best_auc)
    log(f"  Fold {fold+1} best: {best_auc:.6f} ({time.time()-t_fold:.0f}s)")
    del model, train_ds, val_ds, train_dl, val_dl; gc.collect()

# ================================================================
# 5. 결과
# ================================================================
overall_auc = roc_auc_score(y, oof_preds)

log(f"\n{'='*65}")
log(f"  NCF v2 결과 (Full Data + Side Features)")
log(f"{'='*65}")
log(f"  OOF AUC:  {overall_auc:.6f}")
log(f"  Fold별:   {[f'{s:.4f}' for s in fold_scores]}")
log(f"")
log(f"  비교:")
log(f"    LightGBM v9        Kaggle: 0.71")
log(f"    NCF v1 (ID only)   Kaggle: 0.61  (Cold Start 문제)")
log(f"    NCF v2 (+ Side)    OOF:    {overall_auc:.4f}")
log(f"{'='*65}")

submission = pd.DataFrame({'id': test['id'], 'target': test_preds})
submission.to_csv('submission_ncf_v2.csv', index=False)
log(f"\n  submission_ncf_v2.csv 저장 완료")
