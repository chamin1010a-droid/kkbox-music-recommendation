"""
KKBOX Music Recommendation Challenge - LightGBM Pipeline v8
============================================================
v7까지의 교훈:
  - 트리를 늘려도 로컬 AUC만 상승, Kaggle은 0.65 고정
  - 모델 용량이 아니라 "피처의 일반화 능력"이 병목

v8 전략: 모델이 스스로 발견할 수 없는 "행 간 관계" 피처 추가
  1. SVD 임베딩: 유저-곡 상호작용 행렬 분해 → 유저/곡의 잠재 벡터
  2. 시퀀스 피처: 직전 곡과의 관계 (같은 아티스트? 같은 장르?)
  3. 트리는 2000개(lr=0.1)로 유지 — 피처 품질로 승부
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import time
import gc
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

SEED = 42
N_FOLDS = 5
SVD_COMPONENTS = 32  # 유저/곡 잠재벡터 차원
np.random.seed(SEED)

log("=" * 65)
log("  KKBOX Music Recommendation - LightGBM Pipeline v8")
log("  (SVD 임베딩 + 시퀀스 피처 추가)")
log("=" * 65)

# ================================================================
# 1. 데이터 로드
# ================================================================
log("\n[1/8] 데이터 로드 중...")
t0 = time.time()

train = pd.read_csv('train.csv', dtype={
    'msno': 'category', 'song_id': 'category',
    'source_system_tab': 'category', 'source_screen_name': 'category',
    'source_type': 'category', 'target': 'int8'
})
test = pd.read_csv('test.csv', dtype={
    'msno': 'category', 'song_id': 'category',
    'source_system_tab': 'category', 'source_screen_name': 'category',
    'source_type': 'category'
})

songs = pd.read_csv('songs.csv', dtype={
    'song_id': 'category', 'song_length': 'int32',
    'genre_ids': str, 'artist_name': 'category',
    'composer': str, 'lyricist': str,
    'language': 'float32'
}, usecols=['song_id', 'song_length', 'genre_ids', 'artist_name',
            'composer', 'lyricist', 'language'])

members = pd.read_csv('members.csv', dtype={
    'msno': 'category', 'city': 'int8', 'bd': 'int16',
    'registered_via': 'int8'
})

extra = pd.read_csv('song_extra_info.csv', dtype={'song_id': 'category'},
                     usecols=['song_id', 'isrc'])

log(f"  로드 완료 ({time.time()-t0:.1f}초)")
log(f"  Train: {len(train):,} / Test: {len(test):,}")

# ================================================================
# 2. 테이블 전처리 + 병합
# ================================================================
log("\n[2/8] 테이블 전처리 및 병합 중...")
t0 = time.time()

extra['isrc_country'] = extra['isrc'].str[:2]
extra['isrc_year'] = pd.to_numeric(extra['isrc'].str[5:7], errors='coerce')
extra['isrc_year'] = extra['isrc_year'].apply(
    lambda x: 1900 + x if x > 30 else 2000 + x if pd.notna(x) else np.nan
).astype('float32')
extra.drop('isrc', axis=1, inplace=True)

songs = songs.merge(extra, on='song_id', how='left')
del extra; gc.collect()

songs['first_genre'] = songs['genre_ids'].str.split('|').str[0]
songs['first_genre'] = pd.to_numeric(songs['first_genre'], errors='coerce').astype('float32')
songs['genre_count'] = songs['genre_ids'].str.count(r'\|') + 1
songs.loc[songs['genre_ids'].isna(), 'genre_count'] = 0
songs['genre_count'] = songs['genre_count'].astype('int8')

songs['artist_is_composer'] = (
    songs['artist_name'].astype(str).str.lower().str.strip() ==
    songs['composer'].astype(str).str.lower().str.strip()
).astype('int8')
songs['artist_is_lyricist'] = (
    songs['artist_name'].astype(str).str.lower().str.strip() ==
    songs['lyricist'].astype(str).str.lower().str.strip()
).astype('int8')
songs['composer_count'] = songs['composer'].str.count(r'\|') + 1
songs.loc[songs['composer'].isna(), 'composer_count'] = 0
songs['composer_count'] = songs['composer_count'].astype('int8')
songs['lyricist_count'] = songs['lyricist'].str.count(r'\|') + 1
songs.loc[songs['lyricist'].isna(), 'lyricist_count'] = 0
songs['lyricist_count'] = songs['lyricist_count'].astype('int8')
songs['has_composer'] = songs['composer'].notna().astype('int8')
songs['has_lyricist'] = songs['lyricist'].notna().astype('int8')

songs.drop(['genre_ids', 'composer', 'lyricist'], axis=1, inplace=True)

songs['song_length_min'] = (songs['song_length'] / 60000.0).astype('float32')
songs.drop('song_length', axis=1, inplace=True)

members['bd_clean'] = members['bd'].copy()
members.loc[(members['bd_clean'] <= 5) | (members['bd_clean'] >= 80), 'bd_clean'] = 0
members['bd_clean'] = members['bd_clean'].astype('int8')

members['registration_init_time'] = pd.to_datetime(
    members['registration_init_time'], format='%Y%m%d', errors='coerce')
members['expiration_date'] = pd.to_datetime(
    members['expiration_date'], format='%Y%m%d', errors='coerce')
members['membership_days'] = (
    members['expiration_date'] - members['registration_init_time']
).dt.days.astype('float32')
members['registration_year'] = members['registration_init_time'].dt.year.astype('float32')
members.drop(['bd', 'registration_init_time', 'expiration_date'], axis=1, inplace=True)
members['gender'] = members['gender'].astype('category').cat.codes.astype('int8')

train = train.merge(songs, on='song_id', how='left')
train = train.merge(members, on='msno', how='left')
test = test.merge(songs, on='song_id', how='left')
test = test.merge(members, on='msno', how='left')

del songs, members; gc.collect()
log(f"  병합 완료 ({time.time()-t0:.1f}초)")

# ================================================================
# 3. 범주형 인코딩
# ================================================================
log("\n[3/8] 범주형 인코딩 중...")
t0 = time.time()

for col in ['source_system_tab', 'source_screen_name', 'source_type', 'isrc_country']:
    if col in train.columns:
        combined = pd.concat([train[col], test[col]], ignore_index=True)
        codes, _ = pd.factorize(combined)
        train[col] = codes[:len(train)].astype('int16')
        test[col] = codes[len(train):].astype('int16')
        del combined, codes; gc.collect()

combined = pd.concat([train['artist_name'], test['artist_name']], ignore_index=True)
codes, _ = pd.factorize(combined)
train['artist_code'] = codes[:len(train)].astype('int32')
test['artist_code'] = codes[len(train):].astype('int32')
del combined, codes; gc.collect()

for col in ['msno', 'song_id']:
    combined = pd.concat([train[col], test[col]], ignore_index=True)
    codes, _ = pd.factorize(combined)
    train[f'{col}_code'] = codes[:len(train)].astype('int32')
    test[f'{col}_code'] = codes[len(train):].astype('int32')
    del combined, codes; gc.collect()

log(f"  인코딩 완료 ({time.time()-t0:.1f}초)")

# ================================================================
# 4. COUNT + 비율 피처
# ================================================================
log("\n[4/8] Count + 비율 피처 생성 중...")
t0 = time.time()

def add_count_only(train_df, test_df, col, prefix):
    counts = train_df[col].value_counts().to_dict()
    train_df[f'{prefix}_count'] = train_df[col].map(counts).fillna(0).astype('int32')
    test_df[f'{prefix}_count'] = test_df[col].map(counts).fillna(0).astype('int32')
    return train_df, test_df

train, test = add_count_only(train, test, 'msno_code', 'user')
train, test = add_count_only(train, test, 'song_id_code', 'song')
train, test = add_count_only(train, test, 'artist_code', 'artist')
train, test = add_count_only(train, test, 'first_genre', 'genre_stat')
train, test = add_count_only(train, test, 'language', 'lang')

ua_cnt = train.groupby(['msno_code', 'artist_code']).size().reset_index(name='user_artist_count')
train = train.merge(ua_cnt, on=['msno_code', 'artist_code'], how='left')
test = test.merge(ua_cnt, on=['msno_code', 'artist_code'], how='left')
test['user_artist_count'] = test['user_artist_count'].fillna(0).astype('int16')
del ua_cnt; gc.collect()

ug_cnt = train.groupby(['msno_code', 'first_genre']).size().reset_index(name='user_genre_count')
train = train.merge(ug_cnt, on=['msno_code', 'first_genre'], how='left')
test = test.merge(ug_cnt, on=['msno_code', 'first_genre'], how='left')
test['user_genre_count'] = test['user_genre_count'].fillna(0).astype('int16')
del ug_cnt; gc.collect()

ul_cnt = train.groupby(['msno_code', 'language']).size().reset_index(name='user_lang_count')
train = train.merge(ul_cnt, on=['msno_code', 'language'], how='left')
test = test.merge(ul_cnt, on=['msno_code', 'language'], how='left')
test['user_lang_count'] = test['user_lang_count'].fillna(0).astype('int16')
del ul_cnt; gc.collect()

us_cnt = train.groupby(['msno_code', 'song_id_code']).size().reset_index(name='user_song_count')
train = train.merge(us_cnt, on=['msno_code', 'song_id_code'], how='left')
test = test.merge(us_cnt, on=['msno_code', 'song_id_code'], how='left')
test['user_song_count'] = test['user_song_count'].fillna(0).astype('int16')
del us_cnt; gc.collect()

for df in [train, test]:
    df['user_artist_ratio'] = (df['user_artist_count'] / (df['user_count'] + 1)).astype('float32')
    df['user_genre_ratio'] = (df['user_genre_count'] / (df['user_count'] + 1)).astype('float32')
    df['user_lang_ratio'] = (df['user_lang_count'] / (df['user_count'] + 1)).astype('float32')
    df['song_artist_ratio'] = (df['song_count'] / (df['artist_count'] + 1)).astype('float32')
    df['user_song_ratio'] = (df['user_song_count'] / (df['user_count'] + 1)).astype('float32')
    df['song_per_unique_user'] = (df['song_count'] / (df['song_count'].clip(lower=1))).astype('float32')

log(f"  Count + 비율 피처 완료 ({time.time()-t0:.1f}초)")

# ================================================================
# 5. 다양성 + 시퀀스 피처
# ================================================================
log("\n[5/8] 다양성 + 시퀀스 피처 생성 중...")
t0 = time.time()

user_nunique_artist = train.groupby('msno_code')['artist_code'].nunique().reset_index(name='user_unique_artists')
train = train.merge(user_nunique_artist, on='msno_code', how='left')
test = test.merge(user_nunique_artist, on='msno_code', how='left')
test['user_unique_artists'] = test['user_unique_artists'].fillna(0).astype('int32')
del user_nunique_artist; gc.collect()

user_nunique_genre = train.groupby('msno_code')['first_genre'].nunique().reset_index(name='user_unique_genres')
train = train.merge(user_nunique_genre, on='msno_code', how='left')
test = test.merge(user_nunique_genre, on='msno_code', how='left')
test['user_unique_genres'] = test['user_unique_genres'].fillna(0).astype('int32')
del user_nunique_genre; gc.collect()

song_nunique_user = train.groupby('song_id_code')['msno_code'].nunique().reset_index(name='song_unique_users')
train = train.merge(song_nunique_user, on='song_id_code', how='left')
test = test.merge(song_nunique_user, on='song_id_code', how='left')
test['song_unique_users'] = test['song_unique_users'].fillna(0).astype('int32')
del song_nunique_user; gc.collect()

artist_nunique_user = train.groupby('artist_code')['msno_code'].nunique().reset_index(name='artist_unique_users')
train = train.merge(artist_nunique_user, on='artist_code', how='left')
test = test.merge(artist_nunique_user, on='artist_code', how='left')
test['artist_unique_users'] = test['artist_unique_users'].fillna(0).astype('int32')
del artist_nunique_user; gc.collect()

user_nunique_song = train.groupby('msno_code')['song_id_code'].nunique().reset_index(name='user_unique_songs')
train = train.merge(user_nunique_song, on='msno_code', how='left')
test = test.merge(user_nunique_song, on='msno_code', how='left')
test['user_unique_songs'] = test['user_unique_songs'].fillna(0).astype('int32')
del user_nunique_song; gc.collect()

user_nunique_lang = train.groupby('msno_code')['language'].nunique().reset_index(name='user_unique_langs')
train = train.merge(user_nunique_lang, on='msno_code', how='left')
test = test.merge(user_nunique_lang, on='msno_code', how='left')
test['user_unique_langs'] = test['user_unique_langs'].fillna(0).astype('int32')
del user_nunique_lang; gc.collect()

for df in [train, test]:
    df['song_listens_per_user'] = (df['song_count'] / (df['song_unique_users'] + 1)).astype('float32')
    df['user_diversity'] = (df['user_unique_artists'] / (df['user_count'] + 1)).astype('float32')
    df['user_repeat_ratio'] = (1 - df['user_unique_songs'] / (df['user_count'] + 1)).astype('float32')

train['user_listen_order'] = train.groupby('msno_code').cumcount().astype('int32')
train['user_listen_position'] = (train['user_listen_order'] / (train['user_count'])).astype('float32')

test['user_listen_order'] = test.groupby('msno_code').cumcount().astype('int32')
test_user_offset = train.groupby('msno_code').size().reset_index(name='_train_count')
test = test.merge(test_user_offset, on='msno_code', how='left')
test['_train_count'] = test['_train_count'].fillna(0)
test['user_listen_order'] = (test['user_listen_order'] + test['_train_count']).astype('int32')
test['user_listen_position'] = (test['user_listen_order'] / (test['user_count'] + test['_train_count'] + 1)).astype('float32')
test.drop('_train_count', axis=1, inplace=True)
del test_user_offset; gc.collect()

n_train = len(train)
song_first = train.groupby('song_id_code').apply(lambda x: x.index.min()).reset_index(name='song_first_appear')
song_last = train.groupby('song_id_code').apply(lambda x: x.index.max()).reset_index(name='song_last_appear')
song_pos = song_first.merge(song_last, on='song_id_code')
song_pos['song_first_appear'] = (song_pos['song_first_appear'] / n_train).astype('float32')
song_pos['song_last_appear'] = (song_pos['song_last_appear'] / n_train).astype('float32')
song_pos['song_span'] = (song_pos['song_last_appear'] - song_pos['song_first_appear']).astype('float32')
train = train.merge(song_pos, on='song_id_code', how='left')
test = test.merge(song_pos, on='song_id_code', how='left')
for col in ['song_first_appear', 'song_last_appear', 'song_span']:
    test[col] = test[col].fillna(0.5).astype('float32')
del song_first, song_last, song_pos; gc.collect()

artist_first = train.groupby('artist_code').apply(lambda x: x.index.min()).reset_index(name='artist_first_appear')
artist_last = train.groupby('artist_code').apply(lambda x: x.index.max()).reset_index(name='artist_last_appear')
artist_pos = artist_first.merge(artist_last, on='artist_code')
artist_pos['artist_first_appear'] = (artist_pos['artist_first_appear'] / n_train).astype('float32')
artist_pos['artist_last_appear'] = (artist_pos['artist_last_appear'] / n_train).astype('float32')
train = train.merge(artist_pos, on='artist_code', how='left')
test = test.merge(artist_pos, on='artist_code', how='left')
for col in ['artist_first_appear', 'artist_last_appear']:
    test[col] = test[col].fillna(0.5).astype('float32')
del artist_first, artist_last, artist_pos; gc.collect()

train['row_position'] = (np.arange(n_train) / n_train).astype('float32')
n_test = len(test)
test['row_position'] = ((np.arange(n_test) / n_test) * 0.1 + 1.0).astype('float32')

log(f"  다양성 + 위치 피처 완료 ({time.time()-t0:.1f}초)")

# ================================================================
# 6. ★ 신규: SVD 임베딩 (유저-곡 상호작용 행렬 분해)
# ================================================================
log(f"\n[6/8] SVD 임베딩 생성 중 ({SVD_COMPONENTS}차원)...")
t0 = time.time()

# train + test 모든 유저-곡 조합을 사용 (비지도 학습이므로 전체 사용 가능)
all_msno = pd.concat([train['msno_code'], test['msno_code']], ignore_index=True)
all_song = pd.concat([train['song_id_code'], test['song_id_code']], ignore_index=True)

n_users = all_msno.max() + 1
n_songs = all_song.max() + 1
log(f"  유저 수: {n_users:,}, 곡 수: {n_songs:,}")

# 유저-곡 상호작용 행렬 (binary: 들었으면 1)
log("  유저-곡 상호작용 행렬 생성...")
interaction = csr_matrix(
    (np.ones(len(all_msno), dtype=np.float32), (all_msno.values, all_song.values)),
    shape=(n_users, n_songs)
)
# 중복 제거 (같은 유저-곡 조합은 1로 통일)
interaction.data = np.minimum(interaction.data, 1.0)
log(f"  행렬 크기: {interaction.shape}, 비영 원소: {interaction.nnz:,}")

del all_msno, all_song; gc.collect()

# SVD 분해
log(f"  TruncatedSVD({SVD_COMPONENTS}) 학습 중...")
svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED, n_iter=10)
user_factors = svd.fit_transform(interaction)  # (n_users, SVD_COMPONENTS)
log(f"  설명 분산: {svd.explained_variance_ratio_.sum():.4f}")

# 곡 임베딩: V^T (SVD의 components)
song_factors = svd.components_.T  # (n_songs, SVD_COMPONENTS)

log(f"  유저 임베딩: {user_factors.shape}, 곡 임베딩: {song_factors.shape}")

# train/test에 유저 SVD 피처 추가
for i in range(SVD_COMPONENTS):
    col_name = f'user_svd_{i}'
    train[col_name] = user_factors[train['msno_code'].values, i].astype('float32')
    test[col_name] = user_factors[test['msno_code'].values, i].astype('float32')

# train/test에 곡 SVD 피처 추가
for i in range(SVD_COMPONENTS):
    col_name = f'song_svd_{i}'
    train_song_codes = train['song_id_code'].values
    test_song_codes = test['song_id_code'].values
    
    train[col_name] = song_factors[train_song_codes, i].astype('float32')
    test[col_name] = song_factors[test_song_codes, i].astype('float32')

# ★ 유저-곡 SVD 내적 (궁합 점수)
log("  유저-곡 SVD 궁합 점수 계산 중...")
user_vecs_train = user_factors[train['msno_code'].values]  # (n_train, 32)
song_vecs_train = song_factors[train['song_id_code'].values]  # (n_train, 32)
train['svd_dot'] = np.sum(user_vecs_train * song_vecs_train, axis=1).astype('float32')

user_vecs_test = user_factors[test['msno_code'].values]
song_vecs_test = song_factors[test['song_id_code'].values]
test['svd_dot'] = np.sum(user_vecs_test * song_vecs_test, axis=1).astype('float32')

del user_factors, song_factors, interaction, svd
del user_vecs_train, song_vecs_train, user_vecs_test, song_vecs_test
gc.collect()

log(f"  SVD 임베딩 완료 ({time.time()-t0:.1f}초)")

# ================================================================
# 7. ★ 신규: 시퀀스 컨텍스트 피처 (직전 곡과의 관계)
# ================================================================
log("\n[7/8] 시퀀스 컨텍스트 피처 생성 중...")
t0 = time.time()

# Train: 직전 행의 정보
for col, new_name in [('artist_code', 'prev_artist'), 
                       ('first_genre', 'prev_genre'),
                       ('language', 'prev_language'),
                       ('source_type', 'prev_source_type')]:
    train[new_name] = train.groupby('msno_code')[col].shift(1)
    test[new_name] = test.groupby('msno_code')[col].shift(1)

# 직전 곡과 같은 아티스트/장르/언어/소스인지
train['same_artist_as_prev'] = (train['artist_code'] == train['prev_artist']).astype('int8')
train['same_genre_as_prev'] = (train['first_genre'] == train['prev_genre']).astype('int8')
train['same_lang_as_prev'] = (train['language'] == train['prev_language']).astype('int8')
train['same_source_as_prev'] = (train['source_type'] == train['prev_source_type']).astype('int8')

test['same_artist_as_prev'] = (test['artist_code'] == test['prev_artist']).astype('int8')
test['same_genre_as_prev'] = (test['first_genre'] == test['prev_genre']).astype('int8')
test['same_lang_as_prev'] = (test['language'] == test['prev_language']).astype('int8')
test['same_source_as_prev'] = (test['source_type'] == test['prev_source_type']).astype('int8')

# NaN 처리 (첫 번째 행은 직전 곡이 없으므로)
for col in ['same_artist_as_prev', 'same_genre_as_prev', 
            'same_lang_as_prev', 'same_source_as_prev']:
    train[col] = train[col].fillna(-1).astype('int8')
    test[col] = test[col].fillna(-1).astype('int8')

# 직전 곡의 raw 값은 제거 (이미 같은지 여부만 남김)
train.drop(['prev_artist', 'prev_genre', 'prev_language', 'prev_source_type'], axis=1, inplace=True)
test.drop(['prev_artist', 'prev_genre', 'prev_language', 'prev_source_type'], axis=1, inplace=True)

log(f"  시퀀스 컨텍스트 피처 완료 ({time.time()-t0:.1f}초)")

# ================================================================
# 8. 모델 학습
# ================================================================
log("\n[8/8] 모델 학습 준비...")

drop_cols = ['msno', 'song_id', 'target', 'artist_name']
test_id = test['id'].copy() if 'id' in test.columns else None
if 'id' in test.columns:
    drop_cols.append('id')

feature_cols = [c for c in train.columns if c not in drop_cols]
log(f"  사용 피처 ({len(feature_cols)}개)")

# SVD 피처 수 출력
svd_feats = [c for c in feature_cols if 'svd' in c]
seq_feats = [c for c in feature_cols if 'prev' in c or 'same_' in c]
log(f"  - 기존 피처: {len(feature_cols) - len(svd_feats) - len(seq_feats)}개")
log(f"  - SVD 임베딩 피처: {len(svd_feats)}개")
log(f"  - 시퀀스 피처: {len(seq_feats)}개")

X = train[feature_cols].copy()
y = train['target'].copy()
X_test = test[feature_cols].copy()

del train, test; gc.collect()
log(f"  X shape: {X.shape}, 메모리: {X.memory_usage(deep=True).sum()/1e9:.2f}GB")

# 트리 수는 2000으로 유지 (피처 품질로 승부)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 255,
    'max_depth': -1,
    'min_child_samples': 100,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 1.0,
    'reg_lambda': 5.0,
    'n_estimators': 2000,
    'random_state': SEED,
    'verbose': -1,
    'n_jobs': -1,
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_scores = []
feature_importance = pd.DataFrame()

log(f"\n  LightGBM 5-Fold 학습 시작 (SVD + 시퀀스)")
t_start = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    log(f"\n  --- Fold {fold+1}/{N_FOLDS} ---")
    
    X_tr = X.iloc[train_idx]
    X_vl = X.iloc[val_idx]
    y_tr = y.iloc[train_idx]
    y_vl = y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100)
        ]
    )
    
    val_pred = model.predict_proba(X_vl)[:, 1]
    oof_preds[val_idx] = val_pred
    fold_auc = roc_auc_score(y_vl, val_pred)
    fold_scores.append(fold_auc)
    log(f"  Fold {fold+1} AUC: {fold_auc:.6f} (best iter: {model.best_iteration_})")
    
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_,
        'fold': fold + 1
    })
    feature_importance = pd.concat([feature_importance, fi], ignore_index=True)
    
    del X_tr, X_vl, y_tr, y_vl, val_pred
    gc.collect()

overall_auc = roc_auc_score(y, oof_preds)

log(f"\n{'='*65}")
log(f"  v8 최종 결과 (SVD {SVD_COMPONENTS}차원 + 시퀀스 피처):")
log(f"  전체 OOF AUC: {overall_auc:.6f}")
log(f"  Fold별 AUC: {[f'{s:.6f}' for s in fold_scores]}")
log(f"  평균 AUC: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")
log(f"  학습 시간: {time.time()-t_start:.1f}초")

# 피처 중요도
fi_avg = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
log(f"\n  [피처 중요도 Top 25]")
for i, (feat, imp) in enumerate(fi_avg.head(25).items()):
    bar = '#' * int(imp / fi_avg.max() * 30)
    marker = ' ★' if 'svd' in feat or 'same_' in feat else ''
    log(f"    {i+1:2d}. {feat:30s} {imp:8.0f}  {bar}{marker}")

# SVD 피처들의 중요도 합산
svd_total = fi_avg[fi_avg.index.str.contains('svd')].sum()
all_total = fi_avg.sum()
log(f"\n  SVD 피처 기여도: {svd_total/all_total*100:.1f}% ({len(svd_feats)}개 피처)")

seq_total = fi_avg[fi_avg.index.str.contains('same_')].sum()
log(f"  시퀀스 피처 기여도: {seq_total/all_total*100:.1f}% ({len(seq_feats)}개 피처)")

# 제출 파일 저장
submission = pd.DataFrame({'id': test_id, 'target': test_preds})
submission.to_csv('submission_v8.csv', index=False)
log(f"\n  submission_v8.csv 저장 완료 ({len(submission):,}행)")
log(f"  예측값 분포: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

# 차트 저장
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 12))
top_fi = fi_avg.head(30)
colors = []
for feat in top_fi.index:
    if 'svd' in feat:
        colors.append('#FF6B6B')  # SVD 피처: 빨간색
    elif 'same_' in feat:
        colors.append('#4ECDC4')  # 시퀀스 피처: 청록
    else:
        colors.append('#45B7D1')  # 기존 피처: 파란색
colors = colors[::-1]

ax.barh(top_fi.index[::-1], top_fi.values[::-1], color=colors)
ax.set_title(f'Feature Importance v8 (SVD+Seq) — OOF AUC: {overall_auc:.6f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Average Importance')

# 범례
from matplotlib.patches import Patch
legend_items = [
    Patch(facecolor='#FF6B6B', label='SVD 임베딩'),
    Patch(facecolor='#4ECDC4', label='시퀀스 피처'),
    Patch(facecolor='#45B7D1', label='기존 피처'),
]
ax.legend(handles=legend_items, loc='lower right')

plt.tight_layout()
plt.savefig('eda_output/16_feature_importance_v8.png', dpi=150)
plt.close()
log(f"  차트 저장: eda_output/16_feature_importance_v8.png")

log(f"\n{'='*65}")
log(f"  v8 파이프라인 완료!")
log(f"  비교: v6a OOF={0.8519:.4f} → v8 OOF={overall_auc:.4f}")
log(f"  제출 파일: submission_v8.csv")
log(f"{'='*65}")
