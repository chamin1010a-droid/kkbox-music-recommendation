"""
KKBOX Music Recommendation Challenge - LightGBM Pipeline v7
============================================================
v6 실험 결과: ID 특성 유지(Model A)가 성능이 더 좋았고, 트리가 2000개에서도 
최대치에 도달(early stopping 안 됨).

v7 전략:
  - 사용자/곡 ID 피처 유지
  - 트리의 개수(n_estimators)를 5000개로 대폭 증가
  - 더 깊고 세밀한 학습을 위해 learning_rate를 0.1 → 0.05로 감소
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import gc
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

log("=" * 65)
log("  KKBOX Music Recommendation - LightGBM Pipeline v7")
log("  (n_estimators=5000, learning_rate=0.05, ID 포함)")
log("=" * 65)

# ================================================================
# 1. 데이터 로드
# ================================================================
log("\n[1/7] 데이터 로드 중...")
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

# ================================================================
# 2. 테이블 전처리 + 병합
# ================================================================
log("\n[2/7] 테이블 전처리 및 병합 중...")
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
log("\n[3/7] 범주형 인코딩 중...")
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
# 4. COUNT + 비율 피처 (target 미사용)
# ================================================================
log("\n[4/7] Count + 비율 피처 생성 중...")
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
log("\n[5/7] 다양성 + 시퀀스 피처 생성 중...")
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

log(f"  시퀀스 피처 완료 ({time.time()-t0:.1f}초)")

# ================================================================
# 6. 모델 학습
# ================================================================
log("\n[6/7] 모델 학습 준비...")

drop_cols = ['msno', 'song_id', 'target', 'artist_name']
test_id = test['id'].copy() if 'id' in test.columns else None
if 'id' in test.columns:
    drop_cols.append('id')

feature_cols = [c for c in train.columns if c not in drop_cols]
log(f"  사용 피처 ({len(feature_cols)}개)")

X = train[feature_cols].copy()
y = train['target'].copy()
X_test = test[feature_cols].copy()

del train, test; gc.collect()

# 트리를 크게 늘리고 learning_rate를 줄여서 미세하게 조정
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,        # 0.1 -> 0.05 로 줄여서 과적합 방지
    'num_leaves': 255,
    'max_depth': -1,
    'min_child_samples': 100,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 1.0,
    'reg_lambda': 5.0,
    'n_estimators': 5000,         # 2000 -> 5000 으로 크게 증가
    'random_state': SEED,
    'verbose': -1,
    'n_jobs': -1,
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_scores = []
feature_importance = pd.DataFrame()

log(f"\n[7/7] LightGBM 5-Fold 학습 시작 (n_estimators=5000, lr=0.05)")
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
            lgb.early_stopping(150, verbose=False), # early stopping 150으로 여유 줌
            lgb.log_evaluation(200)
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

log(f"\n{'='*50}")
log(f"  v7 최종 결과 (5000 trees, lr=0.05):")
log(f"  전체 OOF AUC: {overall_auc:.6f}")
log(f"  Fold별 AUC: {[f'{s:.6f}' for s in fold_scores]}")
log(f"  평균 AUC: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")
log(f"  학습 시간: {time.time()-t_start:.1f}초")

fi_avg = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
log(f"\n  [피처 중요도 Top 20]")
for i, (feat, imp) in enumerate(fi_avg.head(20).items()):
    bar = '#' * int(imp / fi_avg.max() * 30)
    log(f"    {i+1:2d}. {feat:30s} {imp:8.0f}  {bar}")

submission = pd.DataFrame({'id': test_id, 'target': test_preds})
submission.to_csv('submission_v7.csv', index=False)
log(f"\n  submission_v7.csv 저장 완료")

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 10))
top_fi = fi_avg.head(25)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_fi)))
ax.barh(top_fi.index[::-1], top_fi.values[::-1], color=colors)
ax.set_title(f'Feature Importance v7 (n_estimators=5000) — OOF AUC: {overall_auc:.6f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Average Importance')
plt.tight_layout()
plt.savefig('eda_output/15_feature_importance_v7.png', dpi=150)
plt.close()
log(f"  차트 저장: eda_output/15_feature_importance_v7.png")
log(f"{'='*50}")
