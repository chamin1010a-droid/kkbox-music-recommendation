"""
KKBOX Music Recommendation Challenge - LightGBM Pipeline v3
============================================================
v2의 Target Leakage 문제 수정
핵심 변경: Out-of-Fold Target Encoding + Smoothing
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

# ================================================================
# 0. 설정
# ================================================================
SEED = 42
N_FOLDS = 5
SMOOTH_ALPHA = 10  # 스무딩 강도: 작을수록 원본, 클수록 전체 평균 쪽으로
np.random.seed(SEED)

log("=" * 65)
log("  KKBOX Music Recommendation - LightGBM Pipeline v3")
log("  (Target Leakage 수정 버전)")
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
    'language': 'float32'
}, usecols=['song_id', 'song_length', 'genre_ids', 'artist_name', 'language'])

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
log("\n[2/7] 테이블 전처리 및 병합 중...")
t0 = time.time()

# ISRC → 국가, 연도
extra['isrc_country'] = extra['isrc'].str[:2]
extra['isrc_year'] = pd.to_numeric(extra['isrc'].str[5:7], errors='coerce')
extra['isrc_year'] = extra['isrc_year'].apply(
    lambda x: 1900 + x if x > 30 else 2000 + x if pd.notna(x) else np.nan
).astype('float32')
extra.drop('isrc', axis=1, inplace=True)

songs = songs.merge(extra, on='song_id', how='left')
del extra; gc.collect()

# 장르 처리
songs['first_genre'] = songs['genre_ids'].str.split('|').str[0]
songs['first_genre'] = pd.to_numeric(songs['first_genre'], errors='coerce').astype('float32')
songs['genre_count'] = songs['genre_ids'].str.count(r'\|') + 1
songs.loc[songs['genre_ids'].isna(), 'genre_count'] = 0
songs['genre_count'] = songs['genre_count'].astype('int8')
songs.drop('genre_ids', axis=1, inplace=True)

songs['song_length_min'] = (songs['song_length'] / 60000.0).astype('float32')
songs.drop('song_length', axis=1, inplace=True)

# Members 전처리
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

# 병합
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
# 4. COUNT 피처 (leakage 없음 - target 사용 안 함)
# ================================================================
log("\n[4/7] Count 피처 생성 중...")
t0 = time.time()

def add_count_only(train_df, test_df, col, prefix):
    """Count만 계산 (target 미사용 → leakage 없음)"""
    counts = train_df[col].value_counts().to_dict()
    train_df[f'{prefix}_count'] = train_df[col].map(counts).fillna(0).astype('int32')
    test_df[f'{prefix}_count'] = test_df[col].map(counts).fillna(0).astype('int32')
    return train_df, test_df

# 사용자, 곡, 아티스트, 장르, 언어별 등장 횟수
train, test = add_count_only(train, test, 'msno_code', 'user')
train, test = add_count_only(train, test, 'song_id_code', 'song')
train, test = add_count_only(train, test, 'artist_code', 'artist')
train, test = add_count_only(train, test, 'first_genre', 'genre_stat')
train, test = add_count_only(train, test, 'language', 'lang')

log("  단순 count 피처 완료")

# 교차 count (사용자 × 아티스트, 사용자 × 장르)
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

# 사용자 × 언어
ul_cnt = train.groupby(['msno_code', 'language']).size().reset_index(name='user_lang_count')
train = train.merge(ul_cnt, on=['msno_code', 'language'], how='left')
test = test.merge(ul_cnt, on=['msno_code', 'language'], how='left')
test['user_lang_count'] = test['user_lang_count'].fillna(0).astype('int16')
del ul_cnt; gc.collect()

# 비율 피처
for df in [train, test]:
    df['user_artist_ratio'] = (df['user_artist_count'] / (df['user_count'] + 1)).astype('float32')
    df['user_genre_ratio'] = (df['user_genre_count'] / (df['user_count'] + 1)).astype('float32')
    df['user_lang_ratio'] = (df['user_lang_count'] / (df['user_count'] + 1)).astype('float32')
    # 곡의 상대적 인기도 (해당 아티스트 곡들 중에서)
    df['song_artist_ratio'] = (df['song_count'] / (df['artist_count'] + 1)).astype('float32')

log(f"  Count + 비율 피처 완료 ({time.time()-t0:.1f}초)")

# ================================================================
# 5. 사용자별 고유 아티스트/장르 수 (다양성 피처)
# ================================================================
log("\n[5/7] 다양성 + 추가 피처 생성 중...")
t0 = time.time()

# 사용자별 고유 아티스트 수
user_nunique_artist = train.groupby('msno_code')['artist_code'].nunique().reset_index(name='user_unique_artists')
train = train.merge(user_nunique_artist, on='msno_code', how='left')
test = test.merge(user_nunique_artist, on='msno_code', how='left')
test['user_unique_artists'] = test['user_unique_artists'].fillna(0).astype('int32')
del user_nunique_artist; gc.collect()

# 사용자별 고유 장르 수
user_nunique_genre = train.groupby('msno_code')['first_genre'].nunique().reset_index(name='user_unique_genres')
train = train.merge(user_nunique_genre, on='msno_code', how='left')
test = test.merge(user_nunique_genre, on='msno_code', how='left')
test['user_unique_genres'] = test['user_unique_genres'].fillna(0).astype('int32')
del user_nunique_genre; gc.collect()

# 곡별 고유 청취자 수
song_nunique_user = train.groupby('song_id_code')['msno_code'].nunique().reset_index(name='song_unique_users')
train = train.merge(song_nunique_user, on='song_id_code', how='left')
test = test.merge(song_nunique_user, on='song_id_code', how='left')
test['song_unique_users'] = test['song_unique_users'].fillna(0).astype('int32')
del song_nunique_user; gc.collect()

# 아티스트별 고유 청취자 수
artist_nunique_user = train.groupby('artist_code')['msno_code'].nunique().reset_index(name='artist_unique_users')
train = train.merge(artist_nunique_user, on='artist_code', how='left')
test = test.merge(artist_nunique_user, on='artist_code', how='left')
test['artist_unique_users'] = test['artist_unique_users'].fillna(0).astype('int32')
del artist_nunique_user; gc.collect()

# 곡당 평균 청취 횟수 (song_count / song_unique_users)
for df in [train, test]:
    df['song_listens_per_user'] = (df['song_count'] / (df['song_unique_users'] + 1)).astype('float32')
    # 사용자 다양성 비율 (많은 아티스트를 듣는지, 소수만 듣는지)
    df['user_diversity'] = (df['user_unique_artists'] / (df['user_count'] + 1)).astype('float32')

log(f"  다양성 피처 완료 ({time.time()-t0:.1f}초)")

# ================================================================
# 6. 피처 선택 + 학습
# ================================================================
log("\n[6/7] 피처 선택 + 모델 학습 중...")

drop_cols = ['msno', 'song_id', 'target', 'artist_name']
test_id = test['id'].copy() if 'id' in test.columns else None
if 'id' in test.columns:
    drop_cols.append('id')

feature_cols = [c for c in train.columns if c not in drop_cols]
log(f"  사용 피처 ({len(feature_cols)}개):")
for i, col in enumerate(feature_cols):
    log(f"    {i+1:2d}. {col}")

X = train[feature_cols].copy()
y = train['target'].copy()
X_test = test[feature_cols].copy()

del train, test; gc.collect()
log(f"  X shape: {X.shape}, 메모리: {X.memory_usage(deep=True).sum()/1e9:.2f}GB")

# ================================================================
# 7. LightGBM 학습 (5-Fold CV) + OOF Target Encoding
# ================================================================
log(f"\n[7/7] LightGBM {N_FOLDS}-Fold CV 학습 중...")
log(f"       (OOF Target Encoding 포함)")
t0 = time.time()

GLOBAL_MEAN = y.mean()

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,         # v2보다 낮춤 (과적합 방지)
    'num_leaves': 127,
    'max_depth': -1,
    'min_child_samples': 200,      # v2보다 높임
    'subsample': 0.7,              # v2보다 낮춤
    'colsample_bytree': 0.7,       # v2보다 낮춤
    'reg_alpha': 1.0,              # v2보다 높임
    'reg_lambda': 5.0,             # v2보다 높임
    'n_estimators': 1500,          # 더 많은 라운드, lr 낮추니까
    'random_state': SEED,
    'verbose': -1,
    'n_jobs': -1,
}

# target encoding에 사용할 그룹 컬럼들
te_groups = {
    'msno_code': 'user_te',
    'song_id_code': 'song_te',
    'artist_code': 'artist_te',
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_scores = []
feature_importance = pd.DataFrame()

# Test용 target encoding (train 전체에서 계산 — leakage 없음)
for group_col, te_name in te_groups.items():
    stats = y.groupby(X[group_col]).agg(['sum', 'count'])
    smoothed = (stats['sum'] + SMOOTH_ALPHA * GLOBAL_MEAN) / (stats['count'] + SMOOTH_ALPHA)
    X_test[te_name] = X_test[group_col].map(smoothed).fillna(GLOBAL_MEAN).astype('float32')

# 교차 target encoding도 test용
ua_key = X[['msno_code', 'artist_code']].copy()
ua_key['target'] = y
ua_stats = ua_key.groupby(['msno_code', 'artist_code'])['target'].agg(['sum', 'count'])
ua_smoothed = (ua_stats['sum'] + SMOOTH_ALPHA * GLOBAL_MEAN) / (ua_stats['count'] + SMOOTH_ALPHA)
ua_smoothed = ua_smoothed.reset_index(name='user_artist_te')
X_test = X_test.merge(
    ua_smoothed[['msno_code', 'artist_code', 'user_artist_te']],
    on=['msno_code', 'artist_code'], how='left'
)
X_test['user_artist_te'] = X_test['user_artist_te'].fillna(GLOBAL_MEAN).astype('float32')
del ua_key, ua_stats, ua_smoothed; gc.collect()

# OOF target encoding용 빈 컬럼
for te_name in list(te_groups.values()) + ['user_artist_te']:
    X[te_name] = np.float32(0.0)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    log(f"\n  --- Fold {fold+1}/{N_FOLDS} ---")
    
    # ★ OOF Target Encoding: train fold의 target만 사용하여 val fold의 TE 계산
    y_train_fold = y.iloc[train_idx]
    X_train_fold = X.iloc[train_idx]
    
    for group_col, te_name in te_groups.items():
        stats = y_train_fold.groupby(X_train_fold[group_col]).agg(['sum', 'count'])
        smoothed = (stats['sum'] + SMOOTH_ALPHA * GLOBAL_MEAN) / (stats['count'] + SMOOTH_ALPHA)
        
        # val fold에 적용
        X.loc[val_idx, te_name] = X.loc[val_idx, group_col].map(smoothed).fillna(GLOBAL_MEAN).values.astype('float32')
        # train fold에도 적용 (자기 fold의 통계)
        X.loc[train_idx, te_name] = X.loc[train_idx, group_col].map(smoothed).fillna(GLOBAL_MEAN).values.astype('float32')
    
    # 교차 OOF TE (사용자 × 아티스트)
    ua_fold = X_train_fold[['msno_code', 'artist_code']].copy()
    ua_fold['target'] = y_train_fold.values
    ua_stats = ua_fold.groupby(['msno_code', 'artist_code'])['target'].agg(['sum', 'count'])
    ua_s = (ua_stats['sum'] + SMOOTH_ALPHA * GLOBAL_MEAN) / (ua_stats['count'] + SMOOTH_ALPHA)
    ua_s = ua_s.reset_index(name='user_artist_te')
    
    for idx_set in [train_idx, val_idx]:
        temp = X.loc[idx_set, ['msno_code', 'artist_code']].merge(
            ua_s[['msno_code', 'artist_code', 'user_artist_te']], 
            on=['msno_code', 'artist_code'], how='left'
        )
        X.loc[idx_set, 'user_artist_te'] = temp['user_artist_te'].fillna(GLOBAL_MEAN).values.astype('float32')
    
    del ua_fold, ua_stats, ua_s, temp; gc.collect()
    
    # 학습 feature 목록 (TE 포함)
    all_features = feature_cols + list(te_groups.values()) + ['user_artist_te']
    
    X_tr = X.iloc[train_idx][all_features]
    X_vl = X.iloc[val_idx][all_features]
    y_tr = y.iloc[train_idx]
    y_vl = y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(200)
        ]
    )
    
    val_pred = model.predict_proba(X_vl)[:, 1]
    oof_preds[val_idx] = val_pred
    fold_auc = roc_auc_score(y_vl, val_pred)
    fold_scores.append(fold_auc)
    log(f"  Fold {fold+1} AUC: {fold_auc:.6f} (best iter: {model.best_iteration_})")
    
    # Test 예측
    test_preds += model.predict_proba(X_test[all_features])[:, 1] / N_FOLDS
    
    fi = pd.DataFrame({
        'feature': all_features,
        'importance': model.feature_importances_,
        'fold': fold + 1
    })
    feature_importance = pd.concat([feature_importance, fi], ignore_index=True)
    
    del X_tr, X_vl, y_tr, y_vl, val_pred
    gc.collect()

overall_auc = roc_auc_score(y, oof_preds)
log(f"\n  {'='*50}")
log(f"  전체 OOF AUC: {overall_auc:.6f}")
log(f"  Fold별 AUC: {[f'{s:.6f}' for s in fold_scores]}")
log(f"  평균 AUC: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")
log(f"  학습 완료 ({time.time()-t0:.1f}초)")

# ================================================================
# 결과 저장
# ================================================================
log(f"\n  결과 저장 중...")

fi_avg = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
log("\n  [피처 중요도 Top 15]")
for i, (feat, imp) in enumerate(fi_avg.head(15).items()):
    bar = '#' * int(imp / fi_avg.max() * 30)
    log(f"    {i+1:2d}. {feat:30s} {imp:8.0f}  {bar}")

submission = pd.DataFrame({'id': test_id, 'target': test_preds})
submission.to_csv('submission_v3.csv', index=False)
log(f"\n  submission_v3.csv 저장 완료 ({len(submission):,}행)")
log(f"  예측값 분포: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
top_fi = fi_avg.head(20)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_fi)))
ax.barh(top_fi.index[::-1], top_fi.values[::-1], color=colors)
ax.set_title(f'Feature Importance v3 (Top 20) | OOF AUC: {overall_auc:.6f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Average Importance')
plt.tight_layout()
plt.savefig('eda_output/08_feature_importance_v3.png', dpi=150)
plt.close()

log(f"\n{'='*65}")
log(f"  v3 파이프라인 완료!")
log(f"  OOF AUC: {overall_auc:.6f}")
log(f"  (v2 OOF 0.9597 → 커닝, 이번이 진짜 실력)")
log(f"  제출 파일: submission_v3.csv")
log(f"{'='*65}")
