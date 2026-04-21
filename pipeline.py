"""
KKBOX Music Recommendation Challenge - LightGBM Pipeline v2
============================================================
메모리 최적화 버전 + 골디락스 이론 기반 피처 엔지니어링
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import gc
import sys
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

# ================================================================
# 0. 설정
# ================================================================
SEED = 42
N_FOLDS = 5
np.random.seed(SEED)

log("=" * 65)
log("  KKBOX Music Recommendation - LightGBM Pipeline v2")
log("=" * 65)

# ================================================================
# 1. 데이터 로드 (메모리 절약 dtype)
# ================================================================
log("\n[1/6] 데이터 로드 중...")
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
# 2. ISRC에서 국가/연도 추출 후 songs에 병합
# ================================================================
log("\n[2/6] 테이블 전처리 및 병합 중...")
t0 = time.time()

# ISRC → 국가, 연도
extra['isrc_country'] = extra['isrc'].str[:2].astype('category')
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

# 곡 길이 → 분  
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

# gender → 숫자
members['gender'] = members['gender'].astype('category').cat.codes.astype('int8')

# 병합
train = train.merge(songs, on='song_id', how='left')
train = train.merge(members, on='msno', how='left')
test = test.merge(songs, on='song_id', how='left')
test = test.merge(members, on='msno', how='left')

del songs, members; gc.collect()
log(f"  병합 완료 ({time.time()-t0:.1f}초)")

# ================================================================
# 3. 통계 피처 (골디락스 핵심)
# ================================================================
log("\n[3/6] 통계 피처 엔지니어링 중...")
t0 = time.time()

# 범주형 → codes (factorize 사용: NaN은 -1로, LightGBM이 네이티브 처리)
for col in ['source_system_tab', 'source_screen_name', 'source_type', 'isrc_country']:
    if col in train.columns:
        combined = pd.concat([train[col], test[col]], ignore_index=True)
        codes, uniques = pd.factorize(combined)
        train[col] = codes[:len(train)].astype('int16')
        test[col] = codes[len(train):].astype('int16')
        del combined, codes, uniques; gc.collect()

# artist_name 코드화
combined = pd.concat([train['artist_name'], test['artist_name']], ignore_index=True)
codes, _ = pd.factorize(combined)
train['artist_code'] = codes[:len(train)].astype('int32')
test['artist_code'] = codes[len(train):].astype('int32')
del combined, codes; gc.collect()

# msno, song_id 코드화
for col in ['msno', 'song_id']:
    combined = pd.concat([train[col], test[col]], ignore_index=True)
    codes, _ = pd.factorize(combined)
    train[f'{col}_code'] = codes[:len(train)].astype('int32')
    test[f'{col}_code'] = codes[len(train):].astype('int32')
    del combined, codes; gc.collect()

log("  범주형 인코딩 완료")

def add_stats(train_df, test_df, col, prefix):
    """그룹별 count + target 평균"""
    stats = train_df.groupby(col)['target'].agg(['count', 'mean'])
    stats.columns = [f'{prefix}_count', f'{prefix}_target_mean']
    stats = stats.reset_index()
    
    train_df = train_df.merge(stats, on=col, how='left')
    test_df = test_df.merge(stats, on=col, how='left')
    
    # NaN 처리 (groupby 키에 NaN이 있으면 merge 후 NaN 발생)
    for df in [train_df, test_df]:
        df[f'{prefix}_count'] = df[f'{prefix}_count'].fillna(0).astype('int32')
        df[f'{prefix}_target_mean'] = df[f'{prefix}_target_mean'].fillna(0.5).astype('float32')
    
    return train_df, test_df

# 사용자별 통계
log("  사용자별 통계...")
train, test = add_stats(train, test, 'msno_code', 'user')

# 곡별 통계
log("  곡별 통계...")
train, test = add_stats(train, test, 'song_id_code', 'song')

# 아티스트별 통계
log("  아티스트별 통계...")
train, test = add_stats(train, test, 'artist_code', 'artist')

# 장르별 통계
log("  장르별 통계...")
train, test = add_stats(train, test, 'first_genre', 'genre_stat')

# 언어별 통계
log("  언어별 통계...")
train, test = add_stats(train, test, 'language', 'lang')

# ----------------------------------------------------------
# 교차 통계 (사용자 × 아티스트, 사용자 × 장르)
# ----------------------------------------------------------
log("  교차 통계 (사용자x아티스트)...")

ua_stats = train.groupby(['msno_code', 'artist_code'])['target'].agg(['count', 'mean'])
ua_stats.columns = ['user_artist_count', 'user_artist_target_mean']
ua_stats = ua_stats.reset_index()
ua_stats['user_artist_count'] = ua_stats['user_artist_count'].astype('int16')
ua_stats['user_artist_target_mean'] = ua_stats['user_artist_target_mean'].astype('float32')

train = train.merge(ua_stats, on=['msno_code', 'artist_code'], how='left')
test = test.merge(ua_stats, on=['msno_code', 'artist_code'], how='left')
test['user_artist_count'] = test['user_artist_count'].fillna(0).astype('int16')
test['user_artist_target_mean'] = test['user_artist_target_mean'].fillna(0.5).astype('float32')
del ua_stats; gc.collect()

log("  교차 통계 (사용자x장르)...")

ug_stats = train.groupby(['msno_code', 'first_genre'])['target'].agg(['count', 'mean'])
ug_stats.columns = ['user_genre_count', 'user_genre_target_mean']
ug_stats = ug_stats.reset_index()
ug_stats['user_genre_count'] = ug_stats['user_genre_count'].astype('int16')
ug_stats['user_genre_target_mean'] = ug_stats['user_genre_target_mean'].astype('float32')

train = train.merge(ug_stats, on=['msno_code', 'first_genre'], how='left')
test = test.merge(ug_stats, on=['msno_code', 'first_genre'], how='left')
test['user_genre_count'] = test['user_genre_count'].fillna(0).astype('int16')
test['user_genre_target_mean'] = test['user_genre_target_mean'].fillna(0.5).astype('float32')
del ug_stats; gc.collect()

# 비율 피처
log("  비율 피처 생성...")
for df in [train, test]:
    df['user_artist_ratio'] = (df['user_artist_count'] / (df['user_count'] + 1)).astype('float32')
    df['user_genre_ratio'] = (df['user_genre_count'] / (df['user_count'] + 1)).astype('float32')

log(f"  피처 엔지니어링 완료 ({time.time()-t0:.1f}초)")

# ================================================================
# 4. 최종 피처 선택
# ================================================================
log("\n[4/6] 피처 선택 중...")

# 제거할 컬럼
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

log(f"  X shape: {X.shape}, X_test shape: {X_test.shape}")
log(f"  메모리: X={X.memory_usage(deep=True).sum()/1e9:.2f}GB")

# ================================================================
# 5. LightGBM 학습 (5-Fold CV)
# ================================================================
log(f"\n[5/6] LightGBM {N_FOLDS}-Fold CV 학습 중...")
t0 = time.time()

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 127,
    'max_depth': -1,
    'min_child_samples': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'n_estimators': 1000,
    'random_state': SEED,
    'verbose': -1,
    'n_jobs': -1,
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_scores = []
feature_importance = pd.DataFrame()

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    log(f"\n  --- Fold {fold+1}/{N_FOLDS} ---")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(200)
        ]
    )
    
    # OOF 예측
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    fold_auc = roc_auc_score(y_val, val_pred)
    fold_scores.append(fold_auc)
    log(f"  Fold {fold+1} AUC: {fold_auc:.6f} (best iter: {model.best_iteration_})")
    
    # Test 예측 (평균)
    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
    
    # Feature importance
    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_,
        'fold': fold + 1
    })
    feature_importance = pd.concat([feature_importance, fi], ignore_index=True)
    
    del X_train, X_val, y_train, y_val, val_pred
    gc.collect()

# 전체 OOF AUC
overall_auc = roc_auc_score(y, oof_preds)
log(f"\n  {'='*50}")
log(f"  전체 OOF AUC: {overall_auc:.6f}")
log(f"  Fold별 AUC: {[f'{s:.6f}' for s in fold_scores]}")
log(f"  평균 AUC: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")
log(f"  학습 완료 ({time.time()-t0:.1f}초)")

# ================================================================
# 6. 결과 저장
# ================================================================
log(f"\n[6/6] 결과 저장 중...")

# Feature importance
fi_avg = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
log("\n  [피처 중요도 Top 15]")
for i, (feat, imp) in enumerate(fi_avg.head(15).items()):
    bar = '#' * int(imp / fi_avg.max() * 30)
    log(f"    {i+1:2d}. {feat:30s} {imp:8.0f}  {bar}")

# 제출 파일
submission = pd.DataFrame({'id': test_id, 'target': test_preds})
submission.to_csv('submission.csv', index=False)
log(f"\n  submission.csv 저장 완료 ({len(submission):,}행)")
log(f"  예측값 분포: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

# Feature importance 차트
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
top_fi = fi_avg.head(20)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_fi)))
ax.barh(top_fi.index[::-1], top_fi.values[::-1], color=colors)
ax.set_title(f'Feature Importance (Top 20) | OOF AUC: {overall_auc:.6f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Average Importance')
plt.tight_layout()
plt.savefig('eda_output/07_feature_importance.png', dpi=150)
plt.close()
log("  07_feature_importance.png 저장 완료")

log(f"\n{'='*65}")
log(f"  파이프라인 완료!")
log(f"  OOF AUC: {overall_auc:.6f}")
log(f"  제출 파일: submission.csv")
log(f"{'='*65}")
