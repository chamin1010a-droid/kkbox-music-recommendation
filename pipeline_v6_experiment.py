"""
KKBOX Music Recommendation Challenge - v6 실험
================================================
두 가지 모델을 한 번의 데이터 로드로 연속 실행:

  Model A (v6a): n_estimators=2000, ID 포함 (v5b와 동일)
    → 질문 개수 1000→2000 효과 측정

  Model B (v6b): n_estimators=2000, ID 제거 (msno_code, song_id_code 제거)
    → 과적합 방지 효과 측정

비교 대상: v5 (n_estimators=1000, ID 포함) → Kaggle 0.65
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
log("  KKBOX Music Recommendation - v6 실험")
log("  Model A: n_estimators=2000 + ID 포함")
log("  Model B: n_estimators=2000 + ID 제거 (과적합 방지)")
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

# Composer/Lyricist 관계 피처
log("  Composer/Lyricist 관계 피처 생성...")
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

# 교차 count
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

# 사용자 × 곡 교차 count
us_cnt = train.groupby(['msno_code', 'song_id_code']).size().reset_index(name='user_song_count')
train = train.merge(us_cnt, on=['msno_code', 'song_id_code'], how='left')
test = test.merge(us_cnt, on=['msno_code', 'song_id_code'], how='left')
test['user_song_count'] = test['user_song_count'].fillna(0).astype('int16')
del us_cnt; gc.collect()

# 비율 피처
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

# 다양성 피처
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

log("  다양성 피처 완료")

# 시퀀스/위치 피처
log("  시퀀스/위치 피처 생성 중...")

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

# 곡별 시퀀스
n_train = len(train)
song_first = train.groupby('song_id_code').apply(
    lambda x: x.index.min()
).reset_index(name='song_first_appear')
song_last = train.groupby('song_id_code').apply(
    lambda x: x.index.max()
).reset_index(name='song_last_appear')

song_pos = song_first.merge(song_last, on='song_id_code')
song_pos['song_first_appear'] = (song_pos['song_first_appear'] / n_train).astype('float32')
song_pos['song_last_appear'] = (song_pos['song_last_appear'] / n_train).astype('float32')
song_pos['song_span'] = (song_pos['song_last_appear'] - song_pos['song_first_appear']).astype('float32')

train = train.merge(song_pos, on='song_id_code', how='left')
test = test.merge(song_pos, on='song_id_code', how='left')
for col in ['song_first_appear', 'song_last_appear', 'song_span']:
    test[col] = test[col].fillna(0.5).astype('float32')
del song_first, song_last, song_pos; gc.collect()

# 아티스트별 시퀀스
artist_first = train.groupby('artist_code').apply(
    lambda x: x.index.min()
).reset_index(name='artist_first_appear')
artist_last = train.groupby('artist_code').apply(
    lambda x: x.index.max()
).reset_index(name='artist_last_appear')

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
# 6. 공통 피처 준비
# ================================================================
log("\n[6/7] 공통 피처 준비...")

drop_cols_base = ['msno', 'song_id', 'target', 'artist_name']
test_id = test['id'].copy() if 'id' in test.columns else None
if 'id' in test.columns:
    drop_cols_base.append('id')

# Model A용 피처 (ID 포함)
feature_cols_a = [c for c in train.columns if c not in drop_cols_base]

# Model B용 피처 (ID 제거: msno_code, song_id_code 제거)
id_cols = ['msno_code', 'song_id_code']
feature_cols_b = [c for c in feature_cols_a if c not in id_cols]

log(f"  Model A 피처 ({len(feature_cols_a)}개): ID 포함 (msno_code, song_id_code)")
log(f"  Model B 피처 ({len(feature_cols_b)}개): ID 제거")
log(f"  제거된 피처: {id_cols}")

y = train['target'].copy()

# ================================================================
# LightGBM 공통 파라미터
# ================================================================
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

def run_model(name, feature_cols, train_df, test_df, y, params, submission_name, chart_name):
    """하나의 모델 실험을 실행하고 결과를 반환"""
    log(f"\n{'#'*65}")
    log(f"  {name} 학습 시작")
    log(f"  피처 {len(feature_cols)}개, n_estimators={params['n_estimators']}")
    log(f"{'#'*65}")
    
    t_start = time.time()
    
    X = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    log(f"  X shape: {X.shape}, 메모리: {X.memory_usage(deep=True).sum()/1e9:.2f}GB")
    log(f"  사용 피처:")
    for i, col in enumerate(feature_cols):
        log(f"    {i+1:2d}. {col}")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    feature_importance = pd.DataFrame()
    
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
    
    log(f"\n  {'='*50}")
    log(f"  {name} 결과:")
    log(f"  전체 OOF AUC: {overall_auc:.6f}")
    log(f"  Fold별 AUC: {[f'{s:.6f}' for s in fold_scores]}")
    log(f"  평균 AUC: {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")
    log(f"  학습 시간: {time.time()-t_start:.1f}초")
    
    # 피처 중요도
    fi_avg = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
    log(f"\n  [피처 중요도 Top 20]")
    for i, (feat, imp) in enumerate(fi_avg.head(20).items()):
        bar = '#' * int(imp / fi_avg.max() * 30)
        log(f"    {i+1:2d}. {feat:30s} {imp:8.0f}  {bar}")
    
    # 제출 파일 저장
    submission = pd.DataFrame({'id': test_id, 'target': test_preds})
    submission.to_csv(submission_name, index=False)
    log(f"\n  {submission_name} 저장 완료 ({len(submission):,}행)")
    log(f"  예측값 분포: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")
    
    # 차트 저장
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 10))
    top_fi = fi_avg.head(25)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_fi)))
    ax.barh(top_fi.index[::-1], top_fi.values[::-1], color=colors)
    ax.set_title(f'Feature Importance {name} (Top 25) — OOF AUC: {overall_auc:.6f}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Average Importance')
    plt.tight_layout()
    plt.savefig(chart_name, dpi=150)
    plt.close()
    log(f"  차트 저장: {chart_name}")
    
    del X, X_test
    gc.collect()
    
    return overall_auc, fold_scores, fi_avg

# ================================================================
# 7. 두 모델 실행
# ================================================================

# ---- Model A: n_estimators=2000, ID 포함 ----
auc_a, folds_a, fi_a = run_model(
    name="Model A (2000 + ID 포함)",
    feature_cols=feature_cols_a,
    train_df=train,
    test_df=test,
    y=y,
    params=params,
    submission_name='submission_v6a.csv',
    chart_name='eda_output/13_feature_importance_v6a.png'
)

# ---- Model B: n_estimators=2000, ID 제거 ----
auc_b, folds_b, fi_b = run_model(
    name="Model B (2000 + ID 제거)",
    feature_cols=feature_cols_b,
    train_df=train,
    test_df=test,
    y=y,
    params=params,
    submission_name='submission_v6b.csv',
    chart_name='eda_output/14_feature_importance_v6b.png'
)

# ================================================================
# 최종 비교
# ================================================================
log(f"\n{'='*65}")
log(f"  ★★★ 실험 결과 비교 ★★★")
log(f"{'='*65}")
log(f"")
log(f"  {'모델':<35} {'OOF AUC':>10} {'피처 수':>8}")
log(f"  {'-'*55}")
log(f"  {'v5 (1000, ID 포함) [기준선]':<35} {'0.8370':>10} {'?':>8}")
log(f"  {'Model A (2000, ID 포함)':<35} {auc_a:>10.6f} {len(feature_cols_a):>8}")
log(f"  {'Model B (2000, ID 제거)':<35} {auc_b:>10.6f} {len(feature_cols_b):>8}")
log(f"  {'-'*55}")
log(f"")
log(f"  A vs 기준선 (1000→2000 효과): {auc_a - 0.837:+.6f}")
log(f"  B vs A (ID 제거 효과):         {auc_b - auc_a:+.6f}")
log(f"")
log(f"  ※ OOF AUC가 낮아져도 Kaggle 점수는 올라갈 수 있음 (과적합 감소)")
log(f"  ※ 두 submission 파일 모두 Kaggle에 제출하여 비교 필요")
log(f"")
log(f"  제출 파일:")
log(f"    - submission_v6a.csv (Model A)")
log(f"    - submission_v6b.csv (Model B)")
log(f"{'='*65}")
