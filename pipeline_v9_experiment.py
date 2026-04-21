"""
KKBOX v9 실험: SVD 가중치 vs 차원 수 비교
==========================================
기준선: v8 (SVD 32차원, binary) → Kaggle 0.70

Model A: SVD 32차원 + log(count) 가중치 → 가중치 효과 측정
Model B: SVD 64차원 + binary           → 차원 증가 효과 측정
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
np.random.seed(SEED)

log("=" * 65)
log("  KKBOX v9 실험: SVD 가중치 vs 차원 수")
log("  Model A: 32차원 + log(count) 가중치")
log("  Model B: 64차원 + binary")
log("=" * 65)

# ================================================================
# 1~5: 데이터 로드 + 기존 피처 (v8과 동일)
# ================================================================
log("\n[1/5] 데이터 로드 중...")
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
    'composer': str, 'lyricist': str, 'language': 'float32'
}, usecols=['song_id', 'song_length', 'genre_ids', 'artist_name',
            'composer', 'lyricist', 'language'])
members = pd.read_csv('members.csv', dtype={
    'msno': 'category', 'city': 'int8', 'bd': 'int16', 'registered_via': 'int8'
})
extra = pd.read_csv('song_extra_info.csv', dtype={'song_id': 'category'},
                     usecols=['song_id', 'isrc'])
log(f"  로드 완료 ({time.time()-t0:.1f}초)")

log("\n[2/5] 전처리 + 병합...")
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
members['registration_init_time'] = pd.to_datetime(members['registration_init_time'], format='%Y%m%d', errors='coerce')
members['expiration_date'] = pd.to_datetime(members['expiration_date'], format='%Y%m%d', errors='coerce')
members['membership_days'] = (members['expiration_date'] - members['registration_init_time']).dt.days.astype('float32')
members['registration_year'] = members['registration_init_time'].dt.year.astype('float32')
members.drop(['bd', 'registration_init_time', 'expiration_date'], axis=1, inplace=True)
members['gender'] = members['gender'].astype('category').cat.codes.astype('int8')

train = train.merge(songs, on='song_id', how='left')
train = train.merge(members, on='msno', how='left')
test = test.merge(songs, on='song_id', how='left')
test = test.merge(members, on='msno', how='left')
del songs, members; gc.collect()
log(f"  병합 완료 ({time.time()-t0:.1f}초)")

log("\n[3/5] 인코딩 + Count 피처...")
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

for grp_cols, name in [(['msno_code', 'artist_code'], 'user_artist_count'),
                        (['msno_code', 'first_genre'], 'user_genre_count'),
                        (['msno_code', 'language'], 'user_lang_count'),
                        (['msno_code', 'song_id_code'], 'user_song_count')]:
    cnt = train.groupby(grp_cols).size().reset_index(name=name)
    train = train.merge(cnt, on=grp_cols, how='left')
    test = test.merge(cnt, on=grp_cols, how='left')
    test[name] = test[name].fillna(0).astype('int16')
    del cnt; gc.collect()

for df in [train, test]:
    df['user_artist_ratio'] = (df['user_artist_count'] / (df['user_count'] + 1)).astype('float32')
    df['user_genre_ratio'] = (df['user_genre_count'] / (df['user_count'] + 1)).astype('float32')
    df['user_lang_ratio'] = (df['user_lang_count'] / (df['user_count'] + 1)).astype('float32')
    df['song_artist_ratio'] = (df['song_count'] / (df['artist_count'] + 1)).astype('float32')
    df['user_song_ratio'] = (df['user_song_count'] / (df['user_count'] + 1)).astype('float32')
    df['song_per_unique_user'] = (df['song_count'] / (df['song_count'].clip(lower=1))).astype('float32')

log(f"  인코딩 + Count 완료 ({time.time()-t0:.1f}초)")

log("\n[4/5] 다양성 + 시퀀스 피처...")
t0 = time.time()

for grp, tgt, name in [('msno_code', 'artist_code', 'user_unique_artists'),
                         ('msno_code', 'first_genre', 'user_unique_genres'),
                         ('song_id_code', 'msno_code', 'song_unique_users'),
                         ('artist_code', 'msno_code', 'artist_unique_users'),
                         ('msno_code', 'song_id_code', 'user_unique_songs'),
                         ('msno_code', 'language', 'user_unique_langs')]:
    nuniq = train.groupby(grp)[tgt].nunique().reset_index(name=name)
    train = train.merge(nuniq, on=grp, how='left')
    test = test.merge(nuniq, on=grp, how='left')
    test[name] = test[name].fillna(0).astype('int32')
    del nuniq; gc.collect()

for df in [train, test]:
    df['song_listens_per_user'] = (df['song_count'] / (df['song_unique_users'] + 1)).astype('float32')
    df['user_diversity'] = (df['user_unique_artists'] / (df['user_count'] + 1)).astype('float32')
    df['user_repeat_ratio'] = (1 - df['user_unique_songs'] / (df['user_count'] + 1)).astype('float32')

train['user_listen_order'] = train.groupby('msno_code').cumcount().astype('int32')
train['user_listen_position'] = (train['user_listen_order'] / (train['user_count'])).astype('float32')
test['user_listen_order'] = test.groupby('msno_code').cumcount().astype('int32')
test_user_offset = train.groupby('msno_code').size().reset_index(name='_tc')
test = test.merge(test_user_offset, on='msno_code', how='left')
test['_tc'] = test['_tc'].fillna(0)
test['user_listen_order'] = (test['user_listen_order'] + test['_tc']).astype('int32')
test['user_listen_position'] = (test['user_listen_order'] / (test['user_count'] + test['_tc'] + 1)).astype('float32')
test.drop('_tc', axis=1, inplace=True)
del test_user_offset; gc.collect()

n_train = len(train)
for entity, code_col in [('song', 'song_id_code'), ('artist', 'artist_code')]:
    efirst = train.groupby(code_col).apply(lambda x: x.index.min()).reset_index(name=f'{entity}_first_appear')
    elast = train.groupby(code_col).apply(lambda x: x.index.max()).reset_index(name=f'{entity}_last_appear')
    epos = efirst.merge(elast, on=code_col)
    epos[f'{entity}_first_appear'] = (epos[f'{entity}_first_appear'] / n_train).astype('float32')
    epos[f'{entity}_last_appear'] = (epos[f'{entity}_last_appear'] / n_train).astype('float32')
    if entity == 'song':
        epos['song_span'] = (epos['song_last_appear'] - epos['song_first_appear']).astype('float32')
    train = train.merge(epos, on=code_col, how='left')
    test = test.merge(epos, on=code_col, how='left')
    for c in [f'{entity}_first_appear', f'{entity}_last_appear'] + (['song_span'] if entity == 'song' else []):
        test[c] = test[c].fillna(0.5).astype('float32')
    del efirst, elast, epos; gc.collect()

train['row_position'] = (np.arange(n_train) / n_train).astype('float32')
n_test = len(test)
test['row_position'] = ((np.arange(n_test) / n_test) * 0.1 + 1.0).astype('float32')

# 시퀀스 컨텍스트 피처
for col, new_name in [('artist_code', 'prev_artist'), ('first_genre', 'prev_genre'),
                       ('language', 'prev_language'), ('source_type', 'prev_source_type')]:
    train[new_name] = train.groupby('msno_code')[col].shift(1)
    test[new_name] = test.groupby('msno_code')[col].shift(1)

train['same_artist_as_prev'] = (train['artist_code'] == train['prev_artist']).fillna(-1).astype('int8')
train['same_genre_as_prev'] = (train['first_genre'] == train['prev_genre']).fillna(-1).astype('int8')
train['same_lang_as_prev'] = (train['language'] == train['prev_language']).fillna(-1).astype('int8')
train['same_source_as_prev'] = (train['source_type'] == train['prev_source_type']).fillna(-1).astype('int8')
test['same_artist_as_prev'] = (test['artist_code'] == test['prev_artist']).fillna(-1).astype('int8')
test['same_genre_as_prev'] = (test['first_genre'] == test['prev_genre']).fillna(-1).astype('int8')
test['same_lang_as_prev'] = (test['language'] == test['prev_language']).fillna(-1).astype('int8')
test['same_source_as_prev'] = (test['source_type'] == test['prev_source_type']).fillna(-1).astype('int8')
train.drop(['prev_artist', 'prev_genre', 'prev_language', 'prev_source_type'], axis=1, inplace=True)
test.drop(['prev_artist', 'prev_genre', 'prev_language', 'prev_source_type'], axis=1, inplace=True)

log(f"  다양성 + 시퀀스 완료 ({time.time()-t0:.1f}초)")

# ================================================================
# 5. SVD 생성 함수 + 모델 학습 함수
# ================================================================

# 상호작용 행렬의 raw data (count 기반) 미리 계산
log("\n[5/5] SVD 실험 준비...")
all_msno = pd.concat([train['msno_code'], test['msno_code']], ignore_index=True)
all_song = pd.concat([train['song_id_code'], test['song_id_code']], ignore_index=True)
n_users = all_msno.max() + 1
n_songs = all_song.max() + 1

# count 기반 행렬 (같은 유저-곡 조합의 횟수)
interaction_raw = csr_matrix(
    (np.ones(len(all_msno), dtype=np.float32), (all_msno.values, all_song.values)),
    shape=(n_users, n_songs)
)
# interaction_raw.data는 자동으로 count가 됨 (중복 좌표는 합산됨)
log(f"  행렬 크기: {interaction_raw.shape}, 비영: {interaction_raw.nnz:,}")
log(f"  최대 청취 횟수: {interaction_raw.data.max():.0f}")

del all_msno, all_song; gc.collect()

def build_svd_features(train_df, test_df, interaction, n_components, use_log_weight, tag):
    """SVD 피처를 생성하여 train/test에 추가"""
    log(f"\n  --- SVD 생성: {tag} ---")
    log(f"      차원: {n_components}, 가중치: {'log(count)' if use_log_weight else 'binary'}")
    
    mat = interaction.copy()
    if use_log_weight:
        mat.data = np.log1p(mat.data).astype('float32')  # log(1 + count)
    else:
        mat.data = np.minimum(mat.data, 1.0)  # binary
    
    svd = TruncatedSVD(n_components=n_components, random_state=SEED, n_iter=10)
    user_factors = svd.fit_transform(mat)
    song_factors = svd.components_.T
    log(f"      설명 분산: {svd.explained_variance_ratio_.sum():.4f}")
    
    # 유저/곡 SVD 피처
    for i in range(n_components):
        train_df[f'user_svd_{i}'] = user_factors[train_df['msno_code'].values, i].astype('float32')
        test_df[f'user_svd_{i}'] = user_factors[test_df['msno_code'].values, i].astype('float32')
        train_df[f'song_svd_{i}'] = song_factors[train_df['song_id_code'].values, i].astype('float32')
        test_df[f'song_svd_{i}'] = song_factors[test_df['song_id_code'].values, i].astype('float32')
    
    # 궁합 점수
    u_tr = user_factors[train_df['msno_code'].values]
    s_tr = song_factors[train_df['song_id_code'].values]
    train_df['svd_dot'] = np.sum(u_tr * s_tr, axis=1).astype('float32')
    
    u_te = user_factors[test_df['msno_code'].values]
    s_te = song_factors[test_df['song_id_code'].values]
    test_df['svd_dot'] = np.sum(u_te * s_te, axis=1).astype('float32')
    
    del mat, svd, user_factors, song_factors, u_tr, s_tr, u_te, s_te
    gc.collect()
    
    return train_df, test_df

def remove_svd_features(df):
    """SVD 관련 피처 제거"""
    svd_cols = [c for c in df.columns if c.startswith('user_svd_') or c.startswith('song_svd_') or c == 'svd_dot']
    df.drop(svd_cols, axis=1, inplace=True)
    return df

def run_model(name, train_df, test_df, y, test_id, submission_name, chart_name):
    """LightGBM 학습 + 제출 파일 생성"""
    drop_cols = ['msno', 'song_id', 'target', 'artist_name']
    if 'id' in test_df.columns:
        drop_cols.append('id')
    
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    X = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    svd_count = len([c for c in feature_cols if 'svd' in c])
    log(f"\n  {name}: 피처 {len(feature_cols)}개 (SVD {svd_count}개)")
    
    params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'learning_rate': 0.1, 'num_leaves': 255, 'max_depth': -1,
        'min_child_samples': 100, 'subsample': 0.7, 'colsample_bytree': 0.7,
        'reg_alpha': 1.0, 'reg_lambda': 5.0, 'n_estimators': 2000,
        'random_state': SEED, 'verbose': -1, 'n_jobs': -1,
    }
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    feature_importance = pd.DataFrame()
    t_start = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        log(f"    Fold {fold+1}/{N_FOLDS}...", )
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(500)])
        
        val_pred = model.predict_proba(X_vl)[:, 1]
        oof_preds[val_idx] = val_pred
        fold_auc = roc_auc_score(y_vl, val_pred)
        fold_scores.append(fold_auc)
        log(f"    Fold {fold+1} AUC: {fold_auc:.6f} (best iter: {model.best_iteration_})")
        
        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS
        
        fi = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_, 'fold': fold+1})
        feature_importance = pd.concat([feature_importance, fi], ignore_index=True)
        del X_tr, X_vl, y_tr, y_vl, val_pred; gc.collect()
    
    overall_auc = roc_auc_score(y, oof_preds)
    log(f"\n  ★ {name} 결과:")
    log(f"    OOF AUC: {overall_auc:.6f}")
    log(f"    Fold별: {[f'{s:.6f}' for s in fold_scores]}")
    log(f"    학습 시간: {time.time()-t_start:.1f}초")
    
    # 피처 중요도 Top 15
    fi_avg = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
    log(f"    Top 10 피처:")
    for i, (feat, imp) in enumerate(fi_avg.head(10).items()):
        log(f"      {i+1:2d}. {feat:30s} {imp:8.0f}")
    
    svd_total = fi_avg[fi_avg.index.str.contains('svd')].sum()
    log(f"    SVD 기여도: {svd_total/fi_avg.sum()*100:.1f}%")
    
    # 저장
    submission = pd.DataFrame({'id': test_id, 'target': test_preds})
    submission.to_csv(submission_name, index=False)
    log(f"    {submission_name} 저장 완료")

    # 차트
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    top_fi = fi_avg.head(25)
    colors = ['#FF6B6B' if 'svd' in f else '#45B7D1' for f in top_fi.index][::-1]
    ax.barh(top_fi.index[::-1], top_fi.values[::-1], color=colors)
    ax.set_title(f'{name} — OOF AUC: {overall_auc:.6f}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(chart_name, dpi=150)
    plt.close()
    
    del X, X_test; gc.collect()
    return overall_auc, fold_scores

# ================================================================
# 실험 실행
# ================================================================
test_id = test['id'].copy()
y = train['target'].copy()

# ---- Model A: 32차원 + log(count) 가중치 ----
log("\n" + "=" * 65)
log("  Model A: SVD 32차원 + log(count) 가중치")
log("=" * 65)

train, test = build_svd_features(train, test, interaction_raw, 
                                  n_components=32, use_log_weight=True, tag="A")
auc_a, folds_a = run_model("Model A (32dim, log-weight)", 
                            train, test, y, test_id,
                            'submission_v9a.csv', 
                            'eda_output/17_feature_importance_v9a.png')

# SVD 피처 제거 (Model B를 위해)
train = remove_svd_features(train)
test = remove_svd_features(test)
gc.collect()

# ---- Model B: 64차원 + binary ----
log("\n" + "=" * 65)
log("  Model B: SVD 64차원 + binary")
log("=" * 65)

train, test = build_svd_features(train, test, interaction_raw,
                                  n_components=64, use_log_weight=False, tag="B")
auc_b, folds_b = run_model("Model B (64dim, binary)",
                            train, test, y, test_id,
                            'submission_v9b.csv',
                            'eda_output/18_feature_importance_v9b.png')

del interaction_raw; gc.collect()

# ================================================================
# 최종 비교
# ================================================================
log(f"\n{'='*65}")
log(f"  ★★★ 실험 결과 비교 ★★★")
log(f"{'='*65}")
log(f"")
log(f"  {'모델':<40} {'OOF AUC':>10}")
log(f"  {'-'*52}")
log(f"  {'v8 기준선 (32dim, binary)':<40} {'0.8652':>10}")
log(f"  {'Model A (32dim, log-weight)':<40} {auc_a:>10.6f}")
log(f"  {'Model B (64dim, binary)':<40} {auc_b:>10.6f}")
log(f"  {'-'*52}")
log(f"")
log(f"  가중치 효과 (A vs 기준선): {auc_a - 0.8652:+.6f}")
log(f"  차원 증가 효과 (B vs 기준선): {auc_b - 0.8652:+.6f}")
log(f"")
log(f"  제출 파일: submission_v9a.csv, submission_v9b.csv")
log(f"{'='*65}")
