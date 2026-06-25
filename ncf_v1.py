"""
KKBox Music Recommendation — NCF (Neural Collaborative Filtering) v1
=====================================================================
LightGBM v9 (0.71)과의 비교를 위한 딥러닝 베이스라인.

NCF 핵심 아이디어:
  - 유저와 곡을 각각 임베딩 벡터로 표현
  - GMF(행렬 곱): 유저 벡터 ⊙ 곡 벡터 → 선형적 상호작용 포착
  - MLP(다층 퍼셉트론): [유저 벡터; 곡 벡터] → 비선형 상호작용 학습
  - NeuMF: GMF + MLP 결합 → 최종 예측

LightGBM과의 차이:
  - LightGBM: 사람이 만든 피처(SVD, 시퀀스 등)를 넣어줘야 함
  - NCF: 유저 ID와 곡 ID만으로 상호작용 패턴을 자동으로 학습
  - → "피처 엔지니어링의 가치는 모델이 스스로 발견할 수 없는 정보를 
       만들어주는 것" 이라는 KKBox 교훈의 검증
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import warnings
warnings.filterwarnings('ignore')

import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def log(msg):
    print(msg, flush=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================================================================
# 설정
# ================================================================
SAMPLE_SIZE = 1_000_000   # 빠른 실험용 (None이면 전체 사용)
EMBED_DIM = 32            # 임베딩 차원 (SVD 32차원과 동일하게)
MLP_DIMS = [128, 64, 32]  # MLP 레이어 구조
EPOCHS = 10
BATCH_SIZE = 4096
LR = 0.001
N_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log("=" * 65)
log("  KKBox Music Recommendation — NCF v1")
log(f"  Device: {DEVICE}")
log(f"  Sample: {SAMPLE_SIZE if SAMPLE_SIZE else 'Full'}")
log("=" * 65)

# ================================================================
# 1. 데이터 로드
# ================================================================
log("\n[1/5] 데이터 로드...")
t0 = time.time()

train = pd.read_csv('train.csv', usecols=['msno', 'song_id', 'target'])
test = pd.read_csv('test.csv', usecols=['id', 'msno', 'song_id'])

log(f"  train: {len(train):,}행, test: {len(test):,}행 ({time.time()-t0:.1f}초)")

# ================================================================
# 2. ID 매핑 (문자열 → 정수 인덱스)
# ================================================================
log("\n[2/5] ID 매핑...")
t0 = time.time()

# train + test 전체에서 유니크 ID 수집 (test에만 있는 유저/곡도 포함)
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

log(f"  유저: {n_users:,}명, 곡: {n_songs:,}곡 ({time.time()-t0:.1f}초)")

# 샘플링
if SAMPLE_SIZE and SAMPLE_SIZE < len(train):
    log(f"\n  샘플링: {len(train):,} → {SAMPLE_SIZE:,}행")
    train = train.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)

# ================================================================
# 3. 모델 정의: NeuMF (GMF + MLP)
# ================================================================
log("\n[3/5] 모델 정의...")

class NeuMF(nn.Module):
    """
    Neural Matrix Factorization (He et al., 2017)
    
    구조:
      User ID → [GMF embedding] → element-wise product ─┐
      Song ID → [GMF embedding] ─────────────────────────┘→ concat → output
      
      User ID → [MLP embedding] → concat → MLP layers ──┐
      Song ID → [MLP embedding] ─────────────────────────┘→ concat → output
    """
    def __init__(self, n_users, n_songs, embed_dim, mlp_dims):
        super().__init__()
        
        # GMF 파트: 유저와 곡의 잠재 벡터를 element-wise 곱
        self.gmf_user = nn.Embedding(n_users, embed_dim)
        self.gmf_song = nn.Embedding(n_songs, embed_dim)
        
        # MLP 파트: 유저와 곡의 벡터를 concat 후 비선형 변환
        self.mlp_user = nn.Embedding(n_users, embed_dim)
        self.mlp_song = nn.Embedding(n_songs, embed_dim)
        
        # MLP 레이어 구성
        mlp_layers = []
        input_dim = embed_dim * 2  # concat이므로 2배
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(input_dim, dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.2))
            input_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)
        
        # 최종 예측: GMF 출력(embed_dim) + MLP 출력(mlp_dims[-1]) → 1
        self.output = nn.Linear(embed_dim + mlp_dims[-1], 1)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, user_ids, song_ids):
        # GMF: element-wise product
        gmf_u = self.gmf_user(user_ids)    # (batch, embed_dim)
        gmf_s = self.gmf_song(song_ids)    # (batch, embed_dim)
        gmf_out = gmf_u * gmf_s            # (batch, embed_dim)
        
        # MLP: concat → nonlinear layers
        mlp_u = self.mlp_user(user_ids)    # (batch, embed_dim)
        mlp_s = self.mlp_song(song_ids)    # (batch, embed_dim)
        mlp_input = torch.cat([mlp_u, mlp_s], dim=1)  # (batch, embed_dim*2)
        mlp_out = self.mlp(mlp_input)      # (batch, mlp_dims[-1])
        
        # 결합 → 최종 예측
        combined = torch.cat([gmf_out, mlp_out], dim=1)
        logit = self.output(combined).squeeze()  # (batch,)
        return logit


class KKBoxDataset(Dataset):
    def __init__(self, users, songs, targets=None):
        self.users = torch.LongTensor(users)
        self.songs = torch.LongTensor(songs)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.users[idx], self.songs[idx], self.targets[idx]
        return self.users[idx], self.songs[idx]


# ================================================================
# 4. 학습 (5-Fold CV)
# ================================================================
log("\n[4/5] 학습 시작...")
log(f"  Embedding: {EMBED_DIM}차원, MLP: {MLP_DIMS}")
log(f"  Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")

y = train['target'].values
users_arr = train['user_idx'].values
songs_arr = train['song_idx'].values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_scores = []

test_users = torch.LongTensor(test['user_idx'].values).to(DEVICE)
test_songs = torch.LongTensor(test['song_idx'].values).to(DEVICE)

for fold, (train_idx, val_idx) in enumerate(skf.split(users_arr, y)):
    log(f"\n  --- Fold {fold+1}/{N_FOLDS} ---")
    t_fold = time.time()
    
    # 데이터셋
    train_ds = KKBoxDataset(users_arr[train_idx], songs_arr[train_idx], y[train_idx])
    val_ds = KKBoxDataset(users_arr[val_idx], songs_arr[val_idx], y[val_idx])
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
    
    # 모델
    model = NeuMF(n_users, n_songs, EMBED_DIM, MLP_DIMS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    best_auc = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0
        n_batches = 0
        
        for batch_users, batch_songs, batch_targets in train_dl:
            batch_users = batch_users.to(DEVICE)
            batch_songs = batch_songs.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(batch_users, batch_songs)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = train_loss / n_batches
        
        # --- Validate ---
        model.eval()
        val_preds_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for batch_users, batch_songs, batch_targets in val_dl:
                batch_users = batch_users.to(DEVICE)
                batch_songs = batch_songs.to(DEVICE)
                
                logits = model(batch_users, batch_songs)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds_list.append(probs)
                val_targets_list.append(batch_targets.numpy())
        
        val_preds_epoch = np.concatenate(val_preds_list)
        val_targets_epoch = np.concatenate(val_targets_list)
        val_auc = roc_auc_score(val_targets_epoch, val_preds_epoch)
        
        scheduler.step(-val_auc)
        
        log(f"    Epoch {epoch+1:2d}/{EPOCHS}: "
            f"loss={avg_train_loss:.4f}, val_AUC={val_auc:.6f}"
            f"{' *best*' if val_auc > best_auc else ''}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 4:
                log(f"    Early stopping at epoch {epoch+1}")
                break
    
    # Best 모델로 OOF + Test 예측
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        # OOF
        val_preds_final = []
        for batch_users, batch_songs, _ in val_dl:
            logits = model(batch_users.to(DEVICE), batch_songs.to(DEVICE))
            val_preds_final.append(torch.sigmoid(logits).cpu().numpy())
        oof_preds[val_idx] = np.concatenate(val_preds_final)
        
        # Test (배치 처리)
        test_batch_preds = []
        for i in range(0, len(test_users), BATCH_SIZE*2):
            batch_u = test_users[i:i+BATCH_SIZE*2]
            batch_s = test_songs[i:i+BATCH_SIZE*2]
            logits = model(batch_u, batch_s)
            test_batch_preds.append(torch.sigmoid(logits).cpu().numpy())
        test_preds += np.concatenate(test_batch_preds) / N_FOLDS
    
    fold_auc = best_auc
    fold_scores.append(fold_auc)
    log(f"  Fold {fold+1} best AUC: {fold_auc:.6f} ({time.time()-t_fold:.1f}초)")
    
    del model, optimizer, train_ds, val_ds, train_dl, val_dl
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ================================================================
# 5. 결과 요약 + 제출
# ================================================================
overall_auc = roc_auc_score(y, oof_preds)

log(f"\n{'='*65}")
log(f"  NCF v1 결과")
log(f"{'='*65}")
log(f"  OOF AUC: {overall_auc:.6f}")
log(f"  Fold별:  {[f'{s:.6f}' for s in fold_scores]}")
log(f"  평균:    {np.mean(fold_scores):.6f} (+/- {np.std(fold_scores):.6f})")
log(f"")
log(f"  비교:")
log(f"    LightGBM v9 (SVD+FE)   Kaggle AUC: 0.71")
log(f"    NCF v1 (ID only)       OOF AUC:    {overall_auc:.4f}")
log(f"{'='*65}")

# 제출 파일
submission = pd.DataFrame({'id': test['id'], 'target': test_preds})
submission.to_csv('submission_ncf_v1.csv', index=False)
log(f"\n  submission_ncf_v1.csv 저장 완료")
log(f"  Kaggle에 제출하여 실제 점수 확인 가능")
