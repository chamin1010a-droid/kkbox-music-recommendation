"""
==========================================================================
  Stage 1: Song2Vec — Word2Vec으로 음악 추천 임베딩 학습
==========================================================================

[학습 목표]
  이전 v8/v9에서 SVD로 유저-곡 임베딩을 만들었다.
  SVD는 "누가 어떤 곡을 들었는가"라는 동시출현(co-occurrence) 정보만 사용한다.
  하지만 음악 청취에는 "순서"가 있다:
    - A → B → C 순서로 들었다면, A와 B는 B와 C보다 "가까운" 관계
    - 발라드를 연속 3곡 듣다가 댄스곡을 틀었다면, 그 댄스곡은 특별한 의미

  Word2Vec(2013, Mikolov et al.)은 자연어 처리에서 나온 기법이지만,
  "시퀀스 데이터에서 주변 컨텍스트를 이용해 임베딩을 학습한다"는 본질은
  음악 추천에도 그대로 적용된다.

[핵심 비유]
  자연어:  "나는 오늘 학교에 갔다" → 각 단어의 의미를 주변 단어로 학습
  음악:    [BTS-봄날, IU-밤편지, 악뮤-How Can I Love, ...] → 곡의 의미를 주변 곡으로 학습

  이 비유가 Song2Vec의 전부다. 나머지는 구현 디테일.

==========================================================================
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import time
import gc
import warnings
warnings.filterwarnings('ignore')

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def log(msg):
    print(msg, flush=True)

SEED = 42
np.random.seed(SEED)

log("=" * 70)
log("  Stage 1: Song2Vec — Word2Vec으로 음악 추천 임베딩 학습하기")
log("=" * 70)


# =====================================================================
# Part 1: 데이터 로드 + 시퀀스 구성
# =====================================================================
log("\n" + "=" * 70)
log("  Part 1: 데이터 로드 및 시퀀스 구성")
log("=" * 70)

"""
[이론] Word2Vec이 필요로 하는 데이터 형태

  Word2Vec의 입력은 "문장들의 리스트"이다.
  sentences = [
      ["나", "는", "학생", "이다"],
      ["오늘", "날씨", "가", "좋다"],
      ...
  ]

  우리는 이것을 음악에 적용한다:
  sentences = [
      ["song_001", "song_042", "song_007", ...],  ← 유저A의 청취 시퀀스
      ["song_100", "song_200", "song_003", ...],  ← 유저B의 청취 시퀀스
      ...
  ]

  핵심: 각 유저의 청취 기록을 "시간순으로 정렬된 곡 ID 리스트"로 만든다.
  KKBox 데이터에는 타임스탬프가 없지만, 행 순서가 시간순이므로 그대로 사용.
"""

log("\n[1/6] 데이터 로드 중...")
t0 = time.time()

train = pd.read_csv('train.csv', dtype={
    'msno': str, 'song_id': str,
    'source_system_tab': str, 'source_screen_name': str,
    'source_type': str, 'target': 'int8'
})
songs = pd.read_csv('songs.csv', dtype={
    'song_id': str, 'artist_name': str,
    'genre_ids': str, 'language': 'float32'
}, usecols=['song_id', 'artist_name', 'genre_ids', 'language'])

log(f"  Train: {len(train):,}행, Songs: {len(songs):,}곡")
log(f"  유저 수: {train['msno'].nunique():,}")
log(f"  곡 수: {train['song_id'].nunique():,}")
log(f"  로드 시간: {time.time()-t0:.1f}초")

# 곡 메타데이터 매핑 (나중에 시각화에서 사용)
song_meta = songs.set_index('song_id')[['artist_name', 'genre_ids', 'language']].to_dict('index')

log("\n[2/6] 유저별 청취 시퀀스 구성 중...")
t0 = time.time()

"""
[이론] 시퀀스 구성의 중요성

  Word2Vec은 "윈도우(window)" 안의 단어들이 관련있다고 가정한다.
  예를 들어 window=5이면, 중심 단어 기준 좌우 5개 단어가 컨텍스트.

  음악에서 이는:
  - window=5: "이 곡 앞뒤 5곡이 비슷한 맥락이다"
  - window가 너무 크면: 관련 없는 곡도 같은 맥락으로 취급
  - window가 너무 작으면: 바로 인접한 곡만 관련짓는 근시안

  일반적으로 음악 추천에서는 window=5~10이 적절하다.
  
  또 하나 중요한 점:
  - 시퀀스가 너무 짧은 유저(1~2곡만 들은)는 학습에 기여가 적다.
  - 최소 3곡 이상 들은 유저만 사용한다.
"""

# 유저별로 곡 ID를 시간순(행 순서)으로 리스트화
user_sequences = train.groupby('msno')['song_id'].apply(list).reset_index()
user_sequences.columns = ['msno', 'sequence']
user_sequences['seq_len'] = user_sequences['sequence'].apply(len)

log(f"  전체 유저 수: {len(user_sequences):,}")
log(f"  시퀀스 길이 통계:")
log(f"    평균: {user_sequences['seq_len'].mean():.1f}곡")
log(f"    중앙값: {user_sequences['seq_len'].median():.0f}곡")
log(f"    최대: {user_sequences['seq_len'].max():,}곡")
log(f"    최소: {user_sequences['seq_len'].min()}곡")

# 시퀀스가 3곡 이상인 유저만 필터
MIN_SEQ_LEN = 3
filtered = user_sequences[user_sequences['seq_len'] >= MIN_SEQ_LEN]
sentences = filtered['sequence'].tolist()

log(f"\n  필터 후 유저 수: {len(sentences):,} (최소 {MIN_SEQ_LEN}곡)")
log(f"  총 토큰(곡) 수: {sum(len(s) for s in sentences):,}")
log(f"  시퀀스 구성 시간: {time.time()-t0:.1f}초")


# =====================================================================
# Part 2: Word2Vec 이론 — Skip-gram vs CBOW
# =====================================================================
log("\n" + "=" * 70)
log("  Part 2: Word2Vec 핵심 이론")
log("=" * 70)

"""
[이론] Word2Vec의 두 가지 아키텍처

  ┌─────────────────────────────────────────────────────────────┐
  │  1. CBOW (Continuous Bag of Words)                          │
  │     "주변 단어들로 중심 단어를 예측한다"                         │
  │                                                             │
  │     입력: [봄날, ?, 밤편지, 에잇]  (주변 곡들)                  │
  │     출력: "How Can I Love" (중심 곡 예측)                     │
  │                                                             │
  │     수식: P(중심곡 | 주변곡들) 을 최대화                        │
  │                                                             │
  │     특징:                                                    │
  │     - 빠르다 (주변 단어를 평균내서 한 번에 예측)                  │
  │     - 빈도 높은 단어에 유리 (자주 나오는 곡을 잘 예측)            │
  │     - 일반적인 패턴 포착에 강함                                 │
  │                                                             │
  │  2. Skip-gram                                               │
  │     "중심 단어로 주변 단어를 예측한다" (CBOW의 반대)              │
  │                                                             │
  │     입력: "How Can I Love" (중심 곡)                          │
  │     출력: [봄날, 밤편지, 에잇] 각각을 개별적으로 예측             │
  │                                                             │
  │     수식: P(주변곡 | 중심곡) 을 최대화                          │
  │                                                             │
  │     특징:                                                    │
  │     - 느리다 (주변 단어 각각에 대해 예측 → 업데이트 횟수 많음)     │
  │     - 희귀한 단어에 유리 (적게 나오는 곡도 잘 배운다)             │
  │     - 추천 시스템에서 더 인기 → 나도 이것을 사용                 │
  └─────────────────────────────────────────────────────────────┘

  [왜 Skip-gram이 추천에 더 적합한가?]
  
  음악 데이터에서 인기곡은 수만 번, 비인기곡은 1~2번만 등장한다.
  CBOW는 빈도가 높은 곡 쪽으로 학습이 편향되지만,
  Skip-gram은 각 등장마다 독립적으로 학습하므로 롱테일(비인기곡)도
  양질의 임베딩을 얻을 수 있다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[이론] Negative Sampling — Word2Vec을 실현 가능하게 만든 트릭

  원래 Word2Vec의 출력 레이어는 "전체 어휘 크기"의 Softmax이다.
  KKBox에는 곡이 약 36만개 → 매 학습마다 36만 차원 Softmax를 계산?
  → 불가능!

  해결책: Negative Sampling
  
  핵심 아이디어:
    "실제 함께 나온 곡 쌍(Positive)은 내적이 크게,
     랜덤으로 뽑은 곡 쌍(Negative)은 내적이 작게"
     → 이진 분류 문제로 변환!
  
  예시:
    시퀀스: [..., BTS-봄날, IU-밤편지, ...]
    
    Positive pair: (봄날, 밤편지) → label=1 (실제 이웃)
    Negative pairs: (봄날, 랜덤곡1) → label=0
                    (봄날, 랜덤곡2) → label=0
                    (봄날, 랜덤곡3) → label=0
                    ...
    
    이렇게 하면 36만 차원 Softmax 대신, (1 + neg_samples)개의 이진 분류만 하면 됨.
    보통 neg_samples=5~20을 사용.
  
  Negative를 뽑을 때 빈도 기반 확률 사용:
    P(곡) ∝ freq(곡)^0.75
    
    왜 0.75승? 순수한 빈도 비례(1.0승)로 하면 인기곡만 Negative로 뽑혀서
    비인기곡 페어의 학습이 부실해진다. 0.75승은 빈도 차이를 완화시키는 트릭.
    (0이면 균일분포, 1이면 빈도 비례, 0.75는 그 사이의 "Sweet Spot")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[이론] 학습 과정 시각화

  초기 상태: 모든 곡의 임베딩이 랜덤 위치에 흩어져 있음
  
  Epoch 1:
    "봄날 다음에 밤편지가 나왔네?"
    → 봄날 벡터와 밤편지 벡터를 가까이 당김 (Positive)
    → 봄날 벡터와 랜덤곡 벡터를 멀리 밀어냄 (Negative)
  
  Epoch 10:
    같은 유저가 자주 연속해서 듣는 곡들이 점점 가까워짐
    → 발라드끼리 한 구석으로, 댄스곡끼리 다른 구석으로
  
  Epoch 20:
    장르뿐 아니라 "분위기"까지 반영됨
    → 같은 발라드여도 "이별 발라드"와 "행복한 발라드"가 분리
    → 이것이 SVD에서는 잡기 어려운 시퀀스 기반 미세 패턴!
"""


# =====================================================================
# Part 3: Song2Vec 학습
# =====================================================================
log("\n" + "=" * 70)
log("  Part 3: Song2Vec 학습")
log("=" * 70)

"""
[이론] Word2Vec 하이퍼파라미터의 의미

  vector_size=128:  각 곡을 128차원 벡터로 표현
    - 차원이 클수록 표현력↑, 과적합 위험↑, 학습 시간↑
    - SVD에서 32차원을 썼으므로, 128은 4배 풍부한 표현
    - 보통 64~256 사이가 적당

  window=10:  중심 곡 기준 양쪽 10곡까지 컨텍스트로 사용
    - 음악은 자연어보다 "맥락 전환"이 부드러움
    - 10곡 = 약 30~40분의 연속 청취 → 한 세션 정도
    - 자연어에서는 보통 5가 기본값

  min_count=5:  5번 미만 등장한 곡은 무시
    - 너무 적게 나온 곡은 좋은 임베딩을 학습할 수 없음
    - 5는 "최소 5명의 유저 시퀀스에 등장"을 의미

  sg=1:  Skip-gram 사용 (0이면 CBOW)
    - 위에서 설명한 대로, 추천에는 Skip-gram이 더 적합

  negative=10:  Negative Sample 10개
    - 1개의 Positive에 대해 10개의 Negative 생성
    - 너무 적으면 학습이 불안정, 너무 많으면 느림
    - 5~20이 일반적

  workers=4:  4개 스레드로 병렬 학습
    - gensim은 C로 구현된 멀티스레드 학습 지원

  epochs=20:  전체 데이터를 20번 반복
    - 자연어에서는 5가 기본이지만, 음악 데이터는 시퀀스가 짧아서 더 많이 돌림
"""

log("\n[3/6] Song2Vec (Word2Vec) 학습 중...")
log("  하이퍼파라미터:")
log("    vector_size=128 (임베딩 차원)")
log("    window=10 (컨텍스트 윈도우)")
log("    min_count=5 (최소 등장 횟수)")
log("    sg=1 (Skip-gram)")
log("    negative=10 (네거티브 샘플 수)")
log("    epochs=20")
log(f"  학습 시퀀스 수: {len(sentences):,}")

t0 = time.time()

song2vec = Word2Vec(
    sentences=sentences,    # 유저별 청취 시퀀스 리스트
    vector_size=128,        # 임베딩 차원
    window=10,              # 컨텍스트 윈도우 크기
    min_count=5,            # 최소 등장 횟수
    sg=1,                   # 1=Skip-gram, 0=CBOW
    negative=10,            # Negative Sampling 수
    workers=4,              # 병렬 처리 스레드
    epochs=20,              # 에폭 수
    seed=SEED,
)

train_time = time.time() - t0
vocab_size = len(song2vec.wv)
log(f"\n  ✓ 학습 완료! ({train_time:.1f}초)")
log(f"  어휘 크기: {vocab_size:,}곡 (min_count=5 필터 후)")
log(f"  임베딩 행렬 크기: ({vocab_size}, 128)")

# 메모리에서 학습 데이터 해제 (임베딩만 유지)
del sentences, user_sequences, filtered
gc.collect()


# =====================================================================
# Part 4: 임베딩 품질 분석 — "의미 있는 공간이 만들어졌는가?"
# =====================================================================
log("\n" + "=" * 70)
log("  Part 4: 임베딩 품질 분석")
log("=" * 70)

"""
[이론] 좋은 임베딩이란?

  Word2Vec이 잘 학습됐다면, 임베딩 공간에서:
  1. 같은 아티스트의 곡들이 가까이 모여야 한다
  2. 같은 장르의 곡들이 가까이 모여야 한다
  3. "유사곡"을 찾으면 직관적으로 맞는 결과가 나와야 한다

  검증 방법:
  - most_similar(): 코사인 유사도 기반 유사곡 검색
  - t-SNE: 128차원 → 2차원 축소 시각화
"""

log("\n[4/6] 유사곡 검색 테스트...")

# train에서 가장 인기 있는 곡 Top 20 찾기
song_counts = train['song_id'].value_counts()
top_songs = song_counts.head(20).index.tolist()

# 어휘에 있는 인기곡 중 상위 5개 선택
test_songs = [s for s in top_songs if s in song2vec.wv][:5]

log(f"\n  인기곡 {len(test_songs)}개에 대한 유사곡 검색:")
log("  " + "-" * 64)

for song_id in test_songs:
    meta = song_meta.get(song_id, {})
    artist = meta.get('artist_name', '?')
    genre = meta.get('genre_ids', '?')
    genre = str(genre)[:15] if isinstance(genre, str) else '?'
    count = song_counts.get(song_id, 0)

    log(f"\n  ▶ {song_id[:20]}... (아티스트: {artist}, 장르: {genre})")
    log(f"    전체 등장 횟수: {count:,}회")

    # most_similar: 코사인 유사도 기준 가장 가까운 곡 TOP 5
    """
    [이론] 코사인 유사도

      cos_sim(A, B) = (A · B) / (||A|| × ||B||)

      - 벡터의 "방향"만 비교 (크기 무시)
      - 범위: -1 (반대 방향) ~ 0 (무관) ~ 1 (같은 방향)
      - 임베딩 공간에서 "비슷한 맥락에서 등장하는 곡"끼리 코사인 유사도가 높음
    """
    similar = song2vec.wv.most_similar(song_id, topn=5)

    for rank, (sim_song, sim_score) in enumerate(similar, 1):
        sim_meta = song_meta.get(sim_song, {})
        sim_artist = sim_meta.get('artist_name', '?')
        sim_genre = sim_meta.get('genre_ids', '?')
        sim_genre = str(sim_genre)[:15] if isinstance(sim_genre, str) else '?'
        same_artist = "★ 같은 아티스트!" if sim_artist == artist else ""
        log(f"    {rank}. 유사도={sim_score:.3f} | {sim_artist} (장르: {sim_genre}) {same_artist}")

"""
[이론] 벡터 연산의 마법 — Word2Vec의 가장 유명한 성질

  자연어에서: king - man + woman ≈ queen
  음악에서도 비슷한 연산이 (이론적으로) 가능:
    BTS곡 - 댄스느낌 + 발라드느낌 ≈ BTS의 발라드곡

  다만 현실적으로는 음악 데이터가 자연어만큼 풍부하지 않아서
  완벽하진 않다. 하지만 방향성은 확인할 수 있다.
"""


# =====================================================================
# Part 5: t-SNE 시각화 — 임베딩 공간 탐색
# =====================================================================
log("\n" + "=" * 70)
log("  Part 5: t-SNE 시각화")
log("=" * 70)

"""
[이론] t-SNE (t-distributed Stochastic Neighbor Embedding)

  128차원 공간을 눈으로 볼 수 없으니, 2차원으로 압축해서 시각화한다.
  
  t-SNE의 핵심 원리:
  1. 고차원에서 각 점 쌍의 "이웃 확률"을 계산 (가우시안 커널)
  2. 저차원(2D)에서도 동일한 "이웃 확률"을 유지하려고 노력
  3. KL-Divergence를 최소화하며 2D 좌표를 최적화
  
  주의사항:
  - t-SNE는 거리를 정확하게 보존하지 않음 (지역 구조만 보존)
  - 같은 데이터도 perplexity, learning_rate에 따라 결과가 다름
  - "클러스터가 보인다" = 의미 있음, 하지만 "클러스터 간 거리"는 신뢰하지 마라
  
  perplexity=30: "각 점이 약 30개의 이웃을 가진다"는 가정
  - 데이터 크기에 따라 5~50 사이에서 조정
"""

log("\n[5/6] t-SNE 시각화 준비 중...")
t0 = time.time()

# 시각화할 곡 샘플링 (상위 인기곡 2000개)
# 전체를 t-SNE하면 너무 느리므로 인기곡 위주로 샘플링
N_VIS = 2000
top_n_songs = song_counts.head(N_VIS * 2).index.tolist()
vis_songs = [s for s in top_n_songs if s in song2vec.wv][:N_VIS]

# 임베딩 행렬 추출
vis_vectors = np.array([song2vec.wv[s] for s in vis_songs])
log(f"  시각화 대상: {len(vis_songs)}곡, 벡터 크기: {vis_vectors.shape}")

# 각 곡의 첫 번째 장르 + 언어 정보 추출 (색상용)
vis_genres = []
vis_langs = []
for s in vis_songs:
    meta = song_meta.get(s, {})
    genre = meta.get('genre_ids', '')
    first_genre = genre.split('|')[0] if isinstance(genre, str) and genre else 'unknown'
    lang = meta.get('language', -1)
    lang = lang if lang == lang else -1  # NaN check
    vis_genres.append(first_genre)
    vis_langs.append(lang)

# 상위 빈도 장르 10개만 색상 부여, 나머지는 회색
from collections import Counter
genre_counter = Counter(vis_genres)
top_genres = [g for g, _ in genre_counter.most_common(10)]

genre_colors = {}
cmap = plt.cm.tab10
for i, g in enumerate(top_genres):
    genre_colors[g] = cmap(i)

log("  t-SNE 차원 축소 중... (1~2분 소요)")
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
vis_2d = tsne.fit_transform(vis_vectors)
log(f"  t-SNE 완료 ({time.time()-t0:.1f}초)")

# 시각화 1: 장르별 색상
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# --- 좌: 장르별 ---
ax = axes[0]
# 기타(회색) 먼저 그리기
other_mask = [g not in top_genres for g in vis_genres]
ax.scatter(vis_2d[other_mask, 0], vis_2d[other_mask, 1],
           c='lightgray', s=8, alpha=0.3, label='기타')

# 상위 장르 순서대로
for genre in top_genres:
    mask = [g == genre for g in vis_genres]
    if sum(mask) > 0:
        ax.scatter(vis_2d[mask, 0], vis_2d[mask, 1],
                   c=[genre_colors[genre]], s=15, alpha=0.6, label=f'장르 {genre}')

ax.set_title('Song2Vec 임베딩 — 장르별 분포', fontsize=14, fontweight='bold')
ax.legend(fontsize=7, loc='upper right', ncol=2)
ax.set_xticks([]); ax.set_yticks([])

# --- 우: 언어별 ---
ax = axes[1]
lang_map = {-1.0: '미상', 3.0: '중국어', 31.0: '일본어', 52.0: '영어',
            10.0: '한국어', 17.0: '대만어', 24.0: '기타'}
lang_colors = {3.0: '#FF6B6B', 31.0: '#4ECDC4', 52.0: '#45B7D1',
               10.0: '#96CEB4', 17.0: '#FFEAA7', 24.0: '#DDA0DD'}

for lang_code, color in lang_colors.items():
    mask = [l == lang_code for l in vis_langs]
    if sum(mask) > 0:
        label = lang_map.get(lang_code, f'Lang {int(lang_code)}')
        ax.scatter(vis_2d[mask, 0], vis_2d[mask, 1],
                   c=color, s=15, alpha=0.6, label=label)

# 나머지 언어
other_mask = [l not in lang_colors for l in vis_langs]
ax.scatter(vis_2d[other_mask, 0], vis_2d[other_mask, 1],
           c='lightgray', s=8, alpha=0.3, label='기타')

ax.set_title('Song2Vec 임베딩 — 언어별 분포', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('Song2Vec (Skip-gram, 128dim, window=10) — t-SNE 시각화',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_output/19_song2vec_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
log("  ✓ 시각화 저장: eda_output/19_song2vec_tsne.png")


# =====================================================================
# Part 6: SVD vs Song2Vec 비교
# =====================================================================
log("\n" + "=" * 70)
log("  Part 6: SVD vs Song2Vec — 근본적 차이 비교")
log("=" * 70)

"""
[이론] SVD와 Song2Vec의 근본적 차이

  ┌────────────────────────────────────────────────────────────────┐
  │                    SVD (행렬 분해)                              │
  │                                                                │
  │  입력: 유저-곡 행렬 (동시출현)                                   │
  │        User\Song   곡A  곡B  곡C                               │
  │        유저1        1    1    0                                 │
  │        유저2        0    1    1                                 │
  │                                                                │
  │  학습: ||M - U × V^T||² 최소화                                 │
  │        → 행렬을 두 개의 낮은 차원 행렬로 분해                     │
  │                                                                │
  │  결과: 유저1이 곡A, 곡B를 들었다 → 곡A ≈ 곡B                    │
  │        (순서 무시: A→B든 B→A든 동일)                             │
  │                                                                │
  │  장점: 전역적(Global) 구조 포착, 수학적으로 깔끔                   │
  │  한계: 순서 정보 손실, 새로운 곡 추가 시 재계산 필요               │
  ├────────────────────────────────────────────────────────────────┤
  │                   Song2Vec (시퀀스 임베딩)                      │
  │                                                                │
  │  입력: 청취 시퀀스                                               │
  │        유저1: [곡A, 곡B, 곡D, ...]                              │
  │        유저2: [곡B, 곡C, 곡A, ...]                              │
  │                                                                │
  │  학습: P(주변곡 | 중심곡) 최대화 (Skip-gram)                     │
  │        → 윈도우 내 동시등장 확률을 높이도록 임베딩 학습             │
  │                                                                │
  │  결과: A→B 전환이 많다 → A ≈ B (순서 반영!)                     │
  │        B→A 전환이 적다면 A→B보다 약간 다른 관계 학습              │
  │                                                                │
  │  장점: 순서/컨텍스트 반영, 온라인 학습 가능                       │
  │  한계: 윈도우 밖의 장거리 관계 약함, 하이퍼파라미터 민감           │
  └────────────────────────────────────────────────────────────────┘

  핵심 질문: "같은 곡에 대해 SVD와 Song2Vec이 다른 유사곡을 추천하는가?"
  → 다르다면, 각각이 포착하는 정보가 다르다는 증거
  → 이 경우 둘을 함께 쓰면 더 좋은 추천이 가능 (앙상블 효과)
"""

log("\n  SVD 임베딩 생성 중 (비교용, 128차원)...")
t0 = time.time()

# SVD: 유저-곡 상호작용 행렬 구성
all_msno = train['msno']
all_songs_col = train['song_id']

# 인코딩
msno_codes, msno_uniques = pd.factorize(all_msno)
song_codes, song_uniques = pd.factorize(all_songs_col)

interaction = csr_matrix(
    (np.ones(len(msno_codes), dtype=np.float32), (msno_codes, song_codes)),
    shape=(len(msno_uniques), len(song_uniques))
)
interaction.data = np.minimum(interaction.data, 1.0)  # binary

svd = TruncatedSVD(n_components=128, random_state=SEED, n_iter=10)
user_factors = svd.fit_transform(interaction)
song_factors_svd = svd.components_.T  # (n_songs, 128)

# song_id → SVD 벡터 매핑
svd_song_map = {song_uniques[i]: song_factors_svd[i] for i in range(len(song_uniques))}
log(f"  SVD 완료 ({time.time()-t0:.1f}초), 설명 분산: {svd.explained_variance_ratio_.sum():.4f}")

del user_factors, interaction, svd
gc.collect()

# ── 유사곡 비교 ──
log("\n  SVD vs Song2Vec 유사곡 비교:")
log("  " + "=" * 64)

# 코사인 유사도 함수
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def svd_most_similar(song_id, svd_map, topn=5):
    """SVD 임베딩에서 코사인 유사도 기반 유사곡 검색"""
    if song_id not in svd_map:
        return []
    vec = svd_map[song_id]
    sims = []
    for other_id, other_vec in svd_map.items():
        if other_id == song_id:
            continue
        sim = cosine_sim(vec, other_vec)
        sims.append((other_id, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:topn]

# 비교 곡 3개 선택 (Song2Vec, SVD 양쪽 모두에 있는 곡)
compare_songs = [s for s in test_songs if s in svd_song_map][:3]

# 전체 SVD 맵은 너무 크므로, Song2Vec 어휘에 있는 곡들만 SVD에서도 비교
svd_vocab_map = {s: svd_song_map[s] for s in song2vec.wv.index_to_key if s in svd_song_map}
log(f"  비교 가능 곡 수: {len(svd_vocab_map):,}")

for song_id in compare_songs:
    meta = song_meta.get(song_id, {})
    artist = meta.get('artist_name', '?')

    log(f"\n  ▶ {artist} ({song_id[:20]}...)")

    # Song2Vec 유사곡
    w2v_similar = song2vec.wv.most_similar(song_id, topn=5)
    # SVD 유사곡
    svd_similar = svd_most_similar(song_id, svd_vocab_map, topn=5)

    log(f"    {'Song2Vec 유사곡':<35} {'SVD 유사곡':<35}")
    log(f"    {'-'*35} {'-'*35}")

    for i in range(5):
        # Song2Vec
        w_song, w_score = w2v_similar[i]
        w_artist = song_meta.get(w_song, {}).get('artist_name', '?')
        w_str = f"{w_artist[:15]} ({w_score:.3f})"

        # SVD
        if i < len(svd_similar):
            s_song, s_score = svd_similar[i]
            s_artist = song_meta.get(s_song, {}).get('artist_name', '?')
            s_str = f"{s_artist[:15]} ({s_score:.3f})"
        else:
            s_str = "N/A"

        log(f"    {i+1}. {w_str:<33} {i+1}. {s_str:<33}")

    # Jaccard 유사도: 두 방법의 Top-10 겹침 정도
    w2v_top10 = set(s for s, _ in song2vec.wv.most_similar(song_id, topn=10))
    svd_top10 = set(s for s, _ in svd_most_similar(song_id, svd_vocab_map, topn=10))
    overlap = len(w2v_top10 & svd_top10)
    log(f"    → Top-10 겹침: {overlap}/10 ({overlap*10}%)")

del svd_song_map, svd_vocab_map, song_factors_svd
gc.collect()


# =====================================================================
# Part 7: 핵심 정리
# =====================================================================
log("\n" + "=" * 70)
log("  Part 7: 핵심 정리 — 무엇을 배웠나")
log("=" * 70)

"""
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1 핵심 정리                                               │
│                                                                  │
│  1. Word2Vec은 "시퀀스에서 컨텍스트를 학습하는" 범용 기법이다.      │
│     - 자연어 → 음악, 상품구매, 웹브라우징 등 모든 시퀀스에 적용 가능 │
│                                                                  │
│  2. Skip-gram은 각 곡(단어)에서 주변 곡을 예측하며 임베딩을 학습     │
│     - 이 과정에서 "비슷한 맥락에서 나오는 곡"이 가까운 벡터를 갖게 됨│
│                                                                  │
│  3. Negative Sampling으로 효율적 학습이 가능                       │
│     - 전체 어휘 Softmax 대신, 소수의 랜덤 Negative 샘플만 사용     │
│                                                                  │
│  4. SVD와 Song2Vec은 서로 다른 정보를 포착한다                     │
│     - SVD: 전역적 동시출현 패턴 (누가 뭘 들었나)                    │
│     - Song2Vec: 지역적 시퀀스 패턴 (어떤 순서로 들었나)             │
│     - 둘 다 쓰면 상호보완적!                                       │
│                                                                  │
│  5. 하지만 Song2Vec에도 한계가 있다:                                │
│     - 곡 ID만 사용 → 메타데이터(장르, 아티스트) 활용 불가            │
│     - 유저 개인화 없음 → 모든 유저에 같은 임베딩                    │
│     - 고정 윈도우 → 장거리 의존성 포착 불가                        │
│                                                                  │
│  → 이 한계를 극복하는 것이 Stage 2(NCF)와 Stage 3(Transformer)     │
└──────────────────────────────────────────────────────────────────┘
"""

log("""
  ┌────────────────────────────────────────────┐
  │  핵심 한 줄 요약                            │
  │                                            │
  │  SVD:  "같은 유저가 들은 곡은 비슷하다"       │
  │  W2V:  "연속으로 들은 곡은 비슷하다"          │
  │  NCF:  "유저와 곡의 관계를 신경망이 학습한다"  │  ← Stage 2
  │  Trans: "이 시퀀스 다음에 올 곡을 예측한다"    │  ← Stage 3
  └────────────────────────────────────────────┘
""")

log("  다음 단계: Stage 2 (NCF) — 신경망으로 유저-곡 상호작용 직접 학습")

del train, songs
gc.collect()

log(f"\n  Song2Vec 모델 통계:")
log(f"    학습된 곡 수: {vocab_size:,}")
log(f"    임베딩 차원: 128")
log(f"    학습 시간: {train_time:.1f}초")
log(f"    시각화: eda_output/19_song2vec_tsne.png")

log("\n" + "=" * 70)
log("  Stage 1 완료! 다음: Stage 2 (NCF)")
log("=" * 70)
