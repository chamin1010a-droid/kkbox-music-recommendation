"""
Song2Vec Part 5~7 만 실행하는 이어하기 스크립트
(Word2Vec 학습을 다시 하지 않고, 빠르게 재학습 후 시각화/비교만 수행)
"""
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from scipy.sparse import csr_matrix
from collections import Counter
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import time
import gc
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

SEED = 42
np.random.seed(SEED)

# === 데이터 로드 (최소한만) ===
log("[1] 데이터 로드...")
t0 = time.time()
train = pd.read_csv('train.csv', dtype={'msno': str, 'song_id': str, 'target': 'int8'},
                     usecols=['msno', 'song_id', 'target'])
songs = pd.read_csv('songs.csv', dtype={'song_id': str, 'artist_name': str, 'genre_ids': str, 'language': 'float32'},
                     usecols=['song_id', 'artist_name', 'genre_ids', 'language'])
song_meta = songs.set_index('song_id')[['artist_name', 'genre_ids', 'language']].to_dict('index')
song_counts = train['song_id'].value_counts()
log(f"  완료 ({time.time()-t0:.1f}초)")

# === Word2Vec 빠르게 재학습 (epochs=5로 줄여서) ===
log("[2] Song2Vec 학습 (epochs=5, 빠른 버전)...")
t0 = time.time()
user_sequences = train.groupby('msno')['song_id'].apply(list).reset_index()
user_sequences.columns = ['msno', 'sequence']
user_sequences['seq_len'] = user_sequences['sequence'].apply(len)
sentences = user_sequences[user_sequences['seq_len'] >= 3]['sequence'].tolist()

song2vec = Word2Vec(sentences=sentences, vector_size=128, window=10, min_count=5,
                    sg=1, negative=10, workers=4, epochs=5, seed=SEED)
log(f"  완료 ({time.time()-t0:.1f}초), 어휘: {len(song2vec.wv):,}곡")
del sentences, user_sequences; gc.collect()

# === Part 5: t-SNE 시각화 ===
log("\n[3] t-SNE 시각화...")
t0 = time.time()
N_VIS = 2000
top_n_songs = song_counts.head(N_VIS * 2).index.tolist()
vis_songs = [s for s in top_n_songs if s in song2vec.wv][:N_VIS]
vis_vectors = np.array([song2vec.wv[s] for s in vis_songs])

vis_genres = []
vis_langs = []
for s in vis_songs:
    meta = song_meta.get(s, {})
    genre = meta.get('genre_ids', '')
    first_genre = genre.split('|')[0] if isinstance(genre, str) and genre else 'unknown'
    lang = meta.get('language', -1)
    lang = lang if lang == lang else -1
    vis_genres.append(first_genre)
    vis_langs.append(lang)

genre_counter = Counter(vis_genres)
top_genres = [g for g, _ in genre_counter.most_common(10)]
genre_colors = {}
cmap = plt.cm.tab10
for i, g in enumerate(top_genres):
    genre_colors[g] = cmap(i)

log(f"  시각화 대상: {len(vis_songs)}곡")
log("  t-SNE 차원 축소 중...")
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
vis_2d = tsne.fit_transform(vis_vectors)
log(f"  t-SNE 완료 ({time.time()-t0:.1f}초)")

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# 좌: 장르별
ax = axes[0]
other_mask = np.array([g not in top_genres for g in vis_genres])
ax.scatter(vis_2d[other_mask, 0], vis_2d[other_mask, 1], c='lightgray', s=8, alpha=0.3, label='기타')
for genre in top_genres:
    mask = np.array([g == genre for g in vis_genres])
    if mask.sum() > 0:
        ax.scatter(vis_2d[mask, 0], vis_2d[mask, 1], c=[genre_colors[genre]], s=15, alpha=0.6, label=f'장르 {genre}')
ax.set_title('Song2Vec 임베딩 — 장르별 분포', fontsize=14, fontweight='bold')
ax.legend(fontsize=7, loc='upper right', ncol=2)
ax.set_xticks([]); ax.set_yticks([])

# 우: 언어별
ax = axes[1]
lang_map = {-1.0: '미상', 3.0: '중국어', 31.0: '일본어', 52.0: '영어', 10.0: '한국어', 17.0: '대만어'}
lang_colors_map = {3.0: '#FF6B6B', 31.0: '#4ECDC4', 52.0: '#45B7D1', 10.0: '#96CEB4', 17.0: '#FFEAA7'}
for lang_code, color in lang_colors_map.items():
    mask = np.array([l == lang_code for l in vis_langs])
    if mask.sum() > 0:
        label = lang_map.get(lang_code, f'Lang {int(lang_code)}')
        ax.scatter(vis_2d[mask, 0], vis_2d[mask, 1], c=color, s=15, alpha=0.6, label=label)
other_mask = np.array([l not in lang_colors_map for l in vis_langs])
ax.scatter(vis_2d[other_mask, 0], vis_2d[other_mask, 1], c='lightgray', s=8, alpha=0.3, label='기타')
ax.set_title('Song2Vec 임베딩 — 언어별 분포', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('Song2Vec (Skip-gram, 128dim, window=10) — t-SNE 시각화', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('eda_output/19_song2vec_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
log("  ✓ 시각화 저장: eda_output/19_song2vec_tsne.png")

# === Part 6: SVD vs Song2Vec 비교 ===
log("\n[4] SVD vs Song2Vec 비교...")
t0 = time.time()

msno_codes, msno_uniques = pd.factorize(train['msno'])
song_codes_arr, song_uniques = pd.factorize(train['song_id'])
interaction = csr_matrix(
    (np.ones(len(msno_codes), dtype=np.float32), (msno_codes, song_codes_arr)),
    shape=(len(msno_uniques), len(song_uniques))
)
interaction.data = np.minimum(interaction.data, 1.0)

svd = TruncatedSVD(n_components=128, random_state=SEED, n_iter=10)
user_factors = svd.fit_transform(interaction)
song_factors_svd = svd.components_.T
svd_song_map = {song_uniques[i]: song_factors_svd[i] for i in range(len(song_uniques))}
log(f"  SVD 완료 ({time.time()-t0:.1f}초), 설명 분산: {svd.explained_variance_ratio_.sum():.4f}")
del user_factors, interaction, svd; gc.collect()

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def svd_most_similar(song_id, svd_map, topn=5):
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

top_songs = song_counts.head(20).index.tolist()
test_songs = [s for s in top_songs if s in song2vec.wv and s in svd_song_map][:3]
svd_vocab_map = {s: svd_song_map[s] for s in song2vec.wv.index_to_key if s in svd_song_map}
log(f"  비교 가능 곡 수: {len(svd_vocab_map):,}")

log("\n  SVD vs Song2Vec 유사곡 비교:")
log("  " + "=" * 64)

for song_id in test_songs:
    meta = song_meta.get(song_id, {})
    artist = meta.get('artist_name', '?')
    log(f"\n  ▶ {artist} ({song_id[:20]}...)")

    w2v_similar = song2vec.wv.most_similar(song_id, topn=5)
    svd_similar = svd_most_similar(song_id, svd_vocab_map, topn=5)

    log(f"    {'Song2Vec 유사곡':<35} {'SVD 유사곡':<35}")
    log(f"    {'-'*35} {'-'*35}")

    for i in range(5):
        w_song, w_score = w2v_similar[i]
        w_artist = song_meta.get(w_song, {}).get('artist_name', '?')
        w_str = f"{str(w_artist)[:15]} ({w_score:.3f})"

        if i < len(svd_similar):
            s_song, s_score = svd_similar[i]
            s_artist = song_meta.get(s_song, {}).get('artist_name', '?')
            s_str = f"{str(s_artist)[:15]} ({s_score:.3f})"
        else:
            s_str = "N/A"

        log(f"    {i+1}. {w_str:<33} {i+1}. {s_str:<33}")

    w2v_top10 = set(s for s, _ in song2vec.wv.most_similar(song_id, topn=10))
    svd_top10 = set(s for s, _ in svd_most_similar(song_id, svd_vocab_map, topn=10))
    overlap = len(w2v_top10 & svd_top10)
    log(f"    → Top-10 겹침: {overlap}/10 ({overlap*10}%)")

# === 요약 ===
log("\n" + "=" * 70)
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
log("  Stage 1 완료! 다음: Stage 2 (NCF)")
log("=" * 70)
