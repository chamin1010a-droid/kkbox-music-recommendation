"""
KKBOX Music Recommendation Challenge - EDA
===========================================
데이터를 전체적으로 탐색하고 시각화합니다.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# 저장 폴더
os.makedirs('eda_output', exist_ok=True)

print("=" * 60)
print("  KKBOX Music Recommendation Challenge - EDA")
print("=" * 60)

# =============================================================
# 1. 데이터 로드
# =============================================================
print("\n📂 데이터 로드 중...")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
songs = pd.read_csv('songs.csv')
members = pd.read_csv('members.csv')
extra = pd.read_csv('song_extra_info.csv')

print(f"  train:    {train.shape[0]:>10,}행 × {train.shape[1]}열")
print(f"  test:     {test.shape[0]:>10,}행 × {test.shape[1]}열")
print(f"  songs:    {songs.shape[0]:>10,}행 × {songs.shape[1]}열")
print(f"  members:  {members.shape[0]:>10,}행 × {members.shape[1]}열")
print(f"  extra:    {extra.shape[0]:>10,}행 × {extra.shape[1]}열")

# =============================================================
# 2. 결측치 분석
# =============================================================
print("\n" + "=" * 60)
print("🔍 결측치 분석")
print("=" * 60)

for name, df in [('train', train), ('test', test), ('songs', songs), 
                  ('members', members), ('extra', extra)]:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(f"\n  [{name}] 결측치:")
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            print(f"    {col:25s}: {cnt:>10,}개 ({pct:.1f}%)")
    else:
        print(f"\n  [{name}] 결측치 없음 ✅")

# =============================================================
# 3. Target 분포
# =============================================================
print("\n" + "=" * 60)
print("🎯 Target 분포 (재청취 여부)")
print("=" * 60)

target_counts = train['target'].value_counts()
print(f"  1 (재청취):   {target_counts[1]:>10,}건 ({target_counts[1]/len(train)*100:.1f}%)")
print(f"  0 (미재청취): {target_counts[0]:>10,}건 ({target_counts[0]/len(train)*100:.1f}%)")

fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#FF6B6B', '#4ECDC4']
bars = ax.bar(['미재청취 (0)', '재청취 (1)'], 
              [target_counts[0], target_counts[1]], 
              color=colors, edgecolor='white', linewidth=2)
for bar, val in zip(bars, [target_counts[0], target_counts[1]]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30000, 
            f'{val:,}\n({val/len(train)*100:.1f}%)', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Target 분포', fontsize=14, fontweight='bold')
ax.set_ylabel('건수')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('eda_output/01_target_distribution.png', dpi=150)
plt.close()
print("  → 01_target_distribution.png 저장 완료")

# =============================================================
# 4. Source 분석 (청취 맥락)
# =============================================================
print("\n" + "=" * 60)
print("📱 청취 맥락 분석 (Source)")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# source_system_tab
tab_target = train.groupby('source_system_tab')['target'].agg(['mean', 'count']).sort_values('count', ascending=False)
print("\n  [source_system_tab] 재청취율:")
for idx, row in tab_target.iterrows():
    print(f"    {idx:20s}: {row['mean']:.3f} ({row['count']:>10,}건)")

ax = axes[0]
colors_tab = plt.cm.viridis(np.linspace(0.2, 0.8, len(tab_target)))
bars = ax.barh(tab_target.index[::-1], tab_target['mean'][::-1], color=colors_tab)
ax.set_xlabel('재청취율')
ax.set_title('탭별 재청취율', fontsize=12, fontweight='bold')
ax.axvline(x=train['target'].mean(), color='red', linestyle='--', alpha=0.7, label=f"전체 평균 {train['target'].mean():.3f}")
ax.legend()

# source_screen_name (상위 10개)
screen_target = train.groupby('source_screen_name')['target'].agg(['mean', 'count']).sort_values('count', ascending=False).head(10)
print("\n  [source_screen_name] 재청취율 (상위 10):")
for idx, row in screen_target.iterrows():
    print(f"    {idx:25s}: {row['mean']:.3f} ({row['count']:>10,}건)")

ax = axes[1]
colors_screen = plt.cm.plasma(np.linspace(0.2, 0.8, len(screen_target)))
bars = ax.barh(screen_target.index[::-1], screen_target['mean'][::-1], color=colors_screen)
ax.set_xlabel('재청취율')
ax.set_title('화면별 재청취율 (Top 10)', fontsize=12, fontweight='bold')
ax.axvline(x=train['target'].mean(), color='red', linestyle='--', alpha=0.7)

# source_type
type_target = train.groupby('source_type')['target'].agg(['mean', 'count']).sort_values('count', ascending=False)
print("\n  [source_type] 재청취율:")
for idx, row in type_target.iterrows():
    print(f"    {idx:25s}: {row['mean']:.3f} ({row['count']:>10,}건)")

ax = axes[2]
colors_type = plt.cm.magma(np.linspace(0.2, 0.8, len(type_target)))
bars = ax.barh(type_target.index[::-1], type_target['mean'][::-1], color=colors_type)
ax.set_xlabel('재청취율')
ax.set_title('소스 타입별 재청취율', fontsize=12, fontweight='bold')
ax.axvline(x=train['target'].mean(), color='red', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('eda_output/02_source_analysis.png', dpi=150)
plt.close()
print("  → 02_source_analysis.png 저장 완료")

# =============================================================
# 5. 사용자 분석 (Members)
# =============================================================
print("\n" + "=" * 60)
print("👤 사용자 분석 (Members)")
print("=" * 60)

# 나이 분포 (비정상 값 필터링)
valid_age = members[(members['bd'] > 5) & (members['bd'] < 80)]['bd']
print(f"\n  유효 나이 범위 (5~80세): {len(valid_age):,}명 / {len(members):,}명 ({len(valid_age)/len(members)*100:.1f}%)")
print(f"  나이 통계: 평균 {valid_age.mean():.1f}세, 중위수 {valid_age.median():.0f}세")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 나이 분포
ax = axes[0, 0]
ax.hist(valid_age, bins=75, color='#667EEA', edgecolor='white', alpha=0.85)
ax.set_title('사용자 나이 분포 (유효 값)', fontsize=12, fontweight='bold')
ax.set_xlabel('나이')
ax.set_ylabel('인원수')
ax.axvline(x=valid_age.median(), color='red', linestyle='--', label=f'중위수: {valid_age.median():.0f}세')
ax.legend()

# 성별 분포
ax = axes[0, 1]
gender_counts = members['gender'].value_counts()
print(f"\n  성별 분포:")
for idx, val in gender_counts.items():
    print(f"    {idx}: {val:,}명")
print(f"    미입력: {members['gender'].isnull().sum():,}명")

labels = list(gender_counts.index) + ['미입력']
sizes = list(gender_counts.values) + [members['gender'].isnull().sum()]
colors_gender = ['#FF6B6B', '#4ECDC4', '#95A5A6']
ax.pie(sizes, labels=labels, colors=colors_gender, autopct='%1.1f%%', startangle=90)
ax.set_title('성별 분포', fontsize=12, fontweight='bold')

# 가입 채널
ax = axes[1, 0]
reg_counts = members['registered_via'].value_counts().sort_index()
print(f"\n  가입 채널:")
for idx, val in reg_counts.items():
    print(f"    {idx}: {val:,}명")

ax.bar(reg_counts.index.astype(str), reg_counts.values, color='#A78BFA', edgecolor='white')
ax.set_title('가입 채널 (registered_via)', fontsize=12, fontweight='bold')
ax.set_xlabel('채널 코드')
ax.set_ylabel('인원수')

# 도시 분포
ax = axes[1, 1]
city_counts = members['city'].value_counts().sort_index()
print(f"\n  도시 분포 (상위 5):")
for idx, val in city_counts.head(5).items():
    print(f"    도시 {idx}: {val:,}명")

ax.bar(city_counts.index.astype(str), city_counts.values, color='#F59E0B', edgecolor='white')
ax.set_title('도시 분포', fontsize=12, fontweight='bold')
ax.set_xlabel('도시 코드')
ax.set_ylabel('인원수')

plt.tight_layout()
plt.savefig('eda_output/03_member_analysis.png', dpi=150)
plt.close()
print("  → 03_member_analysis.png 저장 완료")

# =============================================================
# 6. 곡 분석 (Songs)
# =============================================================
print("\n" + "=" * 60)
print("🎵 곡 분석 (Songs)")
print("=" * 60)

# 곡 길이 분석
song_len_min = songs['song_length'] / 60000  # ms → min
valid_len = song_len_min[(song_len_min > 0.5) & (song_len_min < 15)]
print(f"\n  곡 길이 통계 (0.5~15분):")
print(f"    평균: {valid_len.mean():.1f}분, 중위수: {valid_len.median():.1f}분")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 곡 길이 분포
ax = axes[0, 0]
ax.hist(valid_len, bins=100, color='#10B981', edgecolor='white', alpha=0.85)
ax.set_title('곡 길이 분포', fontsize=12, fontweight='bold')
ax.set_xlabel('길이 (분)')
ax.set_ylabel('곡 수')
ax.axvline(x=valid_len.median(), color='red', linestyle='--', label=f'중위수: {valid_len.median():.1f}분')
ax.legend()

# 언어 분포
ax = axes[0, 1]
lang_counts = songs['language'].value_counts().head(10)
print(f"\n  언어 분포 (상위 10):")
for idx, val in lang_counts.items():
    print(f"    언어코드 {idx:.0f}: {val:,}곡")

ax.barh([f"언어 {int(x)}" for x in lang_counts.index[::-1]], lang_counts.values[::-1], 
        color='#EC4899', edgecolor='white')
ax.set_title('언어별 곡 수 (Top 10)', fontsize=12, fontweight='bold')
ax.set_xlabel('곡 수')

# 장르 분석 (genre_ids가 |로 구분된 복수 값)
print(f"\n  장르 분석:")
genre_series = songs['genre_ids'].dropna().astype(str)
# 단일 장르 vs 복수 장르
multi_genre = genre_series.str.contains(r'\|', regex=True).sum()
single_genre = len(genre_series) - multi_genre
print(f"    단일 장르: {single_genre:,}곡")
print(f"    복수 장르: {multi_genre:,}곡")

# 전체 장르 빈도
all_genres = genre_series.str.split('|').explode()
genre_freq = all_genres.value_counts().head(15)
print(f"    고유 장르 수: {all_genres.nunique()}")
print(f"    상위 5 장르: {list(genre_freq.head(5).index)}")

ax = axes[1, 0]
ax.barh([f"장르 {x}" for x in genre_freq.index[::-1]], genre_freq.values[::-1],
        color='#8B5CF6', edgecolor='white')
ax.set_title('장르별 곡 수 (Top 15)', fontsize=12, fontweight='bold')
ax.set_xlabel('곡 수')

# 아티스트 인기도
ax = axes[1, 1]
artist_song_count = songs['artist_name'].value_counts()
print(f"\n  아티스트 분석:")
print(f"    고유 아티스트 수: {artist_song_count.shape[0]:,}")
print(f"    상위 5 아티스트 (곡 수): {list(artist_song_count.head(5).items())}")
print(f"    1곡만 있는 아티스트: {(artist_song_count == 1).sum():,}명 ({(artist_song_count == 1).sum()/len(artist_song_count)*100:.1f}%)")

# 아티스트당 곡 수 분포
bins = [1, 2, 5, 10, 20, 50, 100, 1000]
labels_bin = ['1곡', '2~4곡', '5~9곡', '10~19곡', '20~49곡', '50~99곡', '100+곡']
artist_bins = pd.cut(artist_song_count, bins=bins, labels=labels_bin, right=False)
artist_bin_counts = artist_bins.value_counts().sort_index()
ax.bar(artist_bin_counts.index.astype(str), artist_bin_counts.values, color='#F97316', edgecolor='white')
ax.set_title('아티스트당 곡 수 분포', fontsize=12, fontweight='bold')
ax.set_xlabel('곡 수 구간')
ax.set_ylabel('아티스트 수')
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

plt.tight_layout()
plt.savefig('eda_output/04_song_analysis.png', dpi=150)
plt.close()
print("  → 04_song_analysis.png 저장 완료")

# =============================================================
# 7. 사용자-곡 행동 패턴 분석
# =============================================================
print("\n" + "=" * 60)
print("📊 사용자-곡 행동 패턴")
print("=" * 60)

# 사용자별 청취 횟수
user_listen_count = train['msno'].value_counts()
print(f"\n  사용자별 청취 횟수:")
print(f"    고유 사용자: {user_listen_count.shape[0]:,}명")
print(f"    평균: {user_listen_count.mean():.1f}회, 중위수: {user_listen_count.median():.0f}회")
print(f"    최소: {user_listen_count.min()}회, 최대: {user_listen_count.max():,}회")

# 곡별 청취 횟수
song_listen_count = train['song_id'].value_counts()
print(f"\n  곡별 청취 횟수:")
print(f"    고유 곡 수: {song_listen_count.shape[0]:,}곡")
print(f"    평균: {song_listen_count.mean():.1f}회, 중위수: {song_listen_count.median():.0f}회")
print(f"    1회만 등장: {(song_listen_count == 1).sum():,}곡 ({(song_listen_count == 1).sum()/len(song_listen_count)*100:.1f}%)")

# 사용자별 재청취율
user_target_rate = train.groupby('msno')['target'].mean()
print(f"\n  사용자별 재청취율:")
print(f"    평균: {user_target_rate.mean():.3f}")
print(f"    표준편차: {user_target_rate.std():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 사용자별 청취 횟수 분포
ax = axes[0]
ax.hist(user_listen_count.clip(upper=1000), bins=100, color='#667EEA', edgecolor='white', alpha=0.85)
ax.set_title('사용자별 청취 횟수 분포', fontsize=12, fontweight='bold')
ax.set_xlabel('청취 횟수 (1000 이상 클리핑)')
ax.set_ylabel('사용자 수')

# 곡별 청취 횟수 분포
ax = axes[1]
ax.hist(song_listen_count.clip(upper=100), bins=100, color='#10B981', edgecolor='white', alpha=0.85)
ax.set_title('곡별 청취 횟수 분포', fontsize=12, fontweight='bold')
ax.set_xlabel('청취 횟수 (100 이상 클리핑)')
ax.set_ylabel('곡 수')

# 사용자별 재청취율 분포
ax = axes[2]
ax.hist(user_target_rate, bins=100, color='#EC4899', edgecolor='white', alpha=0.85)
ax.set_title('사용자별 재청취율 분포', fontsize=12, fontweight='bold')
ax.set_xlabel('재청취율')
ax.set_ylabel('사용자 수')
ax.axvline(x=user_target_rate.mean(), color='red', linestyle='--', label=f'평균: {user_target_rate.mean():.3f}')
ax.legend()

plt.tight_layout()
plt.savefig('eda_output/05_behavior_pattern.png', dpi=150)
plt.close()
print("  → 05_behavior_pattern.png 저장 완료")

# =============================================================
# 8. ISRC 분석 (발매 연도/국가 추출)
# =============================================================
print("\n" + "=" * 60)
print("🌍 ISRC 기반 분석 (발매 연도/국가)")
print("=" * 60)

# ISRC 포맷: CC-XXX-YY-NNNNN (CC=국가, YY=연도)
valid_isrc = extra['isrc'].dropna()
print(f"  ISRC 보유: {len(valid_isrc):,}곡 / {len(extra):,}곡 ({len(valid_isrc)/len(extra)*100:.1f}%)")

# 국가 추출 (앞 2글자)
country = valid_isrc.str[:2]
country_counts = country.value_counts().head(10)
print(f"\n  발매 국가 (상위 10):")
for idx, val in country_counts.items():
    print(f"    {idx}: {val:,}곡")

# 연도 추출 (7~8번째 글자)
year_str = valid_isrc.str[5:7].astype(int)
year_full = year_str.apply(lambda x: 1900 + x if x > 30 else 2000 + x)
year_counts = year_full.value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 국가별 분포
ax = axes[0]
ax.barh(country_counts.index[::-1], country_counts.values[::-1], color='#06B6D4', edgecolor='white')
ax.set_title('발매 국가 분포 (ISRC 기준)', fontsize=12, fontweight='bold')
ax.set_xlabel('곡 수')

# 연도별 분포
ax = axes[1]
recent_years = year_counts[year_counts.index >= 1980]
ax.bar(recent_years.index, recent_years.values, color='#8B5CF6', edgecolor='white', alpha=0.85)
ax.set_title('발매 연도 분포 (ISRC 기준)', fontsize=12, fontweight='bold')
ax.set_xlabel('연도')
ax.set_ylabel('곡 수')

plt.tight_layout()
plt.savefig('eda_output/06_isrc_analysis.png', dpi=150)
plt.close()
print("  → 06_isrc_analysis.png 저장 완료")

# =============================================================
# 9. Train/Test 겹침 분석
# =============================================================
print("\n" + "=" * 60)
print("🔗 Train/Test 데이터 겹침 분석")
print("=" * 60)

train_users = set(train['msno'].unique())
test_users = set(test['msno'].unique())
common_users = train_users & test_users
print(f"\n  사용자:")
print(f"    Train 고유: {len(train_users):,}명")
print(f"    Test 고유:  {len(test_users):,}명")
print(f"    겹치는 사용자: {len(common_users):,}명 ({len(common_users)/len(test_users)*100:.1f}%)")

train_songs = set(train['song_id'].unique())
test_songs = set(test['song_id'].unique())
common_songs = train_songs & test_songs
print(f"\n  곡:")
print(f"    Train 고유: {len(train_songs):,}곡")
print(f"    Test 고유:  {len(test_songs):,}곡")
print(f"    겹치는 곡: {len(common_songs):,}곡 ({len(common_songs)/len(test_songs)*100:.1f}%)")

# Train에만 있는 / Test에만 있는
test_only_users = test_users - train_users
test_only_songs = test_songs - train_songs
print(f"\n  ⚠️ Test에만 있는 (Cold Start 문제):")
print(f"    사용자: {len(test_only_users):,}명 ({len(test_only_users)/len(test_users)*100:.1f}%)")
print(f"    곡: {len(test_only_songs):,}곡 ({len(test_only_songs)/len(test_songs)*100:.1f}%)")

# =============================================================
# 10. 요약
# =============================================================
print("\n" + "=" * 60)
print("📝 EDA 요약")
print("=" * 60)
print("""
  ✅ Target 균형: 거의 50:50으로 균형 잡혀 있음
  ✅ 청취 맥락: source_system_tab/screen/type이 재청취율에 큰 영향
  ⚠️ 결측치: songs의 composer/lyricist에 결측 다수
  ⚠️ Members: 나이(bd) 비정상 값 다수, 성별 미입력 다수
  ⚠️ Cold Start: Test에만 있는 사용자/곡 존재
  💡 ISRC: 발매 국가/연도 추출 가능 → 유용한 피처
  💡 사용자별 재청취율 편차 큼 → 개인화 피처 중요
""")

print("📁 모든 시각화가 eda_output/ 폴더에 저장되었습니다.")
print("   완료! ✨")
