# 🎵 KKBox 음악 추천 대회 (Kaggle)

> KKBox Music Recommendation Challenge에 참가하며, **피처 엔지니어링 중심의 접근**으로 유저의 재청취 확률을 예측합니다.
> pipeline v3 → v9까지 반복 실험하며, **데이터 누수(Target Leakage) 발견 → 해결** 과정을 거쳤습니다.

---

## 📌 프로젝트 요약

| 항목 | 내용 |
|------|------|
| **대회** | [KKBox Music Recommendation Challenge](https://www.kaggle.com/c/kkbox-music-recommendation-challenge) |
| **목표** | 유저가 곡을 재청취할 확률 예측 (AUC 기준) |
| **모델** | LightGBM (Gradient Boosting) |
| **데이터** | 730만 행 (train), 유저 3만명, 곡 35만곡 |
| **최종 성과** | Public LB AUC 0.67+ |

---

## 🔄 파이프라인 진화 과정

이 프로젝트의 핵심은 **실험 → 실패 → 원인 분석 → 개선**의 반복입니다.

```
v3  │ 기본 피처 (source, song_length, language)
    │ → Local AUC 0.72, Kaggle 0.62
    │ 
v4  │ Target Encoding 추가 (유저별/곡별 재청취율)
    │ → Local AUC 0.96 🤯 ... Kaggle 0.58 💀
    │ → 데이터 누수 발견!
    │
v5  │ Out-of-Fold Target Encoding + Smoothing으로 누수 해결
    │ → Local AUC 0.71, Kaggle 0.65
    │
v7  │ 시퀀스 피처 추가 (직전 곡과의 관계)
    │ → Kaggle 0.65 유지
    │
v8  │ SVD 임베딩 (유저-곡 잠재 벡터 32차원)
    │ → Kaggle 0.67 달성
    │
v9  │ Song2Vec 스타디 (Word2Vec으로 곡 임베딩)
    │ → 실험적 시도, 유의미한 개선 없음
```

---

## 🚨 핵심 교훈: Target Leakage 사건

v4에서 Local AUC가 **0.96**으로 치솟았지만 Kaggle 점수는 **0.58**로 폭락.

```
원인:
  Target Encoding을 할 때 각 행의 target 값이 자기 자신의 인코딩에 포함됨
  → 모델이 "정답을 보고 정답을 맞추는" 상황
  → Local 검증에서는 높은 점수, 실전에서는 무의미

해결 (v5):
  Out-of-Fold Target Encoding 적용
  → 5-Fold로 분할, 각 fold의 target은 나머지 fold의 통계로만 계산
  → Smoothing 추가: count가 적은 카테고리는 전체 평균으로 수렴
  → Local 0.71, Kaggle 0.65로 정상화
```

이 경험은 **"좋아 보이는 점수가 항상 진짜는 아니다"**는 것을 체감하게 해준 사건이었습니다.

---

## 📊 피처 엔지니어링 상세

### 기본 피처
- `source_system_tab`, `source_screen_name`, `source_type` → Label Encoding
- `song_length`, `language`, `genre_ids` (1st genre 추출)
- `registered_via`, `city` (유저 프로필)

### 통계 피처 (OOF Target Encoding)
- 유저별 재청취율, 곡별 재청취율, 아티스트별 재청취율
- 유저의 전체 청취 횟수, 곡의 전체 등장 횟수

### SVD 임베딩 (v8~)
- 유저-곡 상호작용 행렬 → TruncatedSVD → 32차원 잠재 벡터
- 유저 벡터: "이 유저는 어떤 취향인가"
- 곡 벡터: "이 곡은 어떤 특성인가"

### 시퀀스 피처 (v7~)
- 직전 곡과 같은 아티스트인가?
- 직전 곡과 같은 장르인가?
- 연속 청취 길이 (세션 위치)

---

## 📁 프로젝트 구조

```
kkbox/
├── eda.py                    # 탐색적 데이터 분석 (시각화 10종)
├── pipeline.py               # v1 기본 파이프라인
├── pipeline_v3.py             # v3 Source 피처 추가
├── pipeline_v4.py             # v4 Target Encoding (누수 발생 버전)
├── pipeline_v5.py             # v5 OOF Target Encoding (누수 해결)
├── pipeline_v7.py             # v7 시퀀스 피처 추가
├── pipeline_v8.py             # v8 SVD 임베딩 (최종)
├── pipeline_v9_experiment.py  # v9 실험 (Song2Vec)
├── song2vec_study.py          # Song2Vec 학습 스크립트
└── eda_output/                # EDA 시각화 결과
```

---

## 🔧 기술 스택

- **모델**: LightGBM (gbdt, 2000 trees, lr=0.1)
- **검증**: 5-Fold Stratified CV
- **피처**: OOF Target Encoding, SVD 임베딩, 시퀀스 피처
- **라이브러리**: pandas, numpy, scikit-learn, lightgbm, scipy

---

## 💡 느낀 점

- **데이터 누수는 실전에서만 드러난다**: local 점수만 보고 자만하면 안 됨
- **피처의 질이 모델의 양보다 중요**: 트리 2000개→4000개보다 SVD 32차원 하나가 효과적
- **추천 문제는 Cold Start가 핵심**: 신규 유저/곡에 대한 대응이 성능을 좌우함
- → 이 한계를 넘기 위해 딥러닝(NCF, SASRec) 학습을 시작하게 된 계기

---

## 🔗 관련 프로젝트

- [음원 생애주기 추천 알고리즘](https://github.com/chamin1010a-droid/music-recomender-lifecycle-based) — 개인 청취 데이터 기반 규칙 기반 추천
