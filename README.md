# Computer Vision 이상치 탐지 알고리즘 경진대회
## - 멀티캠퍼스 AI 활용 프로젝트 (AI, 빅데이터, IoT, 클라우드)


프로젝트 기간 : `2022/04/26 ~ 2022/05/19`


경진대회 기간 : `2022/04/01 ~ 2022/05/13`


### 1. 대회 배경
- Computer Vision 이상 탐지(Anomaly Detection) 알고리즘 경진대회
- MVtec AD Dataset에는 15 종류의 사물이 존재하며, 사물의 상태에 따라 이미지 분류
- 사물의 종류를 분류하고, 정상 샘플과 비정상(이상치) 샘플을 분류
- 평가 산식 : Macro-F1 score
- 불균형 데이터 셋을 학습하여 사물의 상태를 잘 분류할 수 있는 알고리즘 개발

### 2. 주제
사물의 종류와 상태를 분류하는 컴퓨터 비전 알고리즘 개발

### 3. 주최 / 주관
주최 / 주관: [데이콘](https://dacon.io/competitions/official/235894/overview/description)

---

# Summary
- Tools : Pytorch, PuTTy(AWS 접속)
- IDE : Jupyter Notebook
- 데이터 불균형으로 인해 Data Augmentation 후 stratified k - fold 적용, ensemble 진행
- Model : efficientnet, regnet 모두 진행하였으나 학습 시간 제한 문제로 상대적으로 시간 대비 성능이 나은 regnet으로 최종 진행
  - 학습 진행 모델 : regnety_040, regnety_064, regnety_080, regnety_120, regnety_160, regnety_320
  - 최종 선택 모델 : regnety_160
  
# [EDA](EDA.md)
# [Code](Code.md)
