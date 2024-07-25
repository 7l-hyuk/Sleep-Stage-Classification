# Sleep Stage Classification

[Sleep-EDF Database Expanded v1.0.0](https://www.physionet.org/content/sleep-edfx/1.0.0/)을 사용하여 sleep stage를 분류합니다. 이 프로젝트에서 sleep stage는 다음 네 가지로 구분합니다.

* Sleep Stage W
  
* LS(Sleep Stage 1 + Sleep Stage 2)

* DS(Sleep Stage 3 + Sleep Stage 4)

* Sleep Stage R

## 📄Related Work

이 프로젝트는 EEG 데이터를 다룹니다. EEG 데이터를 전처리 하고 머신러닝 모델을 학습시키기 위해서 다음과 같은 사전 지식이 필요합니다.

### 1. EEG Data
* [Wikipedia: Electroencephalography](https://en.wikipedia.org/wiki/Electroencephalography)

### 2. Band-pass filter
* [[EEG] Signal Filtering in Python](https://hayoonsong.github.io/study/2022-05-31-filter/)

### 3. Fourier Transform

* [[EEG] Signal Filtering in Python](https://hayoonsong.github.io/study/2022-05-31-filter/)
  
* [Classification of EEG Signals using Fast Fourier Transform
(FFT) and Adaptive Neuro-Fuzzy Inference System (ANFIS)](https://core.ac.uk/download/pdf/235583696.pdf)

## ⚙️Environment

### 1. Package Manager: [poetry](https://python-poetry.org/)

### 2. Tracking Server: [Mlflow](https://mlflow.org/)

### 3. Parameter Management: [Hydra](https://hydra.cc/)

```
python = "^3.11"
pyedflib = "^0.1.37"
matplotlib = "^3.9.0"
pandas = "^2.2.2"
numpy = "1.26.3"
scikit-learn = "^1.5.1"
xgboost = "^2.1.0"
pyspark = "^3.5.1"
hydra-core = "^1.3.2"
mlflow = "^2.14.2"
imbalanced-learn = "^0.12.3"
nbformat = ">=4.2.0"
```

## 🗂️디렉터리 구조

각 모듈의 API는 대부분 sleep_edf/src/main.ipynb에서 실습합니다. 데이터는 직접 다운받은 뒤 sleep_edf/data에 위치시키거나 적절한 경로에 넣고 코드 경로를 변경하면 됩니다.

```
sleep_edf
 ┣ conf ... hydra config file
 ┃ ┣ Bpass_Ftrans
 ┃ ┣ config.yaml
 ┃ ┗ README.md
 ┣ data ... sleep-edf dataset이 위치하는 디렉터리
 ┃ ┗ sleep-edf-database-expanded-1.0.0
 ┃ ┃ ┣ sleep-cassette
 ┃ ┃ ┣ sleep-telemetry
 ┣ experiment ... 실험과 관련된 디렉터리
 ┃ ┣ artifact01
 ┃ ┣ jobs
 ┃ ┃ ┣ write_latex.py ... latex 문서 작성과 관련된 모듈
 ┃ ┣ mlflow_setting
 ┃ ┃ ┗ mlflow_utils.py
 ┃ ┣ mlruns ... mlflow run file
 ┃ ┣ Performance ... 모델 성능괴 관련된 디렉터리
 ┃ ┃ ┣ 20240725_105500 ... 모델 성능 평가 latex 문서
 ┃ ┣ setting
 ┃ ┣ create_experiments.py ... mlflow experiment 생성
 ┃ ┣ main.py
 ┃ ┣ performance_eval.py ... 모델 성능 평가 latex 문서 작성
 ┃ ┗ README.md
 ┣ src
 ┃ ┣ data ... dataframe 생성 관련 디렉터리
 ┃ ┃ ┣ dataframe.py
 ┃ ┃ ┣ stats_utils.py
 ┃ ┣ model
 ┃ ┃ ┣ setting
 ┃ ┃ ┣ model.py
 ┃ ┃ ┣ train.py
 ┗ main.ipynb ... 각 모듈의 API를 이용하여 기본적인 학습 진행
```