# 기계학습·데이터 분석 구현 실습

KAIST 교과 과정에서 수행한 개인 구현을 모았습니다. EE214의 회귀·분류·군집화와 신경망, 기계학습의 bias–variance·로지스틱 회귀·신경망·Langevin sampling, VAE 이미지 생성, 빅데이터 알고리즘의 구현을 담습니다.

| 폴더 | 구현 |
|---|---|
| `ee214/` | Assignment 1–3 개인 제출 notebook의 코드 |
| `machine-learning/` | bias–variance, logistic regression, deep neural network, Langevin sampling |
| `deep-learning/` | VAE 이미지 생성 실습 |
| `big-data/` | HW0–3 개인 제출 Python, HW4 GraphSAGE와 Bloom filter/Spark 구현 |

본인은 과제별 알고리즘과 학습·분석 코드를 작성·실행하고 결과를 정리했습니다. 수업에서 제공한 인터페이스와 데이터 사용을 전제로 진행한 학습 프로젝트입니다. 보관 제출본에서 구현 셀을 선별했으며 문제지·강의·제공 정답·데이터와 notebook의 저장 출력은 포함하지 않습니다. 원 과제의 인터페이스·주석과 개인 구현이 함께 남은 곳은 수업 기반 구현이며 전체 기반 코드를 독자적으로 설계했다는 뜻은 아닙니다.

## 실행 환경

Python 3 환경에서 `python -m pip install -r requirements.txt` 후 `jupyter lab`으로 notebook을 엽니다. notebook의 데이터 경로는 보유한 입력 위치에 맞춥니다. 데이터 없는 저장소이므로 전체 셀을 그대로 실행하면 데이터 로딩 단계에서 멈출 수 있습니다. PyTorch의 CUDA 버전은 사용하는 장비에 맞게 설치합니다. PySpark 실습은 Java와 Spark 환경도 필요합니다.

각 Python 파일의 `main` 또는 파일 하단 실행 부분에서 인자와 입력 파일을 확인할 수 있습니다. 원 실험 환경의 모든 패키지 버전은 기록되지 않아 `requirements.txt`는 필요한 라이브러리 목록이며 당시 환경 lockfile은 아닙니다.

## 결과와 확인 범위

개인 제출 코드가 보존된 항목을 공개용으로 정리했습니다. 현재 공개판은 Python 문법 및 notebook 구조를 검사했으며 전체 데이터 학습·Spark 분산 작업·GPU 실험을 다시 수행한 결과는 아닙니다. 과거 수업의 점수나 모델 성능을 새 실험 결과로 주장하지 않습니다.

일부 과거 notebook은 Colab의 Drive mount와 legacy torchtext API를 사용합니다. 로컬에서는 해당 데이터 경로 셀을 수정하고 원 notebook과 맞는 torchtext 환경을 선택해야 합니다. 모든 notebook을 최신 패키지로 전체 재실행한 상태는 아닙니다.
