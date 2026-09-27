# 기계학습·데이터 분석 구현 실습

회귀·분류·군집화부터 생성 모델과 그래프 학습까지, KAIST 교과 과정에서 구현한 알고리즘을 모았습니다. 모델의 학습 과정과 데이터 처리 방식을 코드로 비교하는 데 초점을 두었습니다.

## 구현

| 폴더 | 내용 |
|---|---|
| `ee214/` | 회귀·분류·군집화와 신경망 실습 |
| `machine-learning/` | bias–variance, 로지스틱 회귀, 심층 신경망, Langevin sampling |
| `deep-learning/` | VAE를 이용한 이미지 생성 |
| `big-data/` | 데이터 처리 알고리즘, GraphSAGE, Bloom filter와 Spark 실습 |

알고리즘 구현, 모델 학습, 결과 분석을 수행했습니다. 교과 실습의 인터페이스와 기반 코드를 사용하며, 학습 데이터는 별도로 준비해야 합니다.

## 실행

Python 3에서 필요한 라이브러리를 설치하고 notebook을 엽니다.

```bash
python -m pip install -r requirements.txt
jupyter lab
```

- Notebook의 데이터 경로를 입력 파일 위치에 맞춥니다. Colab Drive 경로를 사용하는 셀은 로컬 경로로 바꿉니다.
- CUDA는 장비에 맞는 PyTorch 버전을 설치합니다. Spark 실습에는 Java와 Spark 환경이 필요합니다.
- `requirements.txt`는 의존성 목록입니다. Legacy torchtext API를 사용하는 notebook은 해당 API를 지원하는 버전을 사용해야 합니다.
- Python 스크립트의 입력 형식과 인자는 각 파일의 `main` 또는 하단 실행 코드를 참고합니다.

## 테스트

Python 파일과 notebook 코드의 문법 검사를 통과했습니다. Bloom filter는 합성 입력 200개를 삽입한 뒤 false negative가 없음을 확인했습니다. 전체 데이터 학습, GPU 학습, Spark 분산 실행은 테스트 범위에 포함되지 않습니다.
