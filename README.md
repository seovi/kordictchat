# scikit-learn 기반 사전 챗봇 프로젝트

이 프로젝트는 scikit-learn을 사용한 벡터 검색과 임베딩 모델을 통해 사전 데이터를 검색하고 질의에 응답하는 시스템입니다.

## 프로젝트 구성

- `generate_embeddings.py`: 사전 CSV 파일을 읽어 임베딩을 생성하는 스크립트
- `sklearn_vector_search.py`: scikit-learn을 활용한 벡터 검색 엔진 구현
- `app.py`: FastAPI 기반 웹 API 서버

## 설치 방법

<details>
<summary><b>Conda 환경 설정 (권장)</b> - 클릭하여 펼치기</summary>

Conda를 사용하면 의존성 문제를 쉽게 해결할 수 있습니다:

```bash
# Miniconda 설치: https://docs.conda.io/en/latest/miniconda.html

# 새 conda 환경 생성 (Python 3.9 사용)
conda create -n dictenv python=3.9

# 환경 활성화
conda activate dictenv

# 필수 패키지 설치
conda install -c conda-forge numpy==1.22.4 pandas==1.4.3 scikit-learn==1.1.2
conda install -c conda-forge tqdm

# 나머지 패키지 pip로 설치
pip install fastapi==0.95.2 uvicorn==0.22.0 python-dotenv==1.0.0
pip install openai==1.3.0 sentence-transformers==2.2.2
```
</details>

<details>
<summary><b>가상환경 + pip 사용 (대안)</b> - 클릭하여 펼치기</summary>

Python 3.9를 사용하는 경우:

```bash
# Python 3.9 설치: https://www.python.org/downloads/release/python-3913/

# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# 안정적인 버전의 패키지 설치
pip install fastapi==0.95.2 uvicorn==0.22.0
pip install numpy==1.22.4 pandas==1.4.3
pip install scikit-learn==1.1.2 sentence-transformers==2.2.2
pip install python-dotenv==1.0.0 openai==1.3.0 tqdm==4.66.1
```
</details>

<details>
<summary><b>환경 변수 설정 (선택 사항)</b> - 클릭하여 펼치기</summary>

OpenAI API를 사용하려면 `.env` 파일을 생성하고 다음 내용을 추가하세요:

```
OPENAI_API_KEY=your_api_key_here
```

API 키가 없어도 기본 검색 기능은 작동합니다.
</details>

## 실행 방법

<details>
<summary><b>1. 임베딩 생성</b> - 클릭하여 펼치기</summary>

```bash
python generate_embeddings.py --input dictionary.csv --output dictionary_with_embeddings.csv
```

옵션:
- `--input`: 원본 사전 CSV 파일 경로 (기본값: dictionary.csv)
- `--output`: 임베딩이 추가된 CSV 파일 저장 경로 (기본값: dictionary_with_embeddings.csv)
- `--model`: 사용할 SentenceTransformer 모델 (기본값: distiluse-base-multilingual-cased-v1)
- `--batch_size`: 배치 처리 크기 (기본값: 32)
</details>

<details>
<summary><b>2. 벡터 검색 테스트 (선택 사항)</b> - 클릭하여 펼치기</summary>

```bash
python sklearn_vector_search.py --csv dictionary_with_embeddings.csv --query "인공지능"
```

옵션:
- `--csv`: 임베딩이 포함된 CSV 파일 경로
- `--query`: 테스트할 검색어
- `--top-k`: 반환할 검색 결과 수 (기본값: 5)
</details>

<details>
<summary><b>3. API 서버 실행</b> - 클릭하여 펼치기</summary>

```bash
python app.py
```

서버가 시작되면 `http://localhost:8000/docs`에서 API 문서를 확인할 수 있습니다.
</details>

## API 사용법

<details>
<summary><b>단어 검색 API</b> - 클릭하여 펼치기</summary>

**엔드포인트:** `POST /api/query`

**요청 예시:**
```json
{
  "query": "인공지능",
  "top_k": 5,
  "use_rag": true
}
```

**응답 예시:**
```json
{
  "answer": "인공지능은 컴퓨터 과학의 한 분야로, 인간의 학습능력과 추론능력, 지각능력, 자연언어의 이해능력 등을 컴퓨터 프로그램으로 구현한 기술입니다.",
  "sources": [
    {
      "target_code": 42,
      "word": "인공지능",
      "cats": "컴퓨터과학/AI",
      "pos": "명사",
      "definition": "인간의 학습능력과 추론능력, 지각능력, 자연언어의 이해능력 등을 컴퓨터 프로그램으로 구현한 기술",
      "relevance": 0.92
    },
    // 추가 검색 결과...
  ]
}
```
</details>

## 문제 해결

<details>
<summary><b>패키지 설치 오류</b> - 클릭하여 펼치기</summary>

컴파일 오류가 발생하는 경우:
- Conda 환경 사용 (권장)
- 사전 컴파일된 wheel 패키지 사용: `pip install --only-binary=:all: <패키지명>`
</details>

<details>
<summary><b>임베딩 모델 다운로드 오류</b> - 클릭하여 펼치기</summary>

임베딩 모델 다운로드 중 오류가 발생하면:
- 인터넷 연결 확인
- 방화벽/VPN 설정 확인
- 다른 모델 시도: `--model all-MiniLM-L6-v2`
</details>

<details>
<summary><b>CSV 파싱 오류</b> - 클릭하여 펼치기</summary>

CSV 파일 읽기에 문제가 있으면:
```python
df = pd.read_csv("dictionary.csv", encoding='utf-8', engine='python', error_bad_lines=False)
```
</details>

## 호환성 정보

- Python 버전: 3.9 권장 (3.8~3.10 테스트됨)
- 운영체제: Windows, macOS, Linux 지원
- 메모리 요구사항: 최소 4GB RAM (임베딩 모델 로드 시)
