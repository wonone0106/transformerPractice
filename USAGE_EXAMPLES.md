# Usage Examples

이 문서는 한국어-영어 번역 웹 서비스의 사용 예시를 제공합니다.

## 전제 조건

실행하기 전에 다음 파일들이 필요합니다:
- `best_model.pth`: 학습된 모델 가중치 파일
- `spm.model` 또는 `tokenizer/spm.model`: SentencePiece 토크나이저 모델

## 1. 통합 서버 모드 (FastAPI + Gradio)

### 실행

```bash
python run_server.py --mode combined --port 8000
```

### 접속

- **Gradio UI**: http://localhost:8000/gradio
  - 웹 브라우저에서 직접 번역 테스트 가능
  - 한국어 입력 → 번역하기 버튼 클릭 → 영어 결과 확인
  
- **FastAPI 문서**: http://localhost:8000/docs
  - Swagger UI를 통한 API 테스트 가능
  
- **API 엔드포인트**: http://localhost:8000/translate

## 2. FastAPI 모드

### 실행

```bash
python run_server.py --mode fastapi --port 8000
```

### API 호출 예시

#### 번역 요청

```bash
curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "안녕하세요, 만나서 반갑습니다."}'
```

**응답 예시:**
```json
{
  "original": "안녕하세요, 만나서 반갑습니다.",
  "translated": "Hello, nice to meet you."
}
```

#### 최대 길이 지정

```bash
curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "이것은 긴 문장입니다.", "max_length": 50}'
```

#### 헬스체크

```bash
curl http://localhost:8000/health
```

**응답 예시:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 3. Gradio 모드

### 실행

```bash
python run_server.py --mode gradio --port 7860
```

### 접속

- **Gradio UI**: http://localhost:7860

### 사용 방법

1. 왼쪽 텍스트박스에 한국어 문장 입력
2. "번역하기" 버튼 클릭 또는 Enter 키 입력
3. 오른쪽 텍스트박스에서 영어 번역 결과 확인
4. "지우기" 버튼으로 입력/출력 초기화

### 제공되는 예시 문장

- 안녕하세요, 만나서 반갑습니다.
- 오늘 날씨가 정말 좋네요.
- 저는 인공지능을 공부하고 있습니다.
- 한국어를 영어로 번역하는 프로그램입니다.
- 이 모델은 Transformer 구조를 사용합니다.

## Python에서 직접 사용

### 추론 모듈 사용

```python
from inference import get_translator

# 싱글톤 인스턴스 가져오기
translator = get_translator()

# 모델 로드 (최초 1회만 필요)
translator.load_model()

# 번역
korean_text = "안녕하세요, 반갑습니다."
english_text = translator.translate(korean_text)
print(f"원문: {korean_text}")
print(f"번역: {english_text}")
```

### 커스텀 FastAPI 앱에 통합

```python
from fastapi import FastAPI
from inference import get_translator

app = FastAPI()

@app.on_event("startup")
async def startup():
    translator = get_translator()
    translator.load_model()

@app.get("/my-translate")
async def my_translate(text: str):
    translator = get_translator()
    result = translator.translate(text)
    return {"result": result}
```

## 트러블슈팅

### 모델 파일이 없는 경우

```
FileNotFoundError: Model file not found: best_model.pth
```

**해결 방법**: 
1. 먼저 `main.py`를 실행하여 모델을 훈련시키세요
2. 훈련 완료 후 생성된 `best_model_XX.XX.pth` 파일을 `best_model.pth`로 이름 변경

### 토크나이저 파일이 없는 경우

```
FileNotFoundError: Tokenizer file not found: spm.model
```

**해결 방법**:
1. `spm.py`를 실행하여 토크나이저를 훈련시키세요
2. 생성된 `tokenizer/spm.model` 파일 확인
3. 또는 `spm.model` 파일을 프로젝트 루트에 복사

### 포트가 이미 사용 중인 경우

```bash
# 다른 포트 사용
python run_server.py --mode combined --port 8080
```

### GPU 메모리 부족

inference.py의 device 설정은 자동으로 CPU/GPU를 선택합니다:
```python
self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

GPU 메모리가 부족한 경우 CPU 모드로 자동 전환됩니다.

## 프로덕션 배포

### Uvicorn 옵션 설정

```bash
# 워커 프로세스 여러 개 사용
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker 배포 (예시)

```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8000
CMD ["python", "run_server.py", "--mode", "combined", "--port", "8000"]
```

## 추가 정보

- 모든 모드에서 `--host` 옵션으로 바인딩 주소 지정 가능
- 기본 host는 `0.0.0.0` (모든 네트워크 인터페이스)
- 로컬에서만 접속하려면: `--host 127.0.0.1`
