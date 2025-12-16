# Transformer Practice

PyTorch로 구현한 Transformer 모델을 활용한 한국어-영어 번역 프로젝트입니다.

## 📚 데이터

사용 데이터 : [한국어-영어 번역(병렬) 말뭉치](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=126)

## 🚀 설치 및 훈련

### 1. 의존성 설치

```bash
uv sync
```

또는

```bash
pip install -e .
```

### 2. 데이터 준비

1. 위 링크에서 데이터 다운로드
2. `data/train/` 및 `data/valid/` 폴더에 Excel 파일 배치
3. SentencePiece 토크나이저 훈련:
   ```bash
   python spm.py
   ```

### 3. 모델 훈련

```bash
python main.py
```

훈련이 완료되면 `best_model.pth` 파일이 생성됩니다.

---

## 🌐 웹 서비스

Transformer 모델을 활용한 한국어-영어 번역 웹 서비스를 제공합니다.

### 필요 파일
- `best_model.pth`: 학습된 모델 가중치
- `spm.model` 또는 `tokenizer/spm.model`: SentencePiece 토크나이저 모델

### 의존성 설치

```bash
pip install gradio fastapi uvicorn[standard] pydantic
```

### 실행 방법

#### 1. 통합 서버 실행 (FastAPI + Gradio)

```bash
python run_server.py --mode combined --port 8000
```

접속 주소:
- **Gradio UI**: http://localhost:8000/gradio
- **API 엔드포인트**: http://localhost:8000/translate
- **API 문서**: http://localhost:8000/docs

#### 2. FastAPI만 실행

```bash
python run_server.py --mode fastapi --port 8000
```

#### 3. Gradio만 실행

```bash
python run_server.py --mode gradio --port 7860
```

### API 사용 예시

```bash
curl -X POST "http://localhost:8000/translate" \
     -H "Content-Type: application/json" \
     -d '{"text": "안녕하세요, 만나서 반갑습니다."}'
```

응답:
```json
{
  "original": "안녕하세요, 만나서 반갑습니다.",
  "translated": "Hello, nice to meet you."
}
```

### 헬스체크

```bash
curl http://localhost:8000/health
```

응답:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## 📁 프로젝트 구조

```
transformerPractice/
├── transformer.py          # Transformer 모델 구현
├── dataPreprocessing.py    # 데이터 전처리
├── trainer.py              # 훈련 로직
├── main.py                 # 훈련 실행 스크립트
├── inference.py            # 추론 모듈 (싱글톤 패턴)
├── api.py                  # FastAPI 백엔드
├── gradio_app.py           # Gradio 프론트엔드
├── run_server.py           # 통합 서버 실행 스크립트
├── spm.py                  # SentencePiece 훈련
├── pyproject.toml          # 프로젝트 설정
├── config/
│   └── train.yaml          # 훈련 설정
├── data/
│   ├── train/              # 훈련 데이터 (Excel 파일)
│   └── valid/              # 검증 데이터 (Excel 파일)
└── tokenizer/
    └── spm.model           # SentencePiece 토크나이저
```

## 🛠️ 기술 스택

- **Deep Learning**: PyTorch
- **NLP**: SentencePiece
- **Web Framework**: FastAPI, Gradio
- **Configuration**: Hydra
- **Data Processing**: Pandas, openpyxl
