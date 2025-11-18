# 실습 방법

https://spacy.io/models/ko#ko_core_news_sm

1. 해당 링크로 들어가 tokenizer 설치
2. tokenizer 폴더 생성 후 다운 받은 ko_core_news_sm를 tokenizer 폴더로 이동

---

### 폴더 구조
```
project─┬─data
        ├─tokenizer/ko_core_news_sm
        ├─.python-version
        ├─dataPreprocessing.py
        ├─main.py
        ├─pyproject.toml
        ├─README.md
        └─uv.lock

```

### 코드 실행 방법

cmd에서 아래 명령어 실행
```cmd
uv sync
```
