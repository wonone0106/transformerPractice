# 실습 방법

사용 데이터 : [한국어-영어 번역(병렬) 말뭉치](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=126)

https://spacy.io/models/ko#ko_core_news_sm

1. 해당 링크로 들어가 tokenizer 및 데이터 설치
2. dataPreprocessing 폴더에 들어가 tokenizer 폴더 및 데이터 폴더 생성
3. 다운 받은 ko_core_news_sm를 tokenizer 폴더로 이동
4. 데이터는 자신의 마음 껏 test/train 양 조절하기

---

### 폴더 구조

```
project┬─dataPreprocessing┬─data┬─test
       │                  │     └─trian
       │                  ├─tokenizer/ko_core_news_sm
       │                  ├─.python-version
       │                  ├─dataPreprocessing.py
       │                  ├─main.py
       │                  ├─pyproject.toml
       │                  ├─README.md
       │                  └─uv.lock
       ├


```

### 코드 실행 방법

cmd에서 아래 명령어 실행

```cmd
uv sync
```
