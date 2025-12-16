'''
    파일명 : api.py
    설명 : FastAPI 백엔드 서버
    작성일 : 2025-12-16
'''
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from inference import get_translator
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="Korean-English Translation API",
    description="Transformer 모델을 이용한 한국어-영어 번역 API",
    version="1.0.0"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic 모델 정의
class TranslationRequest(BaseModel):
    text: str = Field(..., description="번역할 한국어 텍스트", min_length=1)
    max_length: int = Field(100, description="생성할 최대 토큰 길이", ge=1, le=500)


class TranslationResponse(BaseModel):
    original: str = Field(..., description="원문 텍스트")
    translated: str = Field(..., description="번역된 텍스트")


class HealthResponse(BaseModel):
    status: str = Field(..., description="서버 상태")
    model_loaded: bool = Field(..., description="모델 로드 상태")


@app.on_event("startup")
async def startup_event():
    """
    서버 시작 시 모델 자동 로드
    """
    try:
        translator = get_translator()
        translator.load_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    헬스체크 엔드포인트
    
    Returns:
        서버 및 모델 상태 정보
    """
    translator = get_translator()
    return HealthResponse(
        status="healthy",
        model_loaded=translator.is_loaded
    )


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    한국어를 영어로 번역하는 엔드포인트
    
    Args:
        request: 번역 요청 (한국어 텍스트 포함)
        
    Returns:
        번역 결과
    """
    try:
        translator = get_translator()
        
        if not translator.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded. Please try again later."
            )
        
        translated_text = translator.translate(
            request.text,
            max_length=request.max_length
        )
        
        return TranslationResponse(
            original=request.text,
            translated=translated_text
        )
    
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
