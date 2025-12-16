'''
    파일명 : inference.py
    설명 : Transformer 모델 추론 모듈
    작성일 : 2025-12-16
'''
import torch
import sentencepiece as spm
from transformer import Transformer
import os


class TranslatorSingleton:
    """
    싱글톤 패턴으로 Transformer 모델 인스턴스 관리
    """
    _instance = None
    _model = None
    _sp = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TranslatorSingleton, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path="best_model.pth", tokenizer_path="spm.model"):
        """
        모델과 토크나이저를 로드합니다.
        
        Args:
            model_path: 학습된 모델 가중치 파일 경로
            tokenizer_path: SentencePiece 토크나이저 모델 파일 경로
        """
        if self._model is not None:
            return
        
        # 토크나이저 로드
        self._sp = spm.SentencePieceProcessor()
        
        # 토크나이저 경로 확인
        if not os.path.exists(tokenizer_path):
            tokenizer_path = os.path.join("tokenizer", "spm.model")
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        
        self._sp.load(tokenizer_path)
        
        # 디바이스 설정
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 초기화
        self._model = Transformer(
            src_dim=37000,
            tgt_dim=37000,
            embed_dim=512,
            n_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        ).to(self._device)
        
        # 모델 가중치 로드
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self._device)
        self._model.load_state_dict(checkpoint)
        self._model.eval()
    
    def translate(self, text, max_length=100):
        """
        한국어 텍스트를 영어로 번역합니다.
        
        Args:
            text: 번역할 한국어 텍스트
            max_length: 생성할 최대 토큰 길이
            
        Returns:
            번역된 영어 텍스트
        """
        if self._model is None or self._sp is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # 입력 텍스트 인코딩
        src_tokens = self._sp.encode(text)
        src_tensor = torch.tensor([src_tokens]).to(self._device)
        
        # Greedy decoding
        bos_id = self._sp.bos_id()
        eos_id = self._sp.eos_id()
        
        tgt_tokens = [bos_id]
        
        with torch.no_grad():
            for _ in range(max_length):
                tgt_tensor = torch.tensor([tgt_tokens]).to(self._device)
                
                # 모델 예측
                output = self._model(src_tensor, tgt_tensor)
                
                # 다음 토큰 예측 (greedy)
                next_token = output[0, -1, :].argmax().item()
                
                # EOS 토큰이면 종료
                if next_token == eos_id:
                    break
                
                tgt_tokens.append(next_token)
        
        # 디코딩 (BOS 토큰 제외)
        translated_text = self._sp.decode(tgt_tokens[1:])
        
        return translated_text
    
    @property
    def is_loaded(self):
        """모델이 로드되었는지 확인"""
        return self._model is not None


# 싱글톤 인스턴스 생성
translator = TranslatorSingleton()


def get_translator():
    """
    Translator 싱글톤 인스턴스를 반환합니다.
    
    Returns:
        TranslatorSingleton 인스턴스
    """
    return translator
