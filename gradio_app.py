'''
    파일명 : gradio_app.py
    설명 : Gradio 프론트엔드 인터페이스
    작성일 : 2025-12-16
'''
import gradio as gr
from inference import get_translator
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def translate_text(korean_text):
    """
    한국어 텍스트를 영어로 번역하는 함수
    
    Args:
        korean_text: 번역할 한국어 텍스트
        
    Returns:
        번역된 영어 텍스트
    """
    if not korean_text or korean_text.strip() == "":
        return "번역할 텍스트를 입력해주세요."
    
    try:
        translator = get_translator()
        
        if not translator.is_loaded:
            translator.load_model()
        
        translated = translator.translate(korean_text)
        return translated
    
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return f"번역 중 오류가 발생했습니다: {str(e)}"


def clear_text():
    """
    입력과 출력을 초기화하는 함수
    
    Returns:
        빈 문자열 튜플
    """
    return "", ""


def create_gradio_interface():
    """
    Gradio 인터페이스를 생성하는 함수
    
    Returns:
        Gradio Blocks 인스턴스
    """
    # 예시 문장
    examples = [
        ["안녕하세요, 만나서 반갑습니다."],
        ["오늘 날씨가 정말 좋네요."],
        ["저는 인공지능을 공부하고 있습니다."],
        ["한국어를 영어로 번역하는 프로그램입니다."],
        ["이 모델은 Transformer 구조를 사용합니다."]
    ]
    
    with gr.Blocks(title="Korean-English Translation") as demo:
        gr.Markdown(
            """
            # 🌐 한국어-영어 번역기
            Transformer 모델을 이용한 한국어-영어 번역 서비스입니다.
            """
        )
        
        with gr.Row():
            with gr.Column():
                korean_input = gr.Textbox(
                    label="한국어 입력",
                    placeholder="번역할 한국어 텍스트를 입력하세요...",
                    lines=5
                )
                
                with gr.Row():
                    translate_btn = gr.Button("번역하기", variant="primary")
                    clear_btn = gr.Button("지우기")
            
            with gr.Column():
                english_output = gr.Textbox(
                    label="영어 번역 결과",
                    lines=5,
                    interactive=False
                )
        
        gr.Examples(
            examples=examples,
            inputs=korean_input,
            label="예시 문장"
        )
        
        # 버튼 이벤트 연결
        translate_btn.click(
            fn=translate_text,
            inputs=korean_input,
            outputs=english_output
        )
        
        clear_btn.click(
            fn=clear_text,
            outputs=[korean_input, english_output]
        )
        
        # Enter 키로도 번역 가능하도록 설정
        korean_input.submit(
            fn=translate_text,
            inputs=korean_input,
            outputs=english_output
        )
    
    return demo


def launch_gradio(server_name="0.0.0.0", server_port=7860, root_path=None):
    """
    Gradio 앱을 실행하는 함수
    
    Args:
        server_name: 서버 호스트 주소
        server_port: 서버 포트
        root_path: FastAPI와 통합 시 사용할 root path
    """
    demo = create_gradio_interface()
    
    # 모델 로드
    try:
        translator = get_translator()
        if not translator.is_loaded:
            translator.load_model()
            logger.info("Model loaded successfully for Gradio")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}")
    
    # launch 파라미터 설정
    launch_kwargs = {
        "server_name": server_name,
        "server_port": server_port,
        "share": False
    }
    
    if root_path:
        launch_kwargs["root_path"] = root_path
    
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    launch_gradio()
