'''
    파일명 : run_server.py
    설명 : 통합 서버 실행 스크립트
    작성일 : 2025-12-16
'''
import argparse
import logging
import uvicorn
from threading import Thread

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_fastapi(host="0.0.0.0", port=8000):
    """
    FastAPI 서버를 실행합니다.
    
    Args:
        host: 서버 호스트 주소
        port: 서버 포트
    """
    logger.info(f"Starting FastAPI server on {host}:{port}")
    uvicorn.run("api:app", host=host, port=port, reload=False)


def run_gradio(host="0.0.0.0", port=7860):
    """
    Gradio 서버를 실행합니다.
    
    Args:
        host: 서버 호스트 주소
        port: 서버 포트
    """
    from gradio_app import launch_gradio
    logger.info(f"Starting Gradio server on {host}:{port}")
    launch_gradio(server_name=host, server_port=port)


def run_combined(host="0.0.0.0", port=8000):
    """
    FastAPI와 Gradio를 통합하여 실행합니다.
    
    Args:
        host: 서버 호스트 주소
        port: 서버 포트
    """
    import gradio as gr
    from api import app as fastapi_app
    from gradio_app import create_gradio_interface
    
    logger.info(f"Starting combined server on {host}:{port}")
    
    # Gradio 인터페이스 생성
    gradio_app = create_gradio_interface()
    
    # FastAPI에 Gradio 마운트
    app = gr.mount_gradio_app(fastapi_app, gradio_app, path="/gradio")
    
    logger.info(f"FastAPI available at: http://{host}:{port}")
    logger.info(f"API Docs available at: http://{host}:{port}/docs")
    logger.info(f"Gradio UI available at: http://{host}:{port}/gradio")
    
    # 서버 실행
    uvicorn.run(app, host=host, port=port)


def main():
    """
    메인 함수: 커맨드라인 인자를 파싱하고 서버를 실행합니다.
    """
    parser = argparse.ArgumentParser(
        description="Transformer Translation Web Service"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="combined",
        choices=["fastapi", "gradio", "combined"],
        help="실행 모드: fastapi (FastAPI만), gradio (Gradio만), combined (통합, 기본값)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="서버 호스트 주소 (기본값: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="서버 포트 (기본값: 8000)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting server in {args.mode} mode")
    
    try:
        if args.mode == "fastapi":
            run_fastapi(host=args.host, port=args.port)
        elif args.mode == "gradio":
            run_gradio(host=args.host, port=args.port)
        elif args.mode == "combined":
            run_combined(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        raise


if __name__ == "__main__":
    main()
