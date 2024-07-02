import streamlit as st
import os
import subprocess
import gdown

# 시스템 패키지 설치 (ffmpeg 등)
def install_ffmpeg():
    if not os.path.isfile('/usr/bin/ffmpeg'):
        st.info('Installing ffmpeg...')
        os.system('chmod +x setup.sh')
        os.system('./setup.sh')
        st.success('ffmpeg installed.')

# 모델 체크포인트 다운로드 함수
def download_checkpoint():
    checkpoint_path = 'checkpoints/wav2lip_gan.pth'
    if not os.path.exists(checkpoint_path):
        st.info('Downloading model checkpoint...')
        url = 'https://drive.google.com/uc?id=1PyxYrrjLcKdhdyMMIXlhUYpnoWR9zN-T'
        gdown.download(url, checkpoint_path, quiet=False)
        st.success('Model checkpoint downloaded.')

# ffmpeg 설치
install_ffmpeg()

# Streamlit 애플리케이션 시작 시 체크포인트 다운로드
download_checkpoint()

# 환경 변수에서 OpenAI API 키 가져오기
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    st.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
else:
    st.success("OPENAI_API_KEY 환경 변수가 설정되었습니다.")

# 예시로 inference_for_ci.py 스크립트를 실행하는 방법
if st.button('Run Inference'):
    with st.spinner('Running inference...'):
        env = os.environ.copy()
        env['OPENAI_API_KEY'] = openai_api_key
        command = f'python inference_for_ci.py --checkpoint_path checkpoints/wav2lip_gan.pth --face pic_files/pic.png'
        result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env)

        if result.returncode == 0:
            st.success("Inference script executed successfully!")
            st.text(result.stdout)
        else:
            st.error("Inference script failed!")
            st.text(result.stderr)

# Streamlit 애플리케이션 코드
st.title("Wav2Lip Demo")
st.write("Hello, world!")
