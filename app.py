import streamlit as st
import os
import gdown


# 모델 체크포인트 다운로드 함수
def download_checkpoint():
    checkpoint_path = 'checkpoints/wav2lip_gan.pth'
    if not os.path.exists(checkpoint_path):
        st.info('Downloading model checkpoint...')
        url = 'https://drive.google.com/uc?id=1PyxYrrjLcKdhdyMMIXlhUYpnoWR9zN-T'
        gdown.download(url, checkpoint_path, quiet=False)
        st.success('Model checkpoint downloaded.')


# Streamlit 애플리케이션 시작 시 체크포인트 다운로드
download_checkpoint()

# Streamlit 애플리케이션 코드
st.title("Wav2Lip Demo")
st.write("Hello, world!")

# 예시로 inference_for_ci.py 스크립트를 실행하는 방법
result = os.system('python inference_for_ci.py --checkpoint_path checkpoints/wav2lip_gan.pth --face pic_files/pic.png')
if result == 0:
    st.success("Inference script executed successfully!")
else:
    st.error("Inference script failed!")
