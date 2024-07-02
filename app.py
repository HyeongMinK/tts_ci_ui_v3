import streamlit as st
import os
import subprocess


# 환경 변수에서 OpenAI API 키 가져오기
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    st.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
else:
    st.success("OPENAI_API_KEY 환경 변수가 설정되었습니다.")

# 사용자 입력 받기
checkpoint_path = st.text_input('Checkpoint Path', 'checkpoints/wav2lip_gan.pth')
face_path = st.text_input('Face Video/Image Path', 'pic_files/pic.png')
user_input = st.text_input('Enter additional input')

if st.button('Run Inference'):
    with st.spinner('Running inference...'):
        command = f'python inference_for_ci.py --checkpoint_path {checkpoint_path} --face {face_path} --user_input "{user_input}"'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            st.success("Inference script executed successfully!")
            st.text(result.stdout)
        else:
            st.error("Inference script failed!")
            st.text(result.stderr)

# Streamlit 애플리케이션 코드
st.title("Wav2Lip Demo")
st.write("Hello, world!")
