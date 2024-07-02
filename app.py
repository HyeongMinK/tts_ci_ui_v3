import streamlit as st
import os
import gdown
import numpy as np
import cv2
import torch
from models import Wav2Lip
from openai import OpenAI
import logging

# 로그 설정
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 모델 체크포인트 다운로드 함수
def download_checkpoint():
    checkpoint_path = 'checkpoints/wav2lip_gan.pth'
    if not os.path.exists(checkpoint_path):
        st.info('Downloading model checkpoint...')
        url = 'https://drive.google.com/uc?id=1PyxYrrjLcKdhdyMMIXlhUYpnoWR9zN-T'
        try:
            gdown.download(url, checkpoint_path, quiet=True)
            st.success('Model checkpoint downloaded.')
        except Exception as e:
            logging.error(f"Error downloading model checkpoint: {e}")
            st.error("Failed to download the model checkpoint. Check logs for more details.")

# Streamlit 애플리케이션 시작 시 체크포인트 다운로드
download_checkpoint()

# Streamlit 애플리케이션 코드
st.title("Wav2Lip Demo")
st.write("Hello, world!")

def text_to_speech(client, text, output_audio_path):
    response = client.audio.speech.create(
        model="tts-1",
        voice="echo",
        input=text
    )
    response.stream_to_file(output_audio_path)
    print(f"Audio file saved at {output_audio_path}")

def create_tts_files(api_key):
    client = OpenAI(api_key=api_key)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(current_dir, "audio_files")
    text_dir = os.path.join(current_dir, "text_files")
    
    # audio_files 폴더가 없으면 생성
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    
    # text_files 폴더의 모든 텍스트 파일에 대해 TTS 수행
    for text_file_name in os.listdir(text_dir):
        text_file_path = os.path.join(text_dir, text_file_name)
        with open(text_file_path, "r", encoding="utf-8") as text_file:
            text = text_file.read().strip()
        
        output_audio_path = os.path.join(audio_dir, f"{os.path.splitext(text_file_name)[0]}.wav")
        text_to_speech(client, text, output_audio_path)

def load_model(checkpoint_path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    return model.eval()

def main():
    face = 'pic_files/pic.png'
    checkpoint_path = 'checkpoints/wav2lip_gan.pth'
    if not os.path.isfile(face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(face)]
        fps = 25

    else:
        video_stream = cv2.VideoCapture(face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
            full_frames.append(frame)

    print ("Number of frames available for inference: "+str(len(full_frames)))

    audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")
    
    model = load_model(checkpoint_path)

    # audio_files 폴더의 모든 오디오 파일에 대해 처리
    for audio_file_name in os.listdir(audio_dir):
        audio_file_path = os.path.join(audio_dir, audio_file_name)
        if not audio_file_path.endswith('.wav'):
            print(f'Skipping non-wav file: {audio_file_path}')
            continue

        wav = audio.load_wav(audio_file_path, 16000)
        mel = audio.melspectrogram(wav)
        print(mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80./fps 
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + 16 > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - 16:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + 16])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[:len(mel_chunks)]

        batch_size = 16
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        for i, m in enumerate(mel_chunks):
            idx = 0 if False else i%len(full_frames)
            frame_to_save = full_frames[idx].copy()
            face = cv2.resize(full_frames[idx], (96, 96))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)

            if len(img_batch) >= batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, 96//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                with torch.no_grad():
                    pred = model(torch.FloatTensor(mel_batch).to('cpu'), torch.FloatTensor(img_batch).to('cpu'))

                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                
                for p, f in zip(pred, frame_batch):
                    f[48:48+96, 48:48+96] = p
                    # 비디오 저장
                    out.write(f)

                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        # 나머지 배치 처리
        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, 96//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            with torch.no_grad():
                pred = model(torch.FloatTensor(mel_batch).to('cpu'), torch.FloatTensor(img_batch).to('cpu'))

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            for p, f in zip(pred, frame_batch):
                f[48:48+96, 48:48+96] = p
                # 비디오 저장
                out.write(f)

if __name__ == '__main__':
    api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키를 가져옵니다.
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    create_tts_files(api_key)  # TTS 파일 생성 후 Wav2Lip 실행
    main()
