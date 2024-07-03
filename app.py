import streamlit as st
import os
import numpy as np
import cv2
import torch
import face_detection
import gdown
from models import Wav2Lip
from openai import OpenAI
import tempfile
import subprocess

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

uploaded_file = st.file_uploader("Choose a video/image file", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        face_path = temp_file.name
    
    st.video(face_path) if face_path.split('.')[-1] in ['mp4', 'avi'] else st.image(face_path)

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

# Wav2Lip 코드
def load_model(checkpoint_path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model.eval()

if uploaded_file is not None:
    # 필요한 인자 설정
    args = {
        'checkpoint_path': 'checkpoints/wav2lip_gan.pth',
        'face': 'pic_files/pic.png',
        'outfile': 'results/result_voice.mp4',
        'static': False,
        'fps': 25.0,
        'pads': [0, 10, 0, 0],
        'face_det_batch_size': 16,
        'wav2lip_batch_size': 128,
        'resize_factor': 1,
        'crop': [0, -1, 0, -1],
        'box': [-1, -1, -1, -1],
        'rotate': False,
        'nosmooth': False,
        'img_size': 96,
    }

    # main 함수
    def main(args):
        if not os.path.isfile(args['face']):
            raise ValueError('--face argument must be a valid path to video/image file')

        if args['face'].split('.')[-1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(args['face'])]
            fps = args['fps']
        else:
            video_stream = cv2.VideoCapture(args['face'])
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            full_frames = []
            while True:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if args['resize_factor'] > 1:
                    frame = cv2.resize(frame, (frame.shape[1] // args['resize_factor'], frame.shape[0] // args['resize_factor']))
                if args['rotate']:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                y1, y2, x1, x2 = args['crop']
                if x2 == -1: x2 = frame.shape[1]
                if y2 == -1: y2 = frame.shape[0]
                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)

        print("Number of frames available for inference: " + str(len(full_frames)))

        audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")

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
            mel_idx_multiplier = 80. / fps
            i = 0
            while True:
                start_idx = int(i * mel_idx_multiplier)
                if start_idx + mel_step_size > len(mel[0]):
                    mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                    break
                mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
                i += 1

            print("Length of mel chunks: {}".format(len(mel_chunks)))
            full_frames = full_frames[:len(mel_chunks)]
            batch_size = args['wav2lip_batch_size']
            gen = datagen(full_frames.copy(), mel_chunks)

            for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
                if i == 0:
                    model = load_model(args['checkpoint_path'])
                    print("Model loaded")
                    frame_h, frame_w = full_frames[0].shape[:-1]
                    out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to('cuda' if torch.cuda.is_available() else 'cpu')
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to('cuda' if torch.cuda.is_available() else 'cpu')

                with torch.no_grad():
                    pred = model(mel_batch, img_batch)

                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

                for p, f, c in zip(pred, frames, coords):
                    y1, y2, x1, x2 = c
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    f[y1:y2, x1:x2] = p
                    out.write(f)

            out.release()

            # 오디오 파일 이름을 기반으로 고유한 결과 파일 이름 생성
            audio_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
            result_filename = f'results/result_voice_{audio_filename}.mp4'
            command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_file_path, 'temp/result.avi', result_filename)
            subprocess.call(command, shell=platform.system() != 'Windows')

    # 환경 변수에서 API 키를 가져옵니다.
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    create_tts_files(api_key)  # TTS 파일 생성 후 Wav2Lip 실행
    main(args)

