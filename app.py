import streamlit as st
import os
import gdown
import numpy as np
import scipy
import cv2
import audio
import subprocess
import torch
import face_detection
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
st.write("Click the button below to run Wav2Lip and generate a video.")

if st.button("Run Wav2Lip"):
    api_key = st.text_input("Enter your OpenAI API Key:")

    if not api_key:
        st.error("Please enter your OpenAI API Key.")
    else:
        def text_to_speech(client, text, output_audio_path):
            response = client.audio.speech.create(
                model="tts-1",
                voice="echo",
                input=text
            )
            response.stream_to_file(output_audio_path)
            st.write(f"Audio file saved at {output_audio_path}")

        def create_tts_files(api_key):
            client = OpenAI(api_key=api_key)
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            audio_dir = os.path.join(current_dir, "audio_files")
            text_dir = os.path.join(current_dir, "text_files")
            
            if not os.path.exists(audio_dir):
                os.makedirs(audio_dir)
            
            for text_file_name in os.listdir(text_dir):
                text_file_path = os.path.join(text_dir, text_file_name)
                with open(text_file_path, "r", encoding="utf-8") as text_file:
                    text = text_file.read().strip()
                
                output_audio_path = os.path.join(audio_dir, f"{os.path.splitext(text_file_name)[0]}.wav")
                text_to_speech(client, text, output_audio_path)

        def get_smoothened_boxes(boxes, T):
            for i in range(len(boxes)):
                if i + T > len(boxes):
                    window = boxes[len(boxes) - T:]
                else:
                    window = boxes[i : i + T]
                boxes[i] = np.mean(window, axis=0)
            return boxes

        def face_detect(images):
            detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                                    flip_input=False, device=device)

            batch_size = 16
            
            while 1:
                predictions = []
                try:
                    for i in range(0, len(images), batch_size):
                        predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
                except RuntimeError:
                    if batch_size == 1: 
                        raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                    batch_size //= 2
                    print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                    continue
                break

            results = []
            pady1, pady2, padx1, padx2 = [0, 10, 0, 0]
            for rect, image in zip(predictions, images):
                if rect is None:
                    cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                    raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

                y1 = max(0, rect[1] - pady1)
                y2 = min(image.shape[0], rect[3] + pady2)
                x1 = max(0, rect[0] - padx1)
                x2 = min(image.shape[1], rect[2] + padx2)
                
                results.append([x1, y1, x2, y2])

            boxes = np.array(results)
            if not False: boxes = get_smoothened_boxes(boxes, T=5)
            results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

            del detector
            return results 

        def datagen(frames, mels):
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

            if args.box[0] == -1:
                if not args.static:
                    face_det_results = face_detect(frames)  # BGR2RGB for CNN face detection
                else:
                    face_det_results = face_detect([frames[0]])
            else:
                print('Using the specified bounding box instead of face detection...')
                y1, y2, x1, x2 = args.box
                face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

            for i, m in enumerate(mels):
                idx = 0 if args.static else i % len(frames)
                frame_to_save = frames[idx].copy()
                face, coords = face_det_results[idx].copy()

                face = cv2.resize(face, (args.img_size, args.img_size))

                img_batch.append(face)
                mel_batch.append(m)
                frame_batch.append(frame_to_save)
                coords_batch.append(coords)

                if len(img_batch) >= args.wav2lip_batch_size:
                    img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                    img_masked = img_batch.copy()
                    img_masked[:, args.img_size // 2:] = 0

                    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                    mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                    yield img_batch, mel_batch, frame_batch, coords_batch
                    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

            if len(img_batch) > 0:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, args.img_size // 2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch

        mel_step_size = 16
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} for inference.'.format(device))

        def _load(checkpoint_path):
            if device == 'cuda':
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path,
                                        map_location=lambda storage, loc: storage)
            return checkpoint

        def load_model(path):
            model = Wav2Lip()
            print("Load checkpoint from: {}".format(path))
            checkpoint = _load(path)
            s = checkpoint["state_dict"]
            new_s = {}
            for k, v in s.items():
                new_s[k.replace('module.', '')] = v
            model.load_state_dict(new_s)

            model = model.to(device)
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

        create_tts_files(api_key)
        main()

        st.success("Processing complete! Download your video below.")

        # 결과 비디오 파일 다운로드 버튼 추가
        video_path = "results/result_voice.mp4"
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.download_button(label="Download Video", data=video_bytes, file_name="result_voice.mp4", mime="video/mp4")
