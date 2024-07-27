import os
import numpy as np
import cv2
import torch
import subprocess
import platform
import streamlit as st
import gdown
from openai import OpenAI
from tqdm import tqdm
import face_detection
from models import Wav2Lip
import argparse
import audio
from PIL import Image

# 모델 체크포인트 다운로드 함수

def download_checkpoint():
    checkpoint_path = 'checkpoints/wav2lip.pth'
    if not os.path.exists(checkpoint_path):
        url = 'https://drive.google.com/uc?id=1xhqGmoS2wrEbY1h4SCQcqYra4NpLt7fS'
        gdown.download(url, checkpoint_path, quiet=False)



def text_to_speech(client, text, output_audio_path, input_voice):
    response = client.audio.speech.create(
        model="tts-1",
        voice=input_voice,
        input=text
    )
    response.stream_to_file(output_audio_path)
    print(f"Audio file saved at {output_audio_path}")

def create_tts_files(api_key, txt_n, input_voice):
    client = OpenAI(api_key=api_key)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(current_dir, "audio_files")
    text_dir = os.path.join(current_dir, "text_files")
    
    # audio_files 폴더가 없으면 생성
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    
    # 단일 텍스트 파일에 대해 TTS 수행
    text_file_path = os.path.join(text_dir, txt_n)
    with open(text_file_path, "r", encoding="utf-8") as text_file:
        text = text_file.read().strip()
    
    output_audio_path = os.path.join(audio_dir, f"{os.path.splitext(txt_n)[0]}.wav")
    text_to_speech(client, text, output_audio_path, input_voice)

# Wav2Lip 코드
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
                    help='Name of saved checkpoint to load weights from', default = 'checkpoints/wav2lip.pth')

parser.add_argument('--face', type=str, 
                    help='Filepath of video/image that contains faces to use', default = 'pic_files/pic.png')

parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
                                default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
                    help='Batch size for face detection', default=8)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=48)

parser.add_argument('--resize_factor', default=1, type=int, 
            help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
                    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                    'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    args.static = True

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

    batch_size = args.face_det_batch_size
    
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
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
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results 

def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i%len(frames)
        frame_to_save = [cv2.imread(args.face, cv2.IMREAD_UNCHANGED)]
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    global model
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

def main(face_path):
    global full_frames, mel_chunks, model, detector, predictions, boxes
    args.face=face_path
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    print ("Number of frames available for inference: "+str(len(full_frames)))

    audio_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_files")
    
    result_filenames = []
    
    # audio_files 폴더의 모든 오디오 파일에 대해 처리
    for audio_file_name in os.listdir(audio_dir):
        audio_file_path = os.path.join(audio_dir, audio_file_name)
        if not audio_file_path.endswith('.wav'):
            print(f'Skipping non-wav file: {audio_file_path}')
            continue

        if not audio_file_path.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_file_path, 'temp/temp.wav')
            subprocess.call(command, shell=True)
            audio_file_path = 'temp/temp.wav'

        wav = audio.load_wav(audio_file_path, 16000)
        mel = audio.melspectrogram(wav)
        print(mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80./fps 
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[:len(mel_chunks)]

        batch_size = args.wav2lip_batch_size
        gen = datagen(full_frames.copy(), mel_chunks)

        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                                total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
            if i == 0:
                model = load_model(args.checkpoint_path)
                print ("Model loaded")

                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter('temp/result.avi', 
                                        cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

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

        result_filenames.append(result_filename)

    
    return result_filename

# 폴더 내의 모든 파일 삭제 함수
def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    for subfile in os.listdir(file_path):
                        os.unlink(os.path.join(file_path, subfile))
                    os.rmdir(file_path)
            except Exception as e:
                st.error(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == '__main__':
    st.title("TTS 립싱크 영상 생성기")

    if "process_started" not in st.session_state:
        st.session_state.process_started = False

    if not st.session_state.process_started:
        if st.button("영상 만들기 시작하기"):
            # Streamlit 애플리케이션 시작 시 체크포인트 다운로드
            download_checkpoint()
            # 다운로드 버튼이 눌리면 폴더 내의 모든 파일 삭제
            clear_directory("text_files")
            clear_directory("pic_files")
            clear_directory("results")
            clear_directory("audio_files")
            st.session_state.process_started = True
            st.rerun()
            
    if st.session_state.process_started:
        api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키를 가져옵니다.
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

        # 텍스트 파일 업로드 위젯 추가
        uploaded_file = st.file_uploader("TTS 생성을 위한 텍스트 파일을 업로드 하세요", type="txt")

        if uploaded_file is not None:
            # 업로드된 파일을 text_files 폴더에 저장
            save_path = os.path.join("text_files", uploaded_file.name)
            
            # 디렉토리가 없으면 생성
            if not os.path.exists("text_files"):
                os.makedirs("text_files")

            # 파일 저장
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # 업로드된 텍스트 파일의 내용을 읽고 화면에 표시
            with open(save_path, "r", encoding="utf-8") as f:
                file_contents = f.read()
                st.text_area("업로드된 텍스트 파일 내용", file_contents, height=150)

        # 이미지 파일 업로드 위젯 추가
        uploaded_img_file = st.file_uploader("이미지 파일을 업로드 하세요", type=["jpg", "jpeg", "png"])

        if uploaded_img_file is not None:
            # 업로드된 파일을 pic_files 폴더에 저장
            img_save_path = os.path.join("pic_files", uploaded_img_file.name)
            
            # 디렉토리가 없으면 생성
            if not os.path.exists("pic_files"):
                os.makedirs("pic_files")

            # 파일 저장
            with open(img_save_path, "wb") as f:
                f.write(uploaded_img_file.getvalue())

            # 업로드된 이미지 파일을 열고 화면에 표시
            img = Image.open(img_save_path)
            st.image(img, caption="업로드된 이미지", width=200)

        if uploaded_file is not None and uploaded_img_file is not None:
            voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            selected_voice = st.radio("Select a voice option for TTS", voice_options, index=1)  # Default to "Echo"

            # 선택된 결과를 변수에 저장
            st.session_state.selected_voice = selected_voice

            
            # Streamlit 버튼을 추가하여 TTS 파일 생성 및 Wav2Lip 실행을 트리거
            if st.button("립싱크 영상 생성하기"):
                clear_directory("audio_files")
                clear_directory("results")
                with st.spinner("TTS 파일 생성 중..."):
                    create_tts_files(api_key,uploaded_file.name, st.session_state.selected_voice)  # TTS 파일 생성

                with st.spinner("영상 파일 생성 중..."):
                    result_filename = main(img_save_path)  # Wav2Lip 실행 및 결과 파일 생성

                # 결과 파일에 대해 다운로드 버튼 추가
                if os.path.exists(result_filename):
                    clear_directory("text_files")
                    clear_directory("pic_files")
                    with open(result_filename, "rb") as f:
                        st.success("영상이 성공적으로 생성되었습니다.")
                        download_button = st.download_button(
                            label=f"Download {os.path.basename(result_filename)}",
                            data=f,
                            file_name=os.path.basename(result_filename),
                            mime="video/mp4"
                        )