
# TTS2LipSync

This program combines TTS (Text-to-Speech) and lip-syncing videos using the open-source Wav2Lip project from [Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip). The TTS functionality utilizes OpenAI's API, so you need to use your OpenAI API key.

## Installation and Usage

### Prerequisites

1. **Python**: Ensure you have Python installed on your system.
2. **Pip**: Ensure you have pip installed to manage Python packages.

### Step-by-Step Instructions

1. **Clone the repository**:
    ```sh
    git clone https://github.com/HyeongMinK/TTS2LipSync.git
    cd TTS2LipSync
    ```

2. **Install required libraries**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Download FFmpeg**:
    - For Windows, open PowerShell with administrator rights and run:
    ```sh
    powershell -ExecutionPolicy Bypass -File install_ffmpeg.ps1
    ```

4. **Download the pre-trained Wav2Lip model**:
    - Download the `wav2lip_gan.pth` file from [this link](https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Fwav2lip%5Fgan%2Epth&parent=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels&ga=1) and place it in the `checkpoints` folder.

5. **Prepare your input files**:
    - Place the image you want to use for lip-syncing in the `pic_files` folder.
    - Create text files with the content you want to convert to speech in the `text_files` folder.

6. **Run the inference**:
    ```sh
    python inference_v2.py --checkpoint_path checkpoints\wav2lip_gan.pth --face pic_files\pic.png
    ```

### Output

- The generated audio files from the TTS process will be saved in the `audio_files` folder.
- The resulting lip-synced videos will be saved in the `results` folder.

## Acknowledgements

This project utilizes the following open-source projects:

- [Wav2Lip by Rudrabha](https://github.com/Rudrabha/Wav2Lip)
- [OpenAI API](https://openai.com/)

Please ensure you follow the license agreements and usage policies of these projects.
