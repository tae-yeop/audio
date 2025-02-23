import whisper
from moviepy.editor import VideoFileClip
import os
import logging

logging.basicConfig(level=logging.INFO)

model = whisper.load_model("large-v3", download_root="/purestorage/project/tyk/13_Audio", device='cuda')

def get_text(file_path):
    if file_path != '':
        output_text_transcribe = ''
    
    # Extract audio from video file
    video = VideoFileClip(file_path)
    audio_path = "extracted_audio.mp3"
    video.audio.write_audiofile(audio_path)

    file_stats = os.stat(audio_path)
    logging.info(f'Size of audio file in Bytes: {file_stats.st_size}')

    result = model.transcribe(audio_path)
    return result['text'].strip()

    # if file_stats.st_size <= 30000000:  # 30 MB
    #     result = model.transcribe(audio_path)
    #     return result['text'].strip()
    # else:
    #     logging.error('The audio file is too large to transcribe. Please use a smaller file.')


output_text_transcribe = get_text('/purestorage/project/tyk/13_Audio/test.ts')
# print(type(output_text_transcribe), output_text_transcribe)

filename = './test.txt'
with open(filename, 'w', encoding='utf-8') as file:
    file.write(output_text_transcribe)