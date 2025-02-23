import json
import os
import re
from pydub import AudioSegment, silence
import speech_recognition as sr

# 시간 변환 함수
def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

# 자막 텍스트 분할 함수 (문장 기준)
def split_sentences(text):
    # 정규 표현식을 사용하여 문장 분리
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_endings.split(text)
    return sentences


# 음성 파일에서 음성 구간 추출 함수
def find_audio_segments(audio_chunk, min_silence_len=500, silence_thresh=-40):
    chunks = silence.split_on_silence(
        audio_chunk,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    return chunks

# JSON 파일 읽기
with open('sub.json', 'r', encoding='utf-8') as f:
    subtitles = json.load(f)

audio_filename = f'/purestorage/project/tyk/13_Audio/extracted_audio.mp3'
audio = AudioSegment.from_mp3(audio_filename)

with open('subtitles.srt', 'w', encoding='utf-8') as f:
    index = 1
    for subtitle in subtitles:
        start_time = subtitle['start']
        end_time = subtitle['end']
        text = subtitle['text']

        # 자막 텍스트를 문장 기준으로 분할
        sentences = split_sentences(text)
        
        # 해당 구간의 오디오 데이터 추출
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        audio_chunk = audio[start_ms:end_ms]

        # 음성 구간 찾기
        audio_segments = find_audio_segments(audio_chunk)
        segment_durations = [len(segment) for segment in audio_segments]
        total_duration = sum(segment_durations)

        # 각 문장에 할당할 시간 계산
        sentence_durations = [(duration / total_duration) * (end_ms - start_ms) for duration in segment_durations]
        
        # 자막 블록 작성
        current_time = start_ms
        for i, sentence in enumerate(sentences):
            if i < len(sentence_durations):
                segment_duration = sentence_durations[i]
            else:
                segment_duration = (end_ms - start_ms) / len(sentences)
            
            current_start_time = current_time / 1000
            current_end_time = (current_time + segment_duration) / 1000
            current_time += segment_duration
            
            f.write(f"{index}\n")
            f.write(f"{format_time(current_start_time)} --> {format_time(current_end_time)}\n")
            f.write(f"{sentence}\n\n")
            index += 1
