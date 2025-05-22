import whisper

model = whisper.load_model("large-v3", download_root="/purestorage/AILAB/AI_1/tyk/0_Software/cache", device='cuda')
result = model.transcribe("/purestorage/AILAB/AI_1/tyk/3_CUProjects/audio/speech_lab/whisper/extracted_audio.mp3")
print(result["text"])

output_text_transcribe = result["text"].strip()
filename = './test.txt'

with open(filename, 'w', encoding='utf-8') as file:
    file.write(output_text_transcribe)