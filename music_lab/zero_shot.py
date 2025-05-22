import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import HubertModel, Wav2Vec2Model, ClapModel, Wav2Vec2Processor, Data2VecAudioModel
from datasets import load_dataset
import laion_clap

import librosa            
import argparse
import os


class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.model == "wav2vec2":
            self.backbone = HubertModel.from_pretrained("facebook/hubert-base-ls960")

        elif args.model == "music2vec":
            # https://huggingface.co/m-a-p/music2vec-v1
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")
            self.backbone = Data2VecAudioModel.from_pretrained("m-a-p/music2vec-v1")

        elif args.model == "clap": 
            # https://github.com/LAION-AI/CLAP
            # https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt
            self.backbone = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
            self.backbone.load_ckpt('music_audioset_epoch_15_esc_90.14.pt')
        else:
            raise ValueError("지원되지 않는 backbone 모델입니다.")

        for param in self.backbone.parameters():
            param.requires_grad = False

    def process_wav2vec2(self, audio_data):
        audio_data = self.processor(audio_data, return_tensors="pt")
        audio_data = audio_data.input_values
        return audio_data

    def process_clap(self, audio_data):
        audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
        print(audio_embed[:,-20:])
        print(audio_embed.shape)
        return audio_embed

    def process_music2vec(self, audio_data):
        inputs = self.processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

        print('inputs', inputs) # input_values', 'attention_mask'
        print('inputs type', type(inputs))
        with torch.no_grad():
            outputs = self.backbone(**inputs, output_hidden_states=True)

        all_layer_hidden_states = torch.stack(outputs.hidden_states)
        print(all_layer_hidden_states.shape)

        return all_layer_hidden_states.mean()

    @torch.no_grad()
    def forward(self, audio_data):
        if self.args.model == "wav2vec2":
            audio_emb = self.process_wav2vec2(audio_data)
        elif self.args.model == "music2vec":
            audio_emb = self.process_music2vec(audio_data)
        elif self.args.model == "clap":
            audio_emb = self.process_clap(audio_data)

        return audio_emb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["clap", "music2vec"], default="music2vec")
    parser.add_argument("--music_file_path", type=str, default="/home/data/test_clap_short.wav")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureExtractor(args).to(device)
    model.eval()

    # /purestorage/AILAB/AI_1/tyk/3_CUProjects/audio/music_lab/data/covers80/coversongs/covers32k/Abracadabra/steve_miller_band+Steve_Miller_Band_Live_+09-Abracadabra.mp3
    # /purestorage/AILAB/AI_1/tyk/3_CUProjects/audio/music_lab/data/covers80/coversongs/covers32k/Abracadabra/sugar_ray+14_59+11-Abracadabra.mp3
    # audio_data, _ = librosa.load(args.music_file_path, sr=48000)
    # audio_data = audio_data.reshape(1, -1)
    # print(audio_data.shape)

    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate

    print('sampling_rate', sampling_rate) # 16000

    audio_data = dataset[0]["audio"]["array"]
    print('audio_data', audio_data.shape) # (93680,)
    print('audio_data type', type(audio_data)) # 'numpy.ndarray'>
    audio_data = audio_data.reshape(1, -1)
    print('audio_data', audio_data.shape) # (1, 93680)
    

    model.forward(audio_data)


    # compare test
    audio_data_1, _ = librosa.load(
        "/purestorage/AILAB/AI_1/tyk/3_CUProjects/audio/music_lab/data/covers80/coversongs/covers32k/Abracadabra/steve_miller_band+Steve_Miller_Band_Live_+09-Abracadabra.mp3",
        sr=48000
    )
    audio_data_1 = audio_data_1.reshape(1, -1)


    audio_data_2, _ = librosa.load(
        "/purestorage/AILAB/AI_1/tyk/3_CUProjects/audio/music_lab/data/covers80/coversongs/covers32k/Abracadabra/sugar_ray+14_59+11-Abracadabra.mp3",
        sr=48000
    )
    audio_data_2 = audio_data_2.reshape(1, -1)

    first_emb = model.forward(audio_data_1)
    second_emb = model.forward(audio_data_2)

    print('first_emb', first_emb.shape)
    print(second_emb.shape)


    cosine_sim = F.cosine_similarity(first_emb.unsqueeze(0), second_emb.unsqueeze(0))
    print(cosine_sim)