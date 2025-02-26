import torch
import argparse
import os
from transformers import HubertModel, Wav2Vec2Model, ClapModel
from torch.utils.data import Dataset, DataLoader
import laion_clap


from datasets import load_dataset
from transformers import AutoProcessor, ClapAudioModel

# dataset = load_dataset("ashraq/esc50")
# audio_sample = dataset["train"]["audio"][0]["array"]

# # model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")
# # processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")

# model = laion_clap.CLAP_Module(enable_fusion=False)

# inputs = processor(audios=audio_sample, return_tensors="pt")

# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state
# pooler_output = outputs.pooler_output


# print(pooler_output.shape, last_hidden_state.shape) # torch.Size([1, 768]) torch.Size([1, 768, 2, 32])

# print(type(audio_sample))
# print(audio_sample)
# print(audio_sample.shape) # (220500,)
# print(inputs)
# print(type(inputs))

import laion_clap
model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
model.load_ckpt('/purestorage/AILAB/AI_1/tyk/3_CUProjects/audio/music_lab/music_audioset_epoch_15_esc_90.14.pt') 

from transformers import Wav2Vec2Processor, Data2VecAudioModel
import torch
from torch import nn
from datasets import load_dataset

# load demo audio and set processor
dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True)
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate
processor = Wav2Vec2Processor.from_pretrained("facebook/data2vec-audio-base-960h")

# loading our model weights
model = Data2VecAudioModel.from_pretrained("m-a-p/music2vec-v1")



# audio file is decoded on the fly
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)


# take a look at the output shape, there are 13 layers of representation
# each layer performs differently in different downstream tasks, you should choose empirically
all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
print(all_layer_hidden_states.shape) # [13 layer, 292 timestep, 768 feature_dim]

# for utterance level classification tasks, you can simply reduce the representation in time
time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
print(time_reduced_hidden_states.shape) # [13, 768]

# you can even use a learnable weighted average representation
aggregator = nn.Conv1d(in_channels=13, out_channels=1, kernel_size=1)
weighted_avg_hidden_states = aggregator(time_reduced_hidden_states).squeeze()
print(weighted_avg_hidden_states.shape) # [768]