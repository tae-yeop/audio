import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoFeatrueExtractor, HubertModel
from torch.cuda.amp import GradScaler, autocast

import argparse
import os

from dataclasses import dataclass
@dataclass
class TrainingConfig:
    num_epochs:int = 5
    model_name:str = "music2vec"


from torch.utils.data import Dataset, DataLoader
def load_dataset(dataset_name):
    if dataset_name == "fma":
        dataset = FMADataset()
    elif dataset_name == "gtzan":
        dataset = GTZANDataset()
    elif dataset_name == "da-tacos":
        dataset = DaTacosDataset()
    elif dataset_name == "covers80":
        dataset = Covers80Dataset()
    else:
        raise ValueError("지원되지 않는 데이터셋입니다.")

    train_loader = DatatLoader()


    return train_loader, val_loader


from transformers import AutoFeatureExtractor, HubertModel, Wav2Vec2Model, WavLMModel
class GenrePlagiarismModel(nn.Module):
    def __init__(self, backbone, embed_dim, num_genres):
        super().__init__()
        self.backbone = backbone
        self.genre_classifier = nn.Linear(embed_dim, num_genres)
        self.similarity_fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def encode(self, input_values, attention_mask=None):
        """
        백본에서 임베딩을 얻기
        """
        outputs = self.backbone(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        return hidden_states.mean(dim=1)
    
    def forward(self, input_values_a, input_values_b=None, attn_mask_a=None, attn_mask_b=None):
        if input_values_b is None:
            emb = self.encode(input_values_a, attention_mask=attn_mask_a)
            return self.genre_classifier(emb)
        else:
            emb_a = self.encode(input_values_a, attention_mask=attn_mask_a)
            emb_b = self.encode(input_values_b, attention_mask=attn_mask_b)
            combined = torch.cat([emb_a, emb_b], dim=1)
            return self.similarity_fc(combined)
        
def get_model(model_name, num_genres):
    if model_name == "hubert":
        backbone = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    elif model_name == "wav2vec2":
        backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    else:
        raise ValueError("지원되지 않는 backbone 모델입니다.")

    for param in backbone.parameters():
        param.requires_grad = False

    return GenrePlagiarismModel(backbone, embed_dim=backbone.config.hidden_size, num_genres=num_genres)

if __name__ == '__main__':


    cfg = TrainingConfig()

    model = get_model(cfg.model_name, cfg.num_genres)
    train_loader, val_loader = load_dataset(cfg.dataset, cfg.batch_size)
    criterion_genre = nn.CrossEntropyLoss()
    criterion_plag = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scaler = GradScaler()


    for epoch in range(cfg.num_epochs):
        model.train()
