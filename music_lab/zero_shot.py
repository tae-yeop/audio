import torch
import argparse
import os

from torch.utils.data import Dataset, DataLoader


def get_model():
    """
    Retrieves and initializes a specified audio processing model.

    Parameters:
    None

    Returns:
    model: torch.nn.Module
        The initialized model ready for use on the specified device.
    """
def get_model():
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["clap", "music2vec"], default="clap")
    parser.add_argument("--music_file_path", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, num_genres=args.num_genres).to(device)
    model.eval()

    with torch.no_grad():
