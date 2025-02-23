import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


class NSynthDataset(Dataset):
    def __init__(self, dirs):
        filenames = glob.glob(dirs + '/*.png')



def collate_fn(batch):
    lengths = torch.tensor([elem.shape[-1] for elem in batch])
    return nn.utils.rnn.pad_sequence(batch, batch_first=T), lengths


train_dataset = NSynthDataset(audio_dirs)
train_dataloader = DataLoader(train_dataset, batch_size, collate_fn, num_workers)
sr = train_dataset.sr


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, N, m, s_t, s_f):
        super().__init__()
        self.s_t = s_t
        self.s_f = s_f
        self.layers = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                              out_channels=N,
                                              kernel_size=(3,3),
                                              padding='same'),
                                    nn.ELU(),
                                    nn.Conv2d(in_channels=N,
                                              out_channels=m*N,
                                              kernel_size=(s_f+2,s_t+2),
                                              stride=((s_f,s_t)))
                                    )

        self.skip_connection = nn.Conv2d(in_channels=in_channels,
                                         out_channels=m*N,
                                         kernel_size=(1,1), stride=(s_f, s_t))
        

    def forward(self, x):
        # padding 순서는 conv 만들 때와는 반대임
        return self.layers(F.pad(x, [self.s_t+1, 0, self.s_f, 0])) + self.skip_connection(x)
        

class STFTDiscriminator(nn.Module):
    def __init__(self, C, F_bins):
       super().__init__()
       self.layers = nn.ModuleList([
           nn.Sequential(
               nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(7,7)),
               nn.ELU()
           ),
           nn.Sequential(
               ResidualUnit(),
               nn.ELU()
           ),
           nn.Sequential(
               ResidualUnit(),
               
           )
       ])


for epoch in range(1, N_EPOCHS+1):
    soundstream.train()
    stft_dics.train()
    wave_disc.train()

    