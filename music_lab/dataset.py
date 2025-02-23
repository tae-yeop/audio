import torch
from torch.utils.data import Dataset, DataLoader

import argparse
import os
import requests, zipfile, io, subprocess
import librosa
import pandas as pd

class FMADataset(Dataset):
    def __init__(
              self, 
              root_dir, 
              subset="small", 
              download=False,
              sr=22050, 
              transform=None
    ):
            
            self.root_dir = root_dir
            self.subset = subset  # 'small', 'medium', 'large', or 'full'
            self.audio_dir = os.path.join(root_dir, f"fma_{subset}")
            self.metadata_dir = os.path.join(root_dir, "fma_metadata")
            self.sr = sr # target sampling rate for audio
            self.transform = transform
    

            if download:
                audio_zip = os.path.join(root_dir, f"fma_{subset}.zip")
                meta_zip = os.path.join(root_dir, "fma_metadata.zip")
                if not os.path.isdir(self.audio_dir):
                    os.makedirs(root_dir, exist_ok=True)
                    url = f"https://os.unil.cloud.switch.ch/fma/fma_{subset}.zip"
                    print(f"Downloading {url} ...")
                    r = requests.get(url, stream=True)
                    z = zipfile.ZipFile(io.BytesIO(r.content))
                    z.extractall(path=root_dir)
                if not os.path.isdir(self.metadata_dir):
                     url_meta = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
                print(f"Downloading {url_meta} ...")
                r = requests.get(url_meta, stream=True)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(path=root_dir)
                     
            # 메타데이터 CSV에서 트랙 장르 정보 로드
            tracks_file = os.path.join(self.metadata_dir, "tracks.csv")
            if not os.path.isfile(tracks_file):
                raise FileNotFoundError("FMA metadata file not found. Please download fma_metadata.zip.")

            # tracks.csv에는 Multi-index header가 있어 header=[0,1]로 읽음
            tracks = pd.read_csv(tracks_file, index_col=0, header=[0, 1])

            # Small subset에 해당하는 트랙만 필터 (fma_small 디렉토리에 존재하는 트랙)
            audio_files = []
            for root, _, files in os.walk(self.audio_dir):
                for fname in files:
                    if fname.endswith(".mp3"):
                        audio_files.append(os.path.join(root, fname))
            self.audio_files = sorted(audio_files)

            # 장르 레이블 매핑 생성 (genre_top 컬럼 사용)
            genre_series = tracks[("track", "genre_top")]
            # tracks.csv 내의 genre_top이 NaN인 경우도 있으나 small셋은 8개 장르로 채워져 있음
            self.labels = []
            self.genres = sorted(genre_series.dropna().unique().tolist())  # 장르 이름 리스트
            self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}
            for filepath in self.audio_files:
                track_id = int(os.path.splitext(os.path.basename(filepath))[0])
                genre = genre_series.get(track_id)
                if pd.isna(genre):
                    # 장르 정보가 없을 경우 무시하거나 placeholder 처리
                    label = -1
                else:
                    label = self.genre_to_idx[genre]
                self.labels.append(label)

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        filepath = self.audio_files[idx]
        signal, sr = librosa.load(filepath, sr=self.sr, mono=True)
        waveform = torch.from_numpy(signal).float()
        if self.transform:
            waveform = self.transform(waveform)
        label = self.labels[idx]
        return waveform, label
    

class GTZANDataset(Dataset):
    def __init__(
              self, 
              root_dir, 
              download=False,
              sr=22050, 
              transform=None
    ):
        """
        GTZAN Genre Dataset Loader.
        Loads 30s audio clips and assigns genre labels.
        """
        self.root_dir = root_dir
        self.sr = sr
        self.transform = transform
        self.audio_files = []
        self.labels = []
        self.genre_to_idx = {}

        # 데이터 폴더 (장르별 디렉토리가 있음)
        genres_dir = os.path.join(root_dir, "genres")

        if download and not os.path.isdir(genres_dir):
            os.makedirs(root_dir, exist_ok=True)
            try:
                subprocess.run([
                    "kaggle", "datasets", "download", 
                    "andradaolteanu/gtzan-dataset-music-genre-classification", 
                    "-p", root_dir, "--unzip"
                ], check=True)
            except Exception as e:
                # 공식 사이트에서 다운로드 시도
                url = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
                print(f"Downloading GTZAN from {url} ...")
                r = requests.get(url, stream=True)
                open(os.path.join(root_dir, "genres.tar.gz"), 'wb').write(r.content)
                # 압축 해제
                import tarfile
                tar = tarfile.open(os.path.join(root_dir, "genres.tar.gz"))
                tar.extractall(path=root_dir)
                tar.close()

        # 장르 폴더 탐색
        if not os.path.isdir(genres_dir):
            raise FileNotFoundError("GTZAN 'genres' directory not found. Download the dataset first.")
        
        genre_names = sorted(os.listdir(genres_dir))
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(genre_names)}
        for genre in genre_names:
            genre_folder = os.path.join(genres_dir, genre)
            for fname in os.listdir(genre_folder):
                if fname.endswith(".wav") or fname.endswith(".au"):
                    self.audio_files.append(os.path.join(genre_folder, fname))
                    self.labels.append(self.genre_to_idx[genre])
            
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        filepath = self.audio_files[idx]
        # librosa는 .au 포맷도 읽을 수 있음 (ffmpeg 필요할 수 있음)
        signal, sr = librosa.load(filepath, sr=self.sr, mono=True)
        waveform = torch.from_numpy(signal).float()
        if self.transform:
            waveform = self.transform(waveform)
        label = self.labels[idx]
        return waveform, label
    


class Covers80Dataset(Dataset):
    def __init__(self, 
                 root_dir="data/covers80/covers32k", 
                 download=False, 
                 sr=22050, 
                 transform=None):
        """
        Covers80 Dataset Loader.
        Loads audio tracks and assigns a label for each pair of covers (by song title).
        """
        self.root_dir = root_dir
        self.sr = sr
        self.transform = transform
        self.audio_files = []
        self.labels = []
        self.title_to_idx = {}

        # 다운로드 옵션 처리
        if download and not os.path.isdir(root_dir):
            os.makedirs("data", exist_ok=True)
            url = "http://labrosa.ee.columbia.edu/projects/coversongs/covers80/covers80.tgz"
            print(f"Downloading Covers80 from {url} ...")
            r = requests.get(url, stream=True)
            tar_path = os.path.join("data", "covers80.tgz")
            open(tar_path, 'wb').write(r.content)
            import tarfile
            tar = tarfile.open(tar_path)
            tar.extractall(path="data")
            tar.close()
            
        if not os.path.isdir(root_dir):
            raise FileNotFoundError("Covers80 data not found. Please download and extract covers80.tgz.")
        
        # 곡 제목별 폴더 탐색
        song_titles = sorted(os.listdir(root_dir))
        for title in song_titles:
            title_path = os.path.join(root_dir, title)
            if os.path.isdir(title_path):
                # 새로운 곡 제목(cover pair) 발견 시 인덱스 할당
                if title not in self.title_to_idx:
                    self.title_to_idx[title] = len(self.title_to_idx)
                for fname in os.listdir(title_path):
                    # 오디오 파일 (mp3 또는 wav)만 대상
                    if fname.endswith(".wav") or fname.endswith(".mp3"):
                        self.audio_files.append(os.path.join(title_path, fname))
                        self.labels.append(self.title_to_idx[title])

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        filepath = self.audio_files[idx]
        signal, sr = librosa.load(filepath, sr=self.sr, mono=True)
        waveform = torch.from_numpy(signal).float()
        if self.transform:
            waveform = self.transform(waveform)
        label = self.labels[idx]
        return waveform, label
    

import h5py

class DaTacosDataset(Dataset):
    def __init__(self, 
                 root_dir="data", 
                 subset="benchmark", 
                 feature_type="hpcp", 
                 download=False, 
                 transform=None):
        
        self.root_dir = root_dir
        self.subset = subset  # 'benchmark' or 'coveranalysis'
        self.feature_type = feature_type  # e.g., 'hpcp', 'mfcc', 'cens', etc.
        self.transform = transform
        self.files = []    # list of feature file paths
        self.labels = []   # list of work IDs (cover group labels)
        self.label_to_idx = {}  # map WID -> index
