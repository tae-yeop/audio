### 써볼 수 있는 모델
- 압축 모델
    - EnCodec (2022)
    - DacModel (2023)

- 표현 모델
    - ASTModel
        - spectrogram을 아웃풋으로 낸다
        - spectrogram으로 부터 장르를 구분

    - Hubert
        - 음악에 대해 SSL을 수행하고 나서 쓰면 될듯?

    - [CLAP](https://github.com/LAION-AI/Clap)
        - transformers에도 있고 자체적으로 music에 대해 사전학습한 모델을 주는듯

    - Data2Vec
        - 원래는 스피치용인데 음악에 대해 학습해놓고 weight 올려준게 있음
        - [music2vec-v1](https://huggingface.co/m-a-p/music2vec-v1)


- 생성 모델
    - [MusicGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)
        - transformers MusicgenForConditionalGeneration으로 제공해줌
        - 파인튜닝하는 코드도 있음

    - [MusicLDM]
        - diffusers에서 지원해줌


### 구성
- train.py
    - classifier를 학습시켜서 쓰기

- zero_shot.py
    - 바로 백본에서 임베딩 뽑아서 해보기



### 목표
- 음악을 듣고 비슷한 장르 + 느낌 + 이미지까지 임베딩으로
- 넣어서 음악 생성
- 가사도 생성
- 음악 구조를 따와서 생성
- VAE같은걸 쓰면?
- 서비스를 할거면 틀려도 상관없는거



 