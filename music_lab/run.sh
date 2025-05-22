#!/bin/bash -l

#SBATCH --time=99:00:00
#SBATCH -p 40g
#SBATCH --nodes=1            # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=1  # This needs to match Trainer(devices=...)
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=32
#SBATCH -o ./logs/%A.txt

srun --container-image /purestorage/AILAB/AI_1/tyk/0_Software/audio.sqsh \
    --container-mounts /purestorage:/purestorage,/purestorage/AILAB/AI_1/tyk/0_Software/cache:/home/$USER/.cache \
    --no-container-mount-home --unbuffered \
    --container-writable \
    --container-workdir /purestorage/AILAB/AI_1/tyk/3_CUProjects/audio/music_lab \
    bash -c "
    python zero_shot.py $@
    "

# python dataset.py