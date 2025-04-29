#! /bin/bash
#SBATCH --time=80:00:00
#SBATCH --mem=96GB
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --account=OD-228963

config=configs/your_own_data_plane.txt
CUDA_VISIBLE_DEVICES=0 python3 train_planes.py --config $config --ckpt log/tensorf_VM_wpc_1024_view100_house/tensorf_VM_wpc_1024_view100_house.th