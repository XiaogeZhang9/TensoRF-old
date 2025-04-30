#! /bin/bash
get_last_checkpoint() {
    expname=$1
    checkpoint=$(ls -t log_triplane/${expname}/triplanes_*.pt | head -n1)
    epoch=$(echo $checkpoint | grep -o 'epoch-[0-9]*' | cut -d'-' -f2)
    echo $checkpoint
    # fi
}
ckpt_tri_density=log_triplane/tensorf_VM_wpc_1024_view100_house/triplanes_599.pt
lrs=(0.125 0.025 0.005 0.001 0.0002 0.00004 0.000008)
for lr in ${lrs[@]}; do
    echo "lr_init = $lr"
    config=configs/your_own_data_plane_lr${lr}_total.txt
    cp configs/your_own_data_plane.txt $config
    sed -i "4c expname =  tensorf_VM_wpc_1024_view100_house_lr${lr}_total" $config 
    sed -i '12c lr_init = '$lr'' $config 
    sed -i '8c config='$config'' scripts_sbatch/distill_sigmoid.sh
    
    sed -i "2c #SBATCH --time=80:00:00" scripts_sbatch/distill_sigmoid.sh
    sed -i "9c CUDA_VISIBLE_DEVICES=0 python3 train_planes.py --config \$config --ckpt \
    log/tensorf_VM_wpc_1024_view100_house/tensorf_VM_wpc_1024_view100_house.th \ 
    --ckpt_tri_density $ckpt_tri_density" scripts_sbatch/distill_sigmoid.sh
    sbatch scripts_sbatch/distill_sigmoid.sh
    sleep 5

    # sed -i "2c #SBATCH --time=01:00:00" scripts_sbatch/distill_sigmoid.sh
    # last_ckpt=$(get_last_checkpoint tensorf_VM_wpc_1024_view100_house_lr$lr)
    # sed -i "9c CUDA_VISIBLE_DEVICES=0 python3 train_planes.py --config \$config --ckpt \
    # log/tensorf_VM_wpc_1024_view100_house/tensorf_VM_wpc_1024_view100_house.th --ckpt_tri \
    # $last_ckpt --render_only 1 --render_test 1" scripts_sbatch/distill_sigmoid.sh
    # sbatch scripts_sbatch/distill_sigmoid.sh
    sleep 5
done
