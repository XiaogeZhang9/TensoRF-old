#! /bin/bash
lrs=(0.1 0.025 0.0005)
for lr in ${lrs[@]}; do
    echo "lr_init = $lr"
    cp configs/your_own_data_plane.txt configs/your_own_data_plane_lr$lr.txt
    sed -i "4c expname =  tensorf_VM_wpc_1024_view100_house_lr$lr" configs/your_own_data_plane_lr$lr.txt 
    sed -i '12c lr_init = '$lr'' configs/your_own_data_plane_lr$lr.txt 
    sed -i '8c config=configs/your_own_data_plane_lr'$lr'.txt' scripts_sbatch/distill_sigmoid.sh
    sbatch scripts_sbatch/distill_sigmoid.sh
    sleep 5
done
