#! /bin/bash
lrs=(0.1 0.025 0.0005 0.0001)
for lr in ${lrs[@]}; do
    echo "lr_init = $lr"
    sed -i '12c lr_init = '$lr'' configs/your_own_data_plane.txt 
    sbatch scripts_csiro/distill_sigmoid.sh
    sleep 10
done
