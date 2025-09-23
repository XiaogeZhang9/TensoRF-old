#! /bin/bash
# root_path=log_3dfront/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932_wall_tex
# nerf_path=log_3dfront/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932_wall_tex
textures=$(ls /home/23903981/Documents/3DFRONT/wall_texture) 
for texture in $textures
do
    echo $texture 
    root_path=./synthetic_3dfront_1024_view100_wall_tex/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932_$texture
    output_dir=$root_path/sparse
    mkdir -p $output_dir
    img_dir=$root_path/rgb
    python3 utils/gen_sparse_colmap.py --root_dir $root_path --out_dir $output_dir --img_dir $img_dir
done
