#! /bin/bash
# root_path=log_3dfront/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932_wall_tex
# nerf_path=log_3dfront/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932_wall_tex

root_path=./synthetic_3dfront_1024_view100_shallow/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932

recon_path=$root_path/recon

mkdir -p $recon_path/0
mkdir -p $recon_path/mvs
database_path=$recon_path/0/database.db
colmap feature_extractor --database_path $database_path --image_path $root_path/rgb

colmap exhaustive_matcher --database_path $database_path

colmap point_triangulator --database_path $database_path --image_path $root_path/rgb --input_path $root_path/sparse --output_path $recon_path/0

colmap image_undistorter --image_path $root_path/rgb --input_path $recon_path/0 --output_path $recon_path/mvs

colmap patch_match_stereo --workspace_path $recon_path/mvs
colmap stereo_fusion --workspace_path $recon_path/mvs --output_path $recon_path/dense.ply