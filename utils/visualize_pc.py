import open3d as o3d
import os

pc_folders = '/media/UNIWA\\23903981/Data2/data/WPC_database/original_PCs'

for pc_folder in os.listdir(pc_folders):
    pc_path = os.path.join(pc_folders, pc_folder)
    pc = o3d.io.read_point_cloud(pc_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=pc_folder)
    vis.add_geometry(pc)
    vis.run()
    vis.destroy_window()
