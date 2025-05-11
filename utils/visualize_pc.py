import open3d as o3d
import os
import numpy as np
pc_folders = '/media/UNIWA\\23903981/Data2/data/WPC_database/original_PCs'
pc_folder = 'ship.ply'
# for pc_folder in os.listdir(pc_folders):
pc_path = os.path.join(pc_folders, pc_folder)
pc = o3d.io.read_point_cloud(pc_path)
print(np.array(pc.points).shape)
# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name=pc_folder)
# vis.add_geometry(pc)
# vis.run()
# vis.destroy_window()
