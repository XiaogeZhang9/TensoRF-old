import os
import pycolmap
import open3d as o3d
import numpy as np

def qvec2rotmat_xyzw(qvec):
    """
    Convert quaternion vector to rotation matrix.
    Quaternion format: [qx, qy, qz, qw]
    """
    qvec = qvec / np.linalg.norm(qvec)
    x, y, z, w = qvec[0], qvec[1], qvec[2], qvec[3]
    
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    return R

# Load reconstruction from sparse model directory
# path = 'log_3dfront/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932/dense/sparse'
path = 'log_3dfront/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932/dense/sparse'
recon = pycolmap.Reconstruction(path)

c2ws = []
# Access all registered images and their poses
for img_id, image in recon.images.items():
    print(f" Image ID: {img_id}")
    # import ipdb; ipdb.set_trace()
    quat = image.cam_from_world.rotation.quat
    R = qvec2rotmat_xyzw(quat)
    t = image.cam_from_world.translation
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R
    T_w2c[:3, 3] = t
    R_c2w = R.T
    t_c2w = -R.T @ t
    T_c2w = np.eye(4)
    T_c2w[:3, :3] = R_c2w
    T_c2w[:3, 3] = t_c2w
    c2ws.append(T_c2w)

pc_path = '/media/UNIWA\\23903981/Data2/data/3dfront/output_pc/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932_wall_shallow.ply'
point_cloud = o3d.io.read_point_cloud(pc_path)
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(point_cloud)

# Create camera frustum geometries for each pose
for pose in c2ws:
    # Create coordinate frame showing camera pose
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    cam_frame.transform(pose)
    vis.add_geometry(cam_frame)

# Set default camera viewpoint
ctr = vis.get_view_control()
ctr.set_zoom(0.8)

# Render visualization
vis.run()
vis.destroy_window()
    
c2ws = np.array(c2ws)
print(c2ws.shape)

