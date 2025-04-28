import numpy as np
import open3d as o3d
import glob
import os
import json
import cv2

def define_camera_intrinsics():
    # Define camera intrinsics (example values)
    image_width, image_height = 1024, 1024
    fx, fy = 1024, 1024  # Focal length
    cx, cy = image_width/2, image_height/2  # Principal point

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K, image_width, image_height

def generate_bouding_box(point_file, output_dir):
    # Calculate bounding box and initial voxel size
    point_cloud = o3d.io.read_point_cloud(point_file)
    points = np.asarray(point_cloud.points)
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    # Calculate initial voxel size as 1% of the largest dimension
    bbox_dimensions = max_bound - min_bound
    initial_voxel_size = 1 #np.max(bbox_dimensions) * 0.01
    
    # Write bbox.txt with format: x_min y_min z_min x_max y_max z_max initial_voxel_size
    bbox_data = np.concatenate([min_bound, max_bound, [initial_voxel_size]])
    np.savetxt(f'{output_dir}/bbox.txt', bbox_data.reshape(1, -1), fmt='%.6f')
    
    return min_bound, max_bound

def generate_spherical_poses(point_cloud, n_poses=100, radius_factor=3.0):
    """
    Generate camera poses on a sphere surrounding the point cloud.
    Uses golden spiral distribution for uniform sphere sampling.
    
    Args:
        point_cloud: Open3D point cloud object
        n_poses: Number of camera poses to generate
        radius_factor: Factor to multiply the point cloud's bounding sphere radius
    """
    # Calculate point cloud center and radius
    points = np.asarray(point_cloud.points)
    center = points.mean(axis=0)
    
    # Calculate bounding sphere radius
    distances = np.linalg.norm(points - center, axis=1)
    radius = np.max(distances) * radius_factor
    
    poses = []
    golden_angle = np.pi * (3 - np.sqrt(5))  # Golden angle in radians

    for i in range(n_poses):
        # Calculate spherical coordinates using golden spiral
        y = 1 - (i / float(n_poses - 1)) * 2  # y goes from 1 to -1
        theta = golden_angle * i  # Azimuthal angle
        
        # Convert to Cartesian coordinates
        x = np.sqrt(1 - y*y) * np.cos(theta)
        z = np.sqrt(1 - y*y) * np.sin(theta)
        
        # Scale to desired radius and shift to center
        position = np.array([x, y, z]) * radius + center
        
        # Create look-at matrix
        forward = position - center
        forward = forward / np.linalg.norm(forward)
        
        # Handle the case when camera is directly above/below
        right = np.cross(forward, np.array([0, 1, 0]))
        if np.allclose(right, 0):
            right = np.array([1, 0, 0])
        right = right / np.linalg.norm(right)
        
        # Compute camera orientation
        up = np.cross(right, forward)
        
        # Create rotation matrix (right-handed coordinate system)
        R = np.stack([right, up, -forward], axis=1)
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        
        poses.append(T)
    
    return poses

def render_depth_image(point_cloud, camera_pose, K, width, height):
    # Create renderer with matching window size
    vis = o3d.visualization.VisualizerWithKeyCallback()
    
    # Create window with exact dimensions matching the camera parameters
    vis.create_window(width=width, height=height, visible=False, window_name="render")
    vis.add_geometry(point_cloud)
    
    # Set up virtual camera
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic.intrinsic_matrix = K
    cam.intrinsic.width = width
    cam.intrinsic.height = height
    cam.extrinsic = np.linalg.inv(camera_pose)  # OpenGL convention
    
    # Set the camera view
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    
    # Capture both depth and color
    vis.poll_events()  # Update events
    vis.update_renderer()  # Update renderer
    depth = np.asarray(vis.capture_depth_float_buffer(do_render=True))
    rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    vis.destroy_window()
    return depth, rgb_bgr

def generate_and_save_views(point_file, n_views=100, output_dir="./synthetic_views"):
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate camera poses
    point_cloud = o3d.io.read_point_cloud(point_file)
    K, image_width, image_height = define_camera_intrinsics()
    poses = generate_spherical_poses(point_cloud, n_poses=n_views)
    
    # # Visualize point cloud and camera poses
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # vis.add_geometry(point_cloud)
    
    # # Create camera frustum geometries for each pose
    # for pose in poses:
    #     # Create coordinate frame showing camera pose
    #     cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30)
    #     cam_frame.transform(pose)
    #     vis.add_geometry(cam_frame)
    
    # # Set default camera viewpoint
    # ctr = vis.get_view_control()
    # ctr.set_zoom(0.8)
    
    # # Render visualization
    # vis.run()
    # vis.destroy_window()
    # Save poses and render views
    for i, pose in enumerate(poses):
        # Render depth image
        depth, rgb = render_depth_image(point_cloud, pose, K, image_width, image_height)
        
        print(depth.shape, depth.min(), depth.max())
        print(rgb.shape, rgb.min(), rgb.max())

        # Normalize depth for visualization (0-255)
        depth_vis = (depth / np.max(depth) * 255).astype(np.uint8)
        rgb_vis = (rgb / np.max(rgb) * 255).astype(np.uint8)
        
        # Save images and poses
        cv2.imwrite(f"{output_dir}/depth/0_{i:04d}.png", depth_vis)
        cv2.imwrite(f"{output_dir}/rgb/0_{i:04d}.png", rgb_vis)
        
        # Save camera pose
        np.savetxt(f"{output_dir}/pose/0_{i:04d}.txt", pose, fmt='%.6f')
    
    # Save camera intrinsics (padded to 4x4)
    K_padded = np.pad(K, ((0,1), (0,1)), mode='constant')
    K_padded[3,3] = 1
    np.savetxt(f"{output_dir}/intrinsics.txt", K_padded, fmt='%.6f')
    
    print(f"Generated {n_views} synthetic views in {output_dir}")
    
def prepare_nerf_dataset(data_dir="./synthetic_views", output_dir="./nerf_dataset"): 
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all images and poses
    images = []
    poses = []
    
    for i in range(len(os.listdir(data_dir)) // 3):  # Divide by 3 because we have rgb, depth and pose files
        # Load the pose
        pose = np.loadtxt(f"{data_dir}/pose/{i:04d}.txt")
        
        # Convert from camera-to-world to NeRF format if needed
        # Some NeRF implementations use different conventions
        
        # Copy the image file
        os.system(f"cp {data_dir}/rgb/{i:04d}.png {output_dir}/")
        
        # Add to our lists
        images.append(f"rgb_{i:04d}.png")
        poses.append(pose.tolist())
    
    # Load intrinsics
    K = np.loadtxt(f"{data_dir}/intrinsics.txt")
    
    # Create transforms.json (standard NeRF format)
    image_width = K[0, 2] * 2
    image_height = K[1, 2] * 2
    transforms = {
        "camera_angle_x": 2 * np.arctan(image_width / (2 * K[0, 0])),
        "frames": []
    }
    
    for i, (img, pose) in enumerate(zip(images, poses)):
        frame = {
            "file_path": img,
            "transform_matrix": pose
        }
        transforms["frames"].append(frame)
    
    # Save transforms.json
    with open(f"{output_dir}/transforms.json", 'w') as f:
        json.dump(transforms, f, indent=2)
    
    print(f"NeRF dataset prepared in {output_dir}")
    
def make_dirs(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/rgb", exist_ok=True)
    os.makedirs(f"{output_dir}/depth", exist_ok=True)
    os.makedirs(f"{output_dir}/pose", exist_ok=True)

if __name__ == "__main__":
    # Generate synthetic views
    point_folders = glob.glob('/media/UNIWA\\23903981/Data2/data/WPC_database/original_PCs/*.ply')[:4]
    for point_file in point_folders:
        # import ipdb; ipdb.set_trace()
        base_name = f'./synthetic_wpc_1024_view200/{point_file.split("/")[-1].split(".")[0]}'
        make_dirs(base_name)
        generate_bouding_box(point_file, base_name)
        generate_and_save_views(point_file, n_views=200, output_dir=base_name)
    
    # Prepare NeRF dataset
    # prepare_nerf_dataset(data_dir="./synthetic_wpc", output_dir="./nerf_wpc")