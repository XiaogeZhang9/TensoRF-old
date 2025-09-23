import os

import numpy as np

def write_colmap_cameras_txt(intrinsics_paths, output_path, width=1024, height=1024):
    """
    Generate a COLMAP cameras.txt file from a NeRF-style intrinsics.txt.
    Args:
        intrinsics_path: Path to intrinsics.txt (3x3 matrix, fx, fy, cx, cy)
        output_path: Path to write cameras.txt
        camera_id: Camera ID to use in cameras.txt (default: 1)
        width: Image width (default: 1024)
        height: Image height (default: 1024)
    """
    with open(output_path, "w") as f:
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        for idx, intrinsics_path in enumerate(intrinsics_paths):
            K = np.loadtxt(intrinsics_path)
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            f.write(f"{idx+1} PINHOLE {int(width)} {int(height)} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

def rotmat2qvec(R):
    """Convert rotation matrix to quaternion vector (w, x, y, z)"""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def write_colmap_images_txt(poses_paths, images_paths, output_path):
    """
    Generate a COLMAP images.txt file from a poses.txt.
    """
    with open(output_path, "w") as f:
        f.write("# Image list with one line of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# Number of images: 1\n")
        for idx, (pose_path, image_path) in enumerate(zip(poses_paths, images_paths)):
            Tcw = np.loadtxt(pose_path)
            # R = Tcw[:3, :3]
            # t = Tcw[:3, 3]
            # qvec = rotmat2qvec(R)
            # tvec = -R @ t 

            coord_convert = np.diag([1, 1, 1])
            R_c2w = Tcw[:3, :3] @ coord_convert
            t_c2w = Tcw[:3, 3] @ coord_convert
            R_w2c = R_c2w.T
            t_w2c = -R_w2c @ t_c2w
            qvec = rotmat2qvec(R_w2c)
            tvec = t_w2c
            
            f.write(f"{idx+1} {qvec[0]:.6f} {qvec[1]:.6f} {qvec[2]:.6f} {qvec[3]:.6f} {tvec[0]:.6f} {tvec[1]:.6f} {tvec[2]:.6f} 1 {image_path}\n")
            f.write(f"\n")
            
def write_colmap_points3D_txt(output_path):
    """
    Generate a COLMAP points3D.txt file from a points3d.ply.
    """
    with open(output_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")

if __name__ == "__main__":
    import argparse
    import glob
    parser = argparse.ArgumentParser(description="Generate COLMAP cameras.txt from NeRF intrinsics.txt")
    # parser.add_argument("--root_dir", type=str, default='./synthetic_3dfront_1024_view100_wall_tex/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932', help="root dir")
    # parser.add_argument("--out_dir", type=str, default="./log_3dfront/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932_wall_tex/sparse", help="Output cameras.txt path")
    # parser.add_argument("--img_dir", type=str, default="./log_3dfront/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932_wall_tex/imgs", help="Image directory")
    parser.add_argument("--root_dir", type=str, default='./synthetic_3dfront_1024_view100_wall_tex/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932', help="pose and intrinsics dir") 
    parser.add_argument("--out_dir", type=str, default="./log_3dfront/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932_wall_tex/sparse", help="Output cameras.txt path")
    parser.add_argument("--img_dir", type=str, default="./log_3dfront/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932_wall_tex/rgb", help="Image directory")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    args = parser.parse_args()

    intrinsics_paths = sorted(glob.glob(os.path.join(args.root_dir, 'intrinsics.txt')))
    poses_paths = sorted(glob.glob(os.path.join(args.root_dir, 'pose', '0_*.txt')))
    images_paths = sorted(os.listdir(os.path.join(args.img_dir)))
    images_paths = [imgs_path if imgs_path.endswith('.png') else None for imgs_path in images_paths]
    # images_paths = [imgs_path if imgs_path.startswith('0_') else None for imgs_path in images_paths]
    images_paths = [imgs_path for imgs_path in images_paths if imgs_path is not None]
    print(len(poses_paths), len(images_paths))
    # import ipdb; ipdb.set_trace()
    os.makedirs(args.out_dir, exist_ok=True)
    write_colmap_cameras_txt(
        intrinsics_paths,
        os.path.join(args.out_dir, 'cameras.txt'),
        width=args.width,
        height=args.height
    )
    write_colmap_images_txt(
        poses_paths,
        images_paths,
        os.path.join(args.out_dir, 'images.txt'),
    )
    write_colmap_points3D_txt(
        os.path.join(args.out_dir, 'points3D.txt'),
    )
