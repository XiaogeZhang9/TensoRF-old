import pathlib
import pycolmap
import os
import numpy as np

output_path: pathlib.Path
image_dir: pathlib.Path
def sparse_dense_recon(output_path, image_dir):
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    output_path.mkdir()
    mvs_path = output_path / "mvs"
    database_path = output_path / "database.db"

    pycolmap.extract_features(database_path, image_dir)
    pycolmap.match_exhaustive(database_path)
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    maps[0].write(output_path)
    # dense reconstruction
    pycolmap.undistort_images(mvs_path, output_path, image_dir)
    pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
    pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)


import argparse
import multiprocessing
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pycolmap

from . import logger
from .triangulation import (
    OutputCapture,
    estimation_and_geometric_verification,
    import_features,
    import_matches,
    parse_option_args,
)
from .utils.database import COLMAPDatabase


def create_empty_db(database_path: Path):
    if database_path.exists():
        logger.warning("The database already exists, deleting it.")
        database_path.unlink()
    logger.info("Creating an empty database...")
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    db.commit()
    db.close()


def import_images(
    image_dir: Path,
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: Optional[List[str]] = None,
    options: Optional[Dict[str, Any]] = None,
):
    logger.info("Importing images into the database...")
    if options is None:
        options = {}
    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f"No images found in {image_dir}.")
    with pycolmap.ostream():
        pycolmap.import_images(
            database_path,
            image_dir,
            camera_mode,
            image_list=image_list or [],
            options=options,
        )


def get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(database_path)
    images = {}
    for name, image_id in db.execute("SELECT name, image_id FROM images;"):
        images[name] = image_id
    db.close()
    return images
    
def recon_with_pose(image_path, pose_path):
    pose_files = sorted(os.listdir(os.path.join(pose_path, 'pose')))
    img_files  = sorted(os.listdir(os.path.join(image_path, 'imgs')))
    pose_files = [x for x in pose_files if x.startswith('0_') and x.endswith('.txt')]
    img_files = [x for x in img_files if x.startswith('058879') and x.endswith('.png')]
    
    recon = pycolmap.Reconstruction()

    # Add camera
    camera = pycolmap.Camera(
        model='PINHOLE',
        width=1024,
        height=1024,
        params=[800, 800, 512, 512]
    )
    camera_id = recon.add_camera(camera)

    db_path = f'{os.path.dirname(image_path)}/recons_with_pose/database.db'
    create_empty_db(db_path)
    # Add images with known poses
    for i, (img_path, Tcw) in enumerate(zip(img_files, pose_files)):
        Tcw = np.loadtxt(os.path.join(pose_path, 'pose', Tcw))
        R = Tcw[:3, :3]
        t = Tcw[:3, 3]
        qvec = rotmat2qvec(R)
        tvec = -R @ t  # convert from c2w to COLMAP world-to-camera

        pycolmap.import_images(
            db_path,
            image_path,
            pycolmap.CameraMode.PINHOLE,
            image_list=[img_path],
            options={},
        )
        
        img = pycolmap.Image()
        img.name = os.path.basename(img_path)
        img.camera_id = camera_id
        img.qvec = qvec
        img.tvec = tvec
        image_id = recon.add_image(img)
        recon.register_image(image_id)

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

# image_root = 'log_3dfront/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932/imgs'
# image_dir = pathlib.Path(image_root)
# output_root = 'log_3dfront/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932/recons'
# output_path = pathlib.Path(output_root)
# sparse_dense_recon(output_path, image_dir)
pose_root = 'synthetic_3dfront_1024_view100_wall_shallow/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932'
image_root = 'log_3dfront/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932'
recon_with_pose(image_root, pose_root)