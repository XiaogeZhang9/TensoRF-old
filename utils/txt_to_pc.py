import numpy as np

def txt_to_ply(txt_path, ply_path):
    """
    Reads a txt file where each row contains x y z r g b alpha,
    and saves the data as a PLY file.
    """
    # Load data
    data = np.loadtxt(txt_path)
    print('read data', data.shape)
    if data.shape[1] != 7:
        raise ValueError("Each row must have 7 values: x y z r g b alpha")
    xyz = data[:, :3]
    rgb = (data[:, 3:6] * 255).astype(np.uint8)
    alpha = data[:, 6]
    

    # filter based on sigma
    valid_mask = alpha > 5
    xyz = xyz[valid_mask]
    rgb = rgb[valid_mask]
    alpha = alpha[valid_mask]
    print('filter data', xyz.shape)
    n_points = xyz.shape[0]
    # Write PLY header
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n_points}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]
    data = np.concatenate([xyz, rgb], axis=-1)
    with open(ply_path, 'w') as f:
        for line in header:
            f.write(line + '\n')
        np.savetxt(f, data, fmt='%.6f %.6f %.6f %d %d %d')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--ply_path", type=str, required=True)
    args = parser.parse_args()
    txt_to_ply(args.txt_path, args.ply_path)
