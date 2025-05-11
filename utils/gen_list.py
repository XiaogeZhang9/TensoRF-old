import os
import glob

datadir = 'synthetic_wpc_1024_view100'
with open('split/scannet_scenes.txt', 'w') as f:
    for scene in range(43):
        f.write(f'scene{scene:04d}_00\n')
    