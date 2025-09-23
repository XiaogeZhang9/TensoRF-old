import os
import glob
textures = os.listdir('/home/23903981/Documents/3DFRONT/wall_texture')
for texture in textures:
    root_path = f'./synthetic_3dfront_1024_view100_wall_tex/caf8c80f-961b-4b3f-991a-e1f9036efaee_LivingRoom-45932_{texture}'
    imgs = glob.glob(os.path.join(root_path, 'rgb', '*.png'))
    removed_imgs = [img for img in imgs if img.split('/')[-1].startswith('1_')]
    for img in removed_imgs:
        if os.path.isfile(img):
            print(f"Removing {img}")
            # os.remove(img)
    
