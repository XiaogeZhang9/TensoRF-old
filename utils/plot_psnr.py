import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

root_dir = 'log'
exp_name1 = 'tensorf_VM_wpc_1024_view100_ship_sh'
exp_name2 = 'tensorf_VM_wpc_1024_view100_pencontainer_sh'
exp_name3 = 'tensorf_VM_wpc_1024_view100_pineapple_sh'
exp_name4 = 'tensorf_VM_wpc_1024_view100_mushroom_sh'
exp_name5 = 'tensorf_VM_wpc_1024_view100_house_sh'
exp_name6 = 'tensorf_VM_wpc_1024_view100_litchi_sh'
def read_psnr(root_dir, exp_name):
    files = glob.glob(os.path.join(root_dir, exp_name, 'imgs_vis/*.txt'))
    files.sort()
    psnr_list = []
    iter_list = []
    for file in files:
        print(file)
        iter = int(file.split('/')[-1].split('_')[0])
        with open(file, 'r') as f:
            lines = float(f.readlines()[0].strip())
            psnr_list.append(lines)
            iter_list.append(iter)
    return iter_list, psnr_list

iter_list1, psnr_list1 = read_psnr(root_dir, exp_name1)
iter_list2, psnr_list2 = read_psnr(root_dir, exp_name2)
iter_list3, psnr_list3 = read_psnr(root_dir, exp_name3)
iter_list4, psnr_list4 = read_psnr(root_dir, exp_name4)
iter_list5, psnr_list5 = read_psnr(root_dir, exp_name5)
iter_list6, psnr_list6 = read_psnr(root_dir, exp_name6)
plt.plot(iter_list1, psnr_list1, label=exp_name1)
plt.plot(iter_list2, psnr_list2, label=exp_name2)
plt.plot(iter_list3, psnr_list3, label=exp_name3)
plt.plot(iter_list4, psnr_list4, label=exp_name4)
plt.plot(iter_list5, psnr_list5, label=exp_name5)
plt.plot(iter_list6, psnr_list6, label=exp_name6)
plt.legend()
plt.show()
