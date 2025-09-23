import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


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

def plot_nerf_psnr(ax):
    root_dir = 'log'
    exp_name1 = 'tensorf_VM_wpc_1024_view100_ship_sh'
    exp_name2 = 'tensorf_VM_wpc_1024_view100_pencontainer_sh'
    exp_name3 = 'tensorf_VM_wpc_1024_view100_pineapple_sh'
    exp_name4 = 'tensorf_VM_wpc_1024_view100_mushroom_sh'
    exp_name5 = 'tensorf_VM_wpc_1024_view100_house_sh'
    exp_name6 = 'tensorf_VM_wpc_1024_view100_litchi_sh'
    iter_list1, psnr_list1 = read_psnr(root_dir, exp_name1)
    iter_list2, psnr_list2 = read_psnr(root_dir, exp_name2)
    iter_list3, psnr_list3 = read_psnr(root_dir, exp_name3)
    iter_list4, psnr_list4 = read_psnr(root_dir, exp_name4)
    iter_list5, psnr_list5 = read_psnr(root_dir, exp_name5)
    iter_list6, psnr_list6 = read_psnr(root_dir, exp_name6)
    ax.plot(iter_list1, psnr_list1, label=exp_name1)
    ax.plot(iter_list2, psnr_list2, label=exp_name2)
    ax.plot(iter_list3, psnr_list3, label=exp_name3)
    ax.plot(iter_list4, psnr_list4, label=exp_name4)
    ax.plot(iter_list5, psnr_list5, label=exp_name5)
    ax.plot(iter_list6, psnr_list6, label=exp_name6)
    ax.legend()

def read_psnr2(root_dir, exp_name):
    sur_list = ['018', '028', '038', '048', '058']
    psnr_list = []
    for sur in sur_list:
        path = os.path.join(root_dir, exp_name+'_'+sur, 'imgs_test_all', '0_mean.txt')
        with open(path, 'r') as f:
            lines = f.readlines()
            psnr_list.append(float(lines[0].strip()))
    return psnr_list

def plot_distill_psnr(ax):
    root_dir = 'log_triplane3'
    exp_name1 = 'tensorf_VM_wpc_1024_view100_ship_lr0.005_sh'
    exp_name2 = 'tensorf_VM_wpc_1024_view100_pencontainer_lr0.005_sh'
    exp_name3 = 'tensorf_VM_wpc_1024_view100_pineapple_lr0.005_sh'
    exp_name4 = 'tensorf_VM_wpc_1024_view100_mushroom_lr0.005_sh'
    exp_name5 = 'tensorf_VM_wpc_1024_view100_house_lr0.005_sh'
    exp_name6 = 'tensorf_VM_wpc_1024_view100_litchi_lr0.005_sh'
    psnr_list1 = read_psnr2(root_dir, exp_name1)
    psnr_list2 = read_psnr2(root_dir, exp_name2)
    psnr_list3 = read_psnr2(root_dir, exp_name3)
    psnr_list4 = read_psnr2(root_dir, exp_name4)
    psnr_list5 = read_psnr2(root_dir, exp_name5)
    psnr_list6 = read_psnr2(root_dir, exp_name6)
    ax.plot(psnr_list1, label=exp_name1)
    ax.plot(psnr_list2, label=exp_name2)
    ax.plot(psnr_list3, label=exp_name3)
    ax.plot(psnr_list4, label=exp_name4)
    ax.plot(psnr_list5, label=exp_name5)
    ax.plot(psnr_list6, label=exp_name6)
    ax.legend()

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plot_nerf_psnr(ax[1])
    plot_distill_psnr(ax[0])
    plt.show()