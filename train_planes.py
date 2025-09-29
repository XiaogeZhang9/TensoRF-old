
import os
from tqdm.auto import tqdm
from opt import config_parser



import json, random
from renderer import evaluation_distill, evaluation, evaluation_path
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask

from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from dataLoader.tankstemple import DistillPlane
from dataLoader import dataset_dict
import sys
import shutil
import torch
from tqdm import trange


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#renderer = OctreeRender_trilinear_fast_distill
def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None

def OctreeRender_trilinear_fast_distill_sigma(rays, tensorf, triplanes, N_S, n_c, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):
    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in trange(N_rays_all // chunk + int(N_rays_all % chunk > 0), total=N_rays_all // chunk + int(N_rays_all % chunk > 0), desc="Rendering"):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        with torch.no_grad():
            rgb_map, depth_map = tensorf.forward_distill_sigma(triplanes, N_S, n_c, rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
      
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        torch.cuda.empty_cache()
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None

def OctreeRender_trilinear_fast_distill_color_sh(rays, tensorf, triplanes, N_S, n_c, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):
    from tqdm import trange
    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in trange(N_rays_all // chunk + int(N_rays_all % chunk > 0), total=N_rays_all // chunk + int(N_rays_all % chunk > 0), desc="Rendering"):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        with torch.no_grad():
            rgb_map, depth_map = tensorf.forward_distill_color_sh(triplanes, N_S, n_c, rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
      
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        torch.cuda.empty_cache()
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None

def OctreeRender_trilinear_fast_distill(rays, tensorf, triplanes, N_S, n_c, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):
    from tqdm import trange
    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in trange(N_rays_all // chunk + int(N_rays_all % chunk > 0), total=N_rays_all // chunk + int(N_rays_all % chunk > 0), desc="Rendering"):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        with torch.no_grad():
            rgb_map, depth_map = tensorf.forward_distill(triplanes, N_S, n_c, rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
      
        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        torch.cuda.empty_cache()
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)

@torch.no_grad()
def copy_code(out_path, config_path):
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(os.path.join(out_path, 'code'), exist_ok=True)
    shutil.copyfile(__file__, os.path.join(out_path, 'code', os.path.basename(__file__)))
    shutil.copyfile(config_path, os.path.join(out_path, 'code', os.path.basename(config_path)))
    shutil.copyfile('renderer.py', os.path.join(out_path, 'code', 'renderer.py'))
    shutil.copyfile('models/tensoRF.py', os.path.join(out_path, 'code', 'tensoRF.py'))
    
@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    kwargs.update({'H':args.H})
    kwargs.update({'W':args.W})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)
        
    triplanes = torch.nn.ParameterList([
        torch.nn.Parameter(torch.randn((args.H, args.W), device=device, requires_grad=True)) for _ in range(3)
    ])
    logfolder = ''
    if args.ckpt_tri_density is not None:
        triplanes_density = torch.load(args.ckpt_tri_density, map_location=device)
        triplanes = torch.nn.ParameterList([
            torch.nn.Parameter(torch.cat([triplanes_density[i][:,:args.N_S].clone(), triplanes[i][:,args.N_S:].clone()], dim=1))
            for i in range(len(triplanes))
        ])
        logfolder = os.path.dirname(args.ckpt_tri_density)
    if args.ckpt_tri_color is not None:
        triplanes_color = torch.load(args.ckpt_tri_color, map_location=device)
        triplanes = torch.nn.ParameterList([
            torch.nn.Parameter(torch.cat([triplanes[i][:,:args.N_S].clone(), triplanes_color[i][:,args.N_S:].clone()], dim=1))
            for i in range(len(triplanes))
        ])
        logfolder = os.path.dirname(args.ckpt_tri_color)
    if args.ckpt_tri is not None:
        triplanes_loaded = torch.load(args.ckpt_tri, map_location=device)
        triplanes = torch.nn.ParameterList([torch.nn.Parameter(triplanes_loaded[i]) for i in range(len(triplanes_loaded))])
        logfolder = os.path.dirname(args.ckpt_tri)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation_distill(train_dataset,tensorf, args, OctreeRender_trilinear_fast_distill, triplanes, N_S=args.N_S, n_c=27,
                                    savePath=f'{logfolder}/imgs_train/', N_vis=-1, N_samples=-1, 
                                    white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.ckpt_tri.split("/")[-2]} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:    
        os.makedirs(f'{logfolder}/imgs_test', exist_ok=True)
        
        PSNRs_test = evaluation_distill(test_dataset,tensorf, args, OctreeRender_trilinear_fast_distill, triplanes, N_S=args.N_S, n_c=27, 
                                savePath=f'{logfolder}/imgs_test/', N_vis=-1, N_samples=-1, prtx='0_',
                                white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        
        # evaluation_pc(test_dataset,tensorf, args, args.H, args.W, f'{logfolder}/imgs_test_all/',
        #                         N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        # evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
        #                         N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.ckpt_tri.split("/")[-2]} test all psnr: {np.mean(PSNRs_test)} <========================')
    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, OctreeRender_trilinear_fast_distill, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    copy_code(logfolder, args.config)
    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        kwargs.update({'H':args.H})
        kwargs.update({'W':args.W})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct,
                    H=args.H, W=args.W)

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    triplanes = torch.nn.ParameterList([
        torch.nn.Parameter(torch.randn((args.H, args.W), device=device, requires_grad=True)) for _ in range(3)
    ])
    # optimizer = torch.optim.Adam(triplanes, betas=(0.9, 0.99))
        # Log initial values for debugging
    # print(triplanes[0].grad)
    # initial_values = [p.grad for p in triplanes]
    # initial_values = [p.data.mean().item() for p in triplanes]
    # print(f"Initial triplane mean values: {initial_values}")

    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")
    if args.ckpt_tri_density is not None:
        triplanes_density = torch.load(args.ckpt_tri_density, map_location=device)
        triplanes = torch.nn.ParameterList([
            torch.nn.Parameter(torch.cat([triplanes_density[i][:,:args.N_S].clone(), triplanes[i][:,args.N_S:].clone()], dim=1))
            for i in range(len(triplanes))
        ])
    if args.ckpt_tri_color is not None:
        triplanes_color = torch.load(args.ckpt_tri_color, map_location=device)
        triplanes = torch.nn.ParameterList([
            torch.nn.Parameter(torch.cat([triplanes[i][:,:args.N_S].clone(), triplanes_color[i][:,args.N_S:].clone()], dim=1))
            for i in range(len(triplanes))
        ])
        print('Load triplanes_color', args.ckpt_tri_color)
    if args.ckpt_tri is not None:
        triplanes_loaded = torch.load(args.ckpt_tri, map_location=device)
        triplanes = torch.nn.ParameterList([torch.nn.Parameter(triplanes_loaded[i]) for i in range(len(triplanes_loaded))])
    # os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
    # PSNRs_test = evaluation_distill(test_dataset,tensorf, args, OctreeRender_trilinear_fast_distill, triplanes, N_S=206, n_c=27, 
    #                                 savePath=f'{logfolder}/imgs_test_all/', N_vis=args.N_vis, N_samples=nSamples, 
    #                                 savePath=f'{logfolder}/imgs_test_all/', N_vis=args.N_vis, N_samples=nSamples, 
    #                                 white_bg = white_bg, ndc_ray=ndc_ray,device=device, prtx=f'{args.ckpt_tri.split(".")[0].split("_")[-1]}_')
    # print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')
    optimizer_sigma = torch.optim.Adam(triplanes, betas=(0.9, 0.99), lr=args.lr_init)
    optimizer_color = torch.optim.Adam(triplanes, betas=(0.9, 0.99), lr=args.lr_init)
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    planedataset = DistillPlane(shape=(args.H,args.H,args.H), N_S=args.N_S, n_c=27)
    # planedataset = torch.utils.data.Subset(planedataset, range(1000000))
    dataloader = torch.utils.data.DataLoader(planedataset, batch_size=args.batch_size, shuffle=True)
    dataloader_ordered = torch.utils.data.DataLoader(planedataset, batch_size=args.batch_size, shuffle=False)
    while True:
        # find unique camera poses
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)
        rays_o_unique = torch.unique(rays_train[:, :3], dim=0).to(device)
        if rays_o_unique.shape[0] == 100:
            break
    for iteration in pbar:
        if args.distill_sigma:
            for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Distilling Sigmas"):
                loss_sigma = tensorf.distill_density_total(batch, triplanes, H=args.H, W=args.W, N_S=args.N_S, n_c=27)
                optimizer_sigma.zero_grad()
                loss_sigma.backward()
                optimizer_sigma.step()
                # Save triplanes state
                triplanes_state = [p.data.clone() for p in triplanes] # Clone to avoid memory issues
            summary_writer.add_scalar('train/loss_density', loss_sigma.detach().item(), global_step=idx+iteration*len(dataloader))
            summary_writer.add_scalar('train/grad', triplanes[0].grad.mean().item(), global_step=idx+iteration*len(dataloader))
            initial_values = [p.data.max().item() for p in triplanes]
            print(f"Initial triplane mean values: {initial_values}")        
            if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
                torch.save(triplanes_state, f'{logfolder}/triplanes_sigma_{iteration:03d}.pt')
            # PSNRs, ssim, l_a, l_v = evaluation_distill(test_dataset,tensorf, args, OctreeRender_trilinear_fast_distill_sigma, triplanes, N_S=args.N_S, n_c=27, 
            #                             savePath=f'{logfolder}/imgs_test_sigma/', N_vis=args.N_vis, N_samples=nSamples, 
            #                             white_bg = white_bg, ndc_ray=ndc_ray,device=device, prtx=f'{iteration:04d}_', is_train=True)
            # summary_writer.add_scalar('eval/psnr_sigma', np.mean(PSNRs), global_step=iteration)
            # summary_writer.add_scalar('eval/ssim_sigma', np.mean(ssim), global_step=iteration)
            # summary_writer.add_scalar('eval/l_a_sigma', np.mean(l_a), global_step=iteration)
            # summary_writer.add_scalar('eval/l_v_sigma', np.mean(l_v), global_step=iteration)
            best_psnr = 0
            if np.mean(PSNRs) > best_psnr:
                best_psnr = np.mean(PSNRs)
                torch.save(triplanes_state, f'{logfolder}/triplanes_sigma_best.pt')
                
        if args.distill_color:
            for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Distilling Colors"):
                optimizer_color.zero_grad()
                loss_sigma = tensorf.distill_color_sh(batch, triplanes, rays_o_unique, train_dataset,H=args.H, W=args.W, N_S=args.N_S, n_c=27)
                loss_sigma.backward()
                optimizer_color.step()
                # Save triplanes state
                triplanes_state = [p.data.clone() for p in triplanes] # Clone to avoid memory issues
                summary_writer.add_scalar('train/loss_color', loss_sigma.detach().item(), global_step=idx+iteration*len(dataloader))
                summary_writer.add_scalar('train/grad', triplanes[0].grad.mean().item(), global_step=idx+iteration*len(dataloader))
            initial_values = [p.data.mean().item() for p in triplanes]
            print(f"Initial triplane mean values: {initial_values}")  
            summary_writer.add_scalar('train/x', triplanes[0].data.mean().item(), global_step=(iteration+1)*len(dataloader))      
            summary_writer.add_scalar('train/y', triplanes[1].data.mean().item(), global_step=(iteration+1)*len(dataloader))
            summary_writer.add_scalar('train/z', triplanes[2].data.mean().item(), global_step=(iteration+1)*len(dataloader))
            if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
                torch.save(triplanes_state, f'{logfolder}/triplanes_color_{iteration:03d}.pt')
            PSNRs, ssim, l_a, l_v = evaluation_distill(test_dataset,tensorf, args, OctreeRender_trilinear_fast_distill_color_sh, triplanes, N_S=args.N_S, n_c=27, 
                                        savePath=f'{logfolder}/imgs_test_color/', N_vis=args.N_vis, N_samples=nSamples, 
                                        white_bg = white_bg, ndc_ray=ndc_ray,device=device, prtx=f'{iteration:04d}_', is_train=True)
            summary_writer.add_scalar('eval/psnr_color', np.mean(PSNRs), global_step=iteration)
            summary_writer.add_scalar('eval/ssim_color', np.mean(ssim), global_step=iteration)
            summary_writer.add_scalar('eval/l_a_color', np.mean(l_a), global_step=iteration)
            summary_writer.add_scalar('eval/l_v_color', np.mean(l_v), global_step=iteration)
            PSNRs, ssim, l_a, l_v = evaluation_distill(test_dataset,tensorf, args, OctreeRender_trilinear_fast_distill, triplanes, N_S=args.N_S, n_c=27, 
                            savePath=f'{logfolder}/imgs_test_all/', N_vis=args.N_vis, N_samples=nSamples, 
                            white_bg = white_bg, ndc_ray=ndc_ray,device=device, prtx=f'{iteration:04d}_', is_train=True)
            summary_writer.add_scalar('eval/psnr_total', np.mean(PSNRs), global_step=iteration)
            summary_writer.add_scalar('eval/ssim_total', np.mean(ssim), global_step=iteration)
            summary_writer.add_scalar('eval/l_a_total', np.mean(l_a), global_step=iteration)
            summary_writer.add_scalar('eval/l_v_total', np.mean(l_v), global_step=iteration)
            best_psnr = 0
            if np.mean(PSNRs) > best_psnr:
                best_psnr = np.mean(PSNRs)
                torch.save(triplanes_state, f'{logfolder}/triplanes_color_best.pt')

if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)
    if  args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

