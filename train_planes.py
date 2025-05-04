
import os
from tqdm.auto import tqdm
from opt import config_parser



import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from dataLoader.tankstemple import DistillPlane
from dataLoader import dataset_dict
import sys



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#renderer = OctreeRender_trilinear_fast_distill


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
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)
    
    triplanes = torch.nn.ParameterList([
        torch.nn.Parameter(torch.randn((args.H, args.W), device=device, requires_grad=True)) for _ in range(3)
    ])
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
    if args.ckpt_tri is not None:
        triplanes_loaded = torch.load(args.ckpt_tri, map_location=device)
        triplanes = torch.nn.ParameterList([torch.nn.Parameter(triplanes_loaded[i]) for i in range(len(triplanes_loaded))])
        
    logfolder = os.path.dirname(args.ckpt_tri_color)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(test_dataset,tensorf, args, OctreeRender_trilinear_fast, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.ckpt_tri.split("/")[-2]} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation_distill(test_dataset,tensorf, args, OctreeRender_trilinear_fast_distill, triplanes, N_S=args.N_S, n_c=27, 
                                savePath=f'{logfolder}/imgs_test_all/', N_vis=-1, N_samples=-1, prtx='0_',
                                white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        
        # evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
        #                         N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.ckpt_tri.split("/")[-2]} test all psnr: {np.mean(PSNRs_test)} <========================')
    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, OctreeRender_trilinear_fast, f'{logfolder}/imgs_path_all/',
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
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)



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
    #                                 white_bg = white_bg, ndc_ray=ndc_ray,device=device, prtx=f'{args.ckpt_tri.split(".")[0].split("_")[-1]}_')
    # print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')
    optimizer = torch.optim.Adam(triplanes, betas=(0.9, 0.99), lr=args.lr_init)
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    planedataset = DistillPlane(shape=(args.H,args.H,args.H), N_S=args.N_S, n_c=27)
    # planedataset = torch.utils.data.Subset(planedataset, range(1000000))
    dataloader = torch.utils.data.DataLoader(planedataset, batch_size=args.batch_size, shuffle=True)
    dataloader_ordered = torch.utils.data.DataLoader(planedataset, batch_size=args.batch_size, shuffle=False)
    # for iteration in pbar:
    #     for batch in tqdm(dataloader, total=len(dataloader), desc="Distilling Planes"):
    #         loss_sigma = tensorf.distill_density_total(batch, triplanes, H=800, W=800, N_S=206, n_c=27)
    #         optimizer.zero_grad()
    #         loss_sigma.backward()
    #         optimizer.step()
    #         # Save triplanes state
    #         triplanes_state = [p.data.clone() for p in triplanes] # Clone to avoid memory issues
    #     print(triplanes[0].grad.max())
    #     initial_values = [p.data.max().item() for p in triplanes]
    #     print(f"Initial triplane mean values: {initial_values}")        
    #     torch.save(triplanes_state, f'{logfolder}/triplanes_{iteration:03d}.pt')
    while True:
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)
        rays_o_unique = torch.unique(rays_train[:, :3], dim=0).to(device)
        if rays_o_unique.shape[0] == 100:
            break
    for iteration in pbar:
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Distilling Planes"):
            optimizer.zero_grad()
            loss_sigma = tensorf.distill_color_sh(batch, triplanes, rays_o_unique, train_dataset,H=args.H, W=args.W, N_S=args.N_S, n_c=27)
            loss_sigma.backward()
            optimizer.step()
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
        #     color_chunks = []
        #     color_gts = []
        #     with torch.no_grad():
        #         print(dataloader_ordered)
        #         for idx, batch in tqdm(enumerate(dataloader_ordered), total=len(dataloader_ordered), desc="Distilling Planes"):
        #             loss_sigma, color_chunk, color_gt = tensorf.distill_color_total(batch, triplanes, rays_o_unique, train_dataset,H=args.H, W=args.W, N_S=args.N_S, n_c=27)
        #             color_chunks.append(color_chunk[::256,0,:].detach().cpu())
        #             color_gts.append(color_gt[::256,0,:].detach().cpu())
        #         import ipdb; ipdb.set_trace()
        #         color_chunks = torch.cat(color_chunks, dim=0).detach().cpu().numpy()
        #         color_gts = torch.cat(color_gts, dim=0).detach().cpu().numpy()
        #         import open3d as o3d
        #         coords = tensorf.coords.reshape(-1,3)[::256].detach().cpu().numpy()
        #         pred = o3d.geometry.PointCloud()
        #         pred.points = o3d.utility.Vector3dVector(coords)
        #         pred.colors = o3d.utility.Vector3dVector(color_chunks)
        #         o3d.io.write_point_cloud(f'{logfolder}/color_chunks_full_{iteration:03d}.ply', pred)
        #         gt = o3d.geometry.PointCloud()
        #         gt.points = o3d.utility.Vector3dVector(coords)
        #         gt.colors = o3d.utility.Vector3dVector(color_gts)
        #         o3d.io.write_point_cloud(f'{logfolder}/color_gts_full_{iteration:03d}.ply', gt)
        #         pred = color_chunks.reshape(-1,3)
        #         import matplotlib.pyplot as plt
        #         plt.subplot(2,3,1)
        #         plt.hist(pred[:,0], bins=100)
        #         plt.subplot(2,3,2)
        #         plt.hist(pred[:,1], bins=100)
        #         plt.subplot(2,3,3)
        #         plt.hist(pred[:,2], bins=100)
        #         gt = color_gts.reshape(-1,3)
        #         plt.subplot(2,3,4)
        #         plt.hist(gt[:,0], bins=100)
        #         plt.subplot(2,3,5)
        #         plt.hist(gt[:,1], bins=100)
        #         plt.subplot(2,3,6)
        #         plt.hist(gt[:,2], bins=100)
        #         plt.savefig(f'/media/UNIWA\23903981/Data/onedrive/a-paper3/dist_color.png')
            torch.save(triplanes_state, f'{logfolder}/triplanes_{iteration:03d}.pt')
            PSNRs, ssim, l_a, l_v = evaluation_distill(test_dataset,tensorf, args, OctreeRender_trilinear_fast_distill, triplanes, N_S=args.N_S, n_c=27, 
                                        savePath=f'{logfolder}/imgs_train_all/', N_vis=args.N_vis, N_samples=nSamples, 
                                        white_bg = white_bg, ndc_ray=ndc_ray,device=device, prtx=f'{iteration:04d}_', is_train=True)
            summary_writer.add_scalar('eval/psnr', np.mean(PSNRs), global_step=iteration)
            summary_writer.add_scalar('eval/ssim', np.mean(ssim), global_step=iteration)
            summary_writer.add_scalar('eval/l_a', np.mean(l_a), global_step=iteration)
            summary_writer.add_scalar('eval/l_v', np.mean(l_v), global_step=iteration)

    #     ray_idx = trainingSampler.nextids()
    #     rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

    #     #rgb_map, alphas_map, depth_map, weights, uncertainty
    #     rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.batch_size,
    #                             N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

    #     loss = torch.mean((rgb_map - rgb_train) ** 2)


    #     # loss
    #     total_loss = loss
    #     if Ortho_reg_weight > 0:
    #         loss_reg = tensorf.vector_comp_diffs()
    #         total_loss += Ortho_reg_weight*loss_reg
    #         summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
    #     if L1_reg_weight > 0:
    #         loss_reg_L1 = tensorf.density_L1()
    #         total_loss += L1_reg_weight*loss_reg_L1
    #         summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

    #     if TV_weight_density>0:
    #         TV_weight_density *= lr_factor
    #         loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
    #         total_loss = total_loss + loss_tv
    #         summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
    #     if TV_weight_app>0:
    #         TV_weight_app *= lr_factor
    #         loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
    #         total_loss = total_loss + loss_tv
    #         summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

    #     optimizer.zero_grad()
    #     total_loss.backward()
    #     optimizer.step()

    #     loss = loss.detach().item()
        
    #     PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
    #     summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
    #     summary_writer.add_scalar('train/mse', loss, global_step=iteration)


    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = param_group['lr'] * lr_factor

    #     # Print the current values of the losses.
    #     if iteration % args.progress_refresh_rate == 0:
    #         pbar.set_description(
    #             f'Iteration {iteration:05d}:'
    #             + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
    #             + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
    #             + f' mse = {loss:.6f}'
    #         )
    #         PSNRs = []

    #     if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
    #         PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
    #                                 prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
    #         summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)



    #     if iteration in update_AlphaMask_list:

    #         if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
    #             reso_mask = reso_cur
    #         new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
    #         if iteration == update_AlphaMask_list[0]:
    #             tensorf.shrink(new_aabb)
    #             # tensorVM.alphaMask = None
    #             L1_reg_weight = args.L1_weight_rest
    #             print("continuing L1_reg_weight", L1_reg_weight)


    #         if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
    #             # filter rays outside the bbox
    #             allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
    #             trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)


    #     if iteration in upsamp_list:
    #         n_voxels = N_voxel_list.pop(0)
    #         reso_cur = N_to_reso(n_voxels, tensorf.aabb)
    #         nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
    #         tensorf.upsample_volume_grid(reso_cur)

    #         if args.lr_upsample_reset:
    #             print("reset lr to initial")
    #             lr_scale = 1 #0.1 ** (iteration / args.n_iters)
    #         else:
    #             lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
    #         grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
    #         optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        

    # tensorf.save(f'{logfolder}/{args.expname}.th')

    # if args.render_train:
    #     os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
    #     train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
    #     PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
    #                             N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
    #     print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    # if args.render_test:


    # if args.render_path:
    #     c2ws = test_dataset.render_path
    #     # c2ws = test_dataset.poses
    #     print('========>',c2ws.shape)
    #     os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
    #     evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
    #                             N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


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

