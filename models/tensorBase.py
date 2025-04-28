import torch
import torch.nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time


def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):

    rgb = features
    return rgb

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender_PE(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


def visualize_rays(rays_pts, aabb):
    # Convert distances to 3D points
    # Visualize sampled points and bounding box with open3d
    import open3d as o3d
    
    # Create point cloud from sampled points
    pcd = o3d.geometry.PointCloud()
    points = rays_pts.reshape(-1, 3).detach().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create bounding box lines
    aabb = aabb.detach().cpu().numpy()
    corners = np.array([
        [aabb[0,0], aabb[0,1], aabb[0,2]], # 000
        [aabb[1,0], aabb[0,1], aabb[0,2]], # 100  
        [aabb[1,0], aabb[1,1], aabb[0,2]], # 110
        [aabb[0,0], aabb[1,1], aabb[0,2]], # 010
        [aabb[0,0], aabb[0,1], aabb[1,2]], # 001
        [aabb[1,0], aabb[0,1], aabb[1,2]], # 101
        [aabb[1,0], aabb[1,1], aabb[1,2]], # 111
        [aabb[0,0], aabb[1,1], aabb[1,2]], # 011
    ])
    lines = [[0,1], [1,2], [2,3], [3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7]]
    colors = [[1,0,0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd, line_set])
    
class TensorBase(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, density_n_comp = 8, appearance_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,6.0],
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus'):
        super(TensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device=device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio


        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]


        self.init_svd_volume(gridSize[0], device)

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass
    
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])


    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        """
        Sample points along rays for volume rendering.
        
        Args:
            rays_o: Ray origins [N, 3]
            rays_d: Ray directions [N, 3] 
            is_train: Whether in training mode (adds random jitter if True)
            N_samples: Number of samples per ray. If -1, uses self.nSamples
            
        Returns:
            rays_pts: Sampled 3D points along rays [N, samples, 3]
            interpx: Distance of sampled points along rays [N, samples]
            ~mask_outbbox: Boolean mask of points inside bounding box [N, samples]
        """
        # Use default number of samples if not specified
        N_samples =  self.nSamples # N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        # Handle division by zero in ray directions
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        # Calculate intersection parameters with axis-aligned bounding box (AABB)
        rate_a = (self.aabb[1] - rays_o) / vec  # Intersection with max corner
        rate_b = (self.aabb[0] - rays_o) / vec  # Intersection with min corner
        
        # Get minimum valid intersection distance, clamped between near/far bounds
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        # Generate sampling distances along rays
        rng = torch.arange(N_samples)[None].float()
        if is_train:
            # Add random offset during training for regularization
            rng = rng.repeat(rays_d.shape[-2],1) 
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]

        # Check which points lie outside the bounding box
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):


        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)
        # print(self.aabb)
        # print(dense_xyz.min(), dense_xyz.max())
        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))
        # print(new_aabb)
        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        # #######
        # # Extract ray origins (xyz position) and ray directions (xyz vector) from rays tensor
        # # rays_o: ray origin coordinates in 3D space (x,y,z)
        # # rays_d: ray direction unit vectors in 3D space (dx,dy,dz)

        # # Visualize rays and bounding box with Open3D
        # import open3d as o3d
        
        # # Create line set for rays
        # ray_points = []
        # ray_lines = []
        # ray_length = 1000.0  # Length to extend rays for visualization
        # rays_o, rays_d = all_rays[..., :3], all_rays[..., 3:6]

        # # Sample a subset of rays to visualize (e.g. first 100)
        # for i, idx in enumerate(range(0, len(rays_o), 10000)):
        #     start = rays_o[idx].cpu().numpy()
        #     end = start + ray_length * rays_d[idx].cpu().numpy()
        #     ray_points.extend([start, end])
        #     ray_lines.append([2*i, 2*i+1])
        # # Create Open3D line set for rays
        # ray_lineset = o3d.geometry.LineSet()
        # ray_lineset.points = o3d.utility.Vector3dVector(ray_points)
        # ray_lineset.lines = o3d.utility.Vector2iVector(ray_lines)
        # ray_lineset.colors = o3d.utility.Vector3dVector([[1,0,0] for _ in range(len(ray_lines))])  # Red rays
        
        # # Create bounding box wireframe
        # aabb_points = self.aabb.cpu().numpy()  # [2,3] array of min/max points
        # bbox_points = [
        #     [aabb_points[0,0], aabb_points[0,1], aabb_points[0,2]],  # min point
        #     [aabb_points[1,0], aabb_points[0,1], aabb_points[0,2]],
        #     [aabb_points[1,0], aabb_points[1,1], aabb_points[0,2]], 
        #     [aabb_points[0,0], aabb_points[1,1], aabb_points[0,2]],
        #     [aabb_points[0,0], aabb_points[0,1], aabb_points[1,2]],
        #     [aabb_points[1,0], aabb_points[0,1], aabb_points[1,2]],
        #     [aabb_points[1,0], aabb_points[1,1], aabb_points[1,2]],
        #     [aabb_points[0,0], aabb_points[1,1], aabb_points[1,2]]
        # ]
        # bbox_lines = [[0,1], [1,2], [2,3], [3,0],
        #                 [4,5], [5,6], [6,7], [7,4],
        #                 [0,4], [1,5], [2,6], [3,7]]
        
        # # Create Open3D line set for bounding box
        # bbox_lineset = o3d.geometry.LineSet()
        # bbox_lineset.points = o3d.utility.Vector3dVector(bbox_points) 
        # bbox_lineset.lines = o3d.utility.Vector2iVector(bbox_lines)
        # bbox_lineset.colors = o3d.utility.Vector3dVector([[0,1,0] for _ in range(len(bbox_lines))])  # Green box
        
        # # Visualize
        # o3d.visualization.draw_geometries([ray_lineset, bbox_lineset])
        # import ipdb; ipdb.set_trace()
        # This code chunk processes rays in batches to filter out rays that don't intersect with the scene
        # For each chunk of rays:
        # 1. If bbox_only=True: Checks if rays intersect with the scene bounding box by computing intersection points
        #    - Handles division by zero in ray directions
        #    - Calculates entry/exit points of rays with bounding box
        #    - Keeps rays where exit point is after entry point (t_max > t_min)
        # 2. If bbox_only=False: Samples points along rays and checks if they hit non-empty space
        #    - Samples N_samples points along each ray
        #    - Uses alpha mask to check if any sampled points have non-zero density
        # 3. Stores the boolean mask indicating which rays to keep
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            
            if bbox_only:
                # Handle division by zero in ray directions by replacing 0s with small epsilon
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                
                # Calculate intersection parameters with axis-aligned bounding box (AABB)
                # rate_a: intersection parameters with max corner of AABB
                rate_a = (self.aabb[1] - rays_o) / vec  # (max_corner - ray_origin) / ray_direction
                # rate_b: intersection parameters with min corner of AABB  
                rate_b = (self.aabb[0] - rays_o) / vec  # (min_corner - ray_origin) / ray_direction
                
                # Find entry point (t_min) - furthest of the near intersections
                t_min = torch.minimum(rate_a, rate_b).amax(-1)  # Latest entry across x,y,z
                # Find exit point (t_max) - closest of the far intersections
                t_max = torch.maximum(rate_a, rate_b).amin(-1)  # Earliest exit across x,y,z
                
                # Ray intersects AABB if exit is after entry
                mask_inbbox = t_max > t_min  # True if ray intersects box
            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)
                print('{}/{}'.format(torch.sum(mask_inbbox), chunk))
            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]


    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)


    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
            

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        

        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha


    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            
            # Calculate distances between adjacent samples along the ray
            # z_vals contains the distance of each sample point from ray origin
            # We compute differences between consecutive z values to get segment lengths
            # For the last sample, we append a zero distance
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            
            # In NDC space, we need to account for ray direction normalization
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)  # Length of direction vectors
            dists = dists * rays_norm  # Scale distances by ray length
            viewdirs = viewdirs / rays_norm  # Normalize directions to unit vectors
            
        else:
            # For regular world space coordinates
            # Sample points along each ray using regular sampling strategy
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            
            # Calculate distances between samples, same as in NDC case
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            
        # Reshape viewing directions to match sampled points
        # Each sampled point needs its corresponding viewing direction for rendering
        # -1 preserves total size, 1 adds a new dimension, 3 is for xyz coordinates
        # expand replicates the directions for each sample along the ray
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        if self.alphaMask is not None:
            # Sample alpha values for valid ray points from the alpha mask volume
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            
            # Create mask for points with non-zero alpha
            alpha_mask = alphas > 0
            
            # Start with invalid rays
            ray_invalid = ~ray_valid
            
            # For valid rays, mark as invalid if alpha is zero
            ray_invalid[ray_valid] |= (~alpha_mask)
            
            # Update valid ray mask
            ray_valid = ~ray_invalid


        # Initialize empty tensors for density (sigma) and color (rgb) for all sampled points
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)  # Shape: [N_rays, N_samples]
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)  # Shape: [N_rays, N_samples, 3]

        # For valid ray points, compute density features and convert to actual density values
        if ray_valid.any():
            # Normalize coordinates to [-1,1] range for grid sampling
            xyz_sampled = self.normalize_coord(xyz_sampled)
            # import ipdb; ipdb.set_trace()
            # Get density features for valid points
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid]) # factorize 3d to plane and line
            # Convert features to density values 
            validsigma = self.feature2density(sigma_feature)
            # Assign computed densities back to valid points
            sigma[ray_valid] = validsigma

        # Convert raw density (sigma) to alpha values and compute weights for volume rendering
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        # Only compute colors for points with significant contribution to final image
        app_mask = weight > self.rayMarch_weight_thres

        # For points that contribute significantly, compute RGB colors
        if app_mask.any():
            # Get appearance features
            import ipdb; ipdb.set_trace()
            app_features = self.compute_appfeature(xyz_sampled[app_mask]) #[N,3]
            # Render RGB values using the chosen rendering module
            valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features) #[N,3]
            # Assign computed colors back to contributing points
            rgb[app_mask] = valid_rgbs #[B,R,3]

        # Compute final pixel colors through volume rendering equation
        acc_map = torch.sum(weight, -1)  # Accumulated opacity along each ray
        rgb_map = torch.sum(weight[..., None] * rgb, -2)  # Weighted sum of colors

        # Handle background color - either white or random during training
        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])  # Add white background weighted by remaining transparency

        # Ensure output colors are in valid range
        rgb_map = rgb_map.clamp(0,1)

        # Compute depth map without tracking gradients
        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)  # Expected depth
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]  # Add background depth

        return rgb_map, depth_map # Return final pixel colors and depth values