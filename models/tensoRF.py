from .tensorBase import *
from .sh import eval_sh_bases
from dataLoader.ray_utils import get_ray_directions
from tqdm import trange

class TensorVM(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVM, self).__init__(aabb, gridSize, device, **kargs)
        

    def init_svd_volume(self, res, device):
        self.plane_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, res), device=device))
        self.line_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, 1), device=device))
        self.basis_mat = torch.nn.Linear(self.app_n_comp * 3, self.app_dim, bias=False, device=device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.line_coef, 'lr': lr_init_spatialxyz}, {'params': self.plane_coef, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_features(self, xyz_sampled):

        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach()
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach()

        plane_feats = F.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        return sigma_feature, app_features

    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_feats = F.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        
        return app_features
    

    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            # print(self.line_coef.shape, vector_comps[idx].shape)
            n_comp, n_size = vector_comps[idx].shape[:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape)
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape,non_diagonal.shape)
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        
        return self.vectorDiffs(self.line_coef[:,-self.density_n_comp:]) + self.vectorDiffs(self.line_coef[:,:self.app_n_comp])
    
    
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        # plane_coef[0] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[0].data, size=(res_target[1], res_target[0]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[0] = torch.nn.Parameter(
        #     F.interpolate(line_coef[0].data, size=(res_target[2], 1), mode='bilinear', align_corners=True))
        # plane_coef[1] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[1].data, size=(res_target[2], res_target[0]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[1] = torch.nn.Parameter(
        #     F.interpolate(line_coef[1].data, size=(res_target[1], 1), mode='bilinear', align_corners=True))
        # plane_coef[2] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[2].data, size=(res_target[2], res_target[1]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[2] = torch.nn.Parameter(
        #     F.interpolate(line_coef[2].data, size=(res_target[0], 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        # self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        # self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        scale = res_target[0]/self.line_coef.shape[2] #assuming xyz have the same scale
        plane_coef = F.interpolate(self.plane_coef.detach().data, scale_factor=scale, mode='bilinear',align_corners=True)
        line_coef  = F.interpolate(self.line_coef.detach().data, size=(res_target[0],1), mode='bilinear',align_corners=True)
        self.plane_coef, self.line_coef = torch.nn.Parameter(plane_coef), torch.nn.Parameter(line_coef)
        self.compute_stepSize(res_target)
        print(f'upsamping to {res_target}')


class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)
        self.max_coords = torch.max(self.aabb[0], self.aabb[1])
        self.min_coords = torch.min(self.aabb[0], self.aabb[1])
        self.x_coords = torch.linspace(self.min_coords[0], self.max_coords[0], 800, device=device)
        self.y_coords = torch.linspace(self.min_coords[1], self.max_coords[1], 800, device=device)
        self.z_coords = torch.linspace(self.min_coords[2], self.max_coords[2], 800, device=device)
        self.coords = torch.stack(torch.meshgrid([
            self.x_coords, self.y_coords, self.z_coords
        ]), dim=-1).to(device)
        self.normalized_coors = torch.stack(torch.meshgrid([
            torch.linspace(-1, 1, 800, device=device),
            torch.linspace(-1, 1, 800, device=device),
            torch.linspace(-1, 1, 800, device=device)
        ]), dim=-1).to(device)
    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    
    

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}, {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz}, {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars


    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 #+ reg(self.density_line[idx]) * 1e-3
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 #+ reg(self.app_line[idx]) * 1e-3
        return total

    def compute_densityfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature


    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat((plane_coef_point * line_coef_point).T)


    def distill_density_total(self, idx, triplanes, H, W, N_S, n_c):
        # i = idx // (H * H)
        # j = (idx % (H * H)) // H 
        # k = idx % H
        i = (idx % (H*H)) // H
        j = (idx % (H*H)) % H
        k = idx // (H*H)
        
        # Compute density prediction using multiplication across feature dimension
        # This is the key operation that ensures gradients flow to all triplanes
        sigma_chunk = torch.sum(triplanes[0][i,:N_S] * triplanes[1][j,:N_S] * triplanes[2][k,:N_S], dim=-1)  # [batch_size]

        # import ipdb; ipdb.set_trace()
        # Compute partial outer product for this chunk
        # sigma_chunk = torch.sum(x_image[i,:N_S]* y_image[j,:N_S]* z_image[k,:N_S], dim=-1)
        with torch.no_grad():
            sigma_gt = self.compute_densityfeature(self.normalized_coors[i,j,k].reshape(-1,3))
        loss_func = torch.nn.MSELoss()
        # import ipdb; ipdb.set_trace()
        loss_sigma = loss_func(sigma_chunk, sigma_gt)
        return loss_sigma
    
    def distill_color_total(self, idx, triplanes, rays_o_unique,H, W, N_S, n_c):
        batch_size = idx.shape[0]
        ray_size = rays_o_unique.shape[0]
        i = (idx % (H*H)) // H
        j = (idx % (H*H)) % H
        k = idx // (H*H)
        N = (W-N_S)//n_c

        # x_image, y_image, z_image = triplanes[0], triplanes[1], triplanes[2]
        x_feats = triplanes[0][i]  # Shape: [batch_size, N_S+n_c*3]
        y_feats = triplanes[1][j]  # Shape: [batch_size, N_S+n_c*3]
        z_feats = triplanes[2][k]  # Shape: [batch_size, N_S+n_c*3]
        
        # Reshape inputs for batch processing
        x_sigma = x_feats[:,N_S:].reshape(batch_size, n_c, N) # [B, n_c, N]
        y_sigma = y_feats[:,N_S:].reshape(batch_size, n_c, N) # [B, n_c, N]
        z_sigma = z_feats[:,N_S:].reshape(batch_size, n_c, N) # [B, n_c, N]
        
        # Compute product across N dimension for all components at once
        color_coef = torch.einsum('bcn,bcn,bcn->bc', x_sigma, y_sigma, z_sigma) # [B, n_c]
        color_coef = color_coef.reshape(batch_size, 3, 9) # [B, 3, 9]

        # color_coef = torch.zeros((batch_size,n_c), device=x_sigma.device)
        # for p in range(n_c):
        #     color_coef[:,p] = torch.sum(x_sigma[i,p*N:(p+1)*N]*y_sigma[j,p*N:(p+1)*N]*z_sigma[k,p*N:(p+1)*N], dim=1) # [B,27]
        # color_coef = color_coef.reshape(batch_size,3,9) # [B,3,9]        # Subtract each coordinate from all rays to get directions
        coords_expanded = self.normalized_coors[i,j,k].reshape(-1,1,3)  # [B,1,3]
        rays_expanded = rays_o_unique.reshape(1,-1,3)  # [1,R,3]
        dirs = coords_expanded - rays_expanded  # [B,R,3]
        
        dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
        base_dir = eval_sh_bases(2, dirs.reshape(-1,3)).reshape(batch_size,ray_size,9)  # [B,R,9]

        # Apply color coefficients
        color_chunk = torch.matmul(base_dir, color_coef.permute(0, 2, 1)) # [B,R,3]
        color_chunk = torch.sigmoid(color_chunk)
        # Compute partial outer product for this chunk
        # sigma_chunk = torch.sum(x_image[i,:N_S]* y_image[j,:N_S]* z_image[k,:N_S], dim=-1)
        coords = coords_expanded.repeat(1,ray_size,1).reshape(-1,3)
        with torch.no_grad():
            app_features = self.compute_appfeature(coords)
            color_gt = self.renderModule(coords, dirs.reshape(-1,3), app_features).reshape(batch_size,ray_size,3)
        loss_func = torch.nn.MSELoss()
        loss_sigma = loss_func(color_chunk, color_gt)
        return loss_sigma
    
    def calculate_density_block(self, x_basis, y_basis, z_basis, i, j, k, chunk_size, H):
        # Compute outer products in chunks to reduce memory
        sigma = torch.zeros((H,H,H), device=x_basis.device)

        i_end = min(i + chunk_size, H)
        x_chunk = x_basis[i:i_end,:,:,:]
        
        j_end = min(j + chunk_size, H)
        y_chunk = y_basis[:,j:j_end,:,:]
        
        k_end = min(k + chunk_size, H)
        z_chunk = z_basis[:,:,k:k_end,:]
        # Compute partial outer product for this chunk
        sigma_chunk = torch.sum(x_chunk * y_chunk * z_chunk, dim=-1)
        return sigma_chunk
    
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))


        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def get_local_volume_indices(self, valid_xyz):
        x, y, z = valid_xyz[:, 0], valid_xyz[:, 1], valid_xyz[:, 2]
        # Find the insertion points using searchsorted
        x_indices_right, y_indices_right, z_indices_right = torch.searchsorted(self.x_coords, valid_xyz[:, 0], right=True), torch.searchsorted(self.y_coords, valid_xyz[:, 1], right=True), torch.searchsorted(self.z_coords, valid_xyz[:, 2], right=True)
        x_indices_right, y_indices_right, z_indices_right = torch.clamp(x_indices_right, 0, 799), torch.clamp(y_indices_right, 0, 799), torch.clamp(z_indices_right, 0, 799)
        
        x_indices_left, y_indices_left, z_indices_left = x_indices_right - 1, y_indices_right - 1, z_indices_right - 1
        x_indices_left, y_indices_left, z_indices_left = torch.clamp(x_indices_left, 0, 799), torch.clamp(y_indices_left, 0, 799), torch.clamp(z_indices_left, 0, 799)
        
        # Create local volume indices for all 8 corners of each voxel
        local_volume_indices = torch.stack([
            torch.stack([x_indices_left, y_indices_left, z_indices_left], dim=1),
            torch.stack([x_indices_left, y_indices_left, z_indices_right], dim=1), 
            torch.stack([x_indices_left, y_indices_right, z_indices_left], dim=1),
            torch.stack([x_indices_left, y_indices_right, z_indices_right], dim=1),
            torch.stack([x_indices_right, y_indices_left, z_indices_left], dim=1),
            torch.stack([x_indices_right, y_indices_left, z_indices_right], dim=1),
            torch.stack([x_indices_right, y_indices_right, z_indices_left], dim=1),
            torch.stack([x_indices_right, y_indices_right, z_indices_right], dim=1)
        ], dim=0)
        return local_volume_indices, x_indices_left, y_indices_left, z_indices_left, x_indices_right, y_indices_right, z_indices_right
    
    def trilinear_interp_coords(self, V, x, y, z, x_left, y_left, z_left, x_right, y_right, z_right):
        """
        V: tensor of shape [2, 2, 2] with values at the 8 voxel corners
        (x_left, x_right), (y_left, y_right), (z_left, z_right): voxel bounds
        x, y, z: target coordinate inside the cube
        """
        # Compute relative positions in [0, 1]
        tx = (x - x_left) / (x_right - x_left)
        ty = (y - y_left) / (y_right - y_left)
        tz = (z - z_left) / (z_right - z_left)

        weights = torch.stack([
            (1-tx)*(1-ty)*(1-tz), # w000
            (1-tx)*(1-ty)*tz, # w001
            (1-tx)*ty*(1-tz), # w010
            (1-tx)*ty*tz, # w011
            tx*(1-ty)*(1-tz), # w100
            tx*(1-ty)*tz, # w101
            tx*ty*(1-tz), # w110
            tx*ty*tz # w111
            ], dim=0)

        if V.dim() == 2:
            weights = weights.unsqueeze(-1) # Shape: [8, N, 1]
            interp = (V.reshape(8, -1, 3) * weights).sum(dim=0)
        else:
            interp = (V.reshape(8, -1) * weights).sum(dim=0)
        return interp

    def compute_densityfeature_distill(self, triplanes, valid_xyz, local_volume_indices, N_S):
        x_image, y_image, z_image = triplanes[0][:,:N_S], triplanes[1][:,:N_S], triplanes[2][:,:N_S]
        sigma_chunk = torch.zeros((8*valid_xyz.shape[0],), device=valid_xyz.device)
        local_volume_indices = local_volume_indices.reshape(-1, 3)
        # Vectorize the computation by using matrix multiplication
        # Reshape indices for efficient indexing
        i, j, k = local_volume_indices[:,0], local_volume_indices[:,1], local_volume_indices[:,2]
        
        # Get all features at once using advanced indexing
        x_feats = x_image[i]  # Shape: [batch_size, N_S] 
        y_feats = y_image[j]  # Shape: [batch_size, N_S]
        z_feats = z_image[k]  # Shape: [batch_size, N_S]
        # Compute density prediction using multiplication across feature dimension
        try:
            sigma_chunk = torch.sum(x_feats * y_feats * z_feats, dim=-1)
        except torch.cuda.OutOfMemoryError as e:
            import ipdb; ipdb.set_trace()
        # for idx, (i, j, k) in enumerate(local_volume_indices):
        #     sigma_chunk[idx] = torch.sum(x_image[i]* y_image[j]* z_image[k], dim=-1)

        return sigma_chunk
    
    def forward_distill(self, triplanes, N_S, n_c, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        # import ipdb; ipdb.set_trace()
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
            # xyz_sampled = self.normalize_coord(xyz_sampled)
            valid_xyz =xyz_sampled[ray_valid]

            local_volume_indices, x_indices_left, y_indices_left, z_indices_left, x_indices_right, y_indices_right, z_indices_right = self.get_local_volume_indices(valid_xyz)
            sigma_chunk = self.compute_densityfeature_distill(triplanes, valid_xyz, local_volume_indices, N_S)
            # sigma_chunk = sigma_chunk.reshape(-1, 1)
            sigma_pred = self.trilinear_interp_coords(sigma_chunk, valid_xyz[:, 0], valid_xyz[:, 1], valid_xyz[:, 2], 
                                         self.x_coords[x_indices_left], self.y_coords[y_indices_left], self.z_coords[z_indices_left], 
                                         self.x_coords[x_indices_right], self.y_coords[y_indices_right], self.z_coords[z_indices_right])
            # Get density features for valid points
            # sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid]) # factorize 3d to plane and line
            # Convert features to density values 
            validsigma = self.feature2density(sigma_pred)
            # Assign computed densities back to valid points
            sigma[ray_valid] = validsigma
        # Convert raw density (sigma) to alpha values and compute weights for volume rendering
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        # Only compute colors for points with significant contribution to final image
        app_mask = weight > self.rayMarch_weight_thres
        N = (triplanes[0].shape[1] - N_S) // n_c
        # For points that contribute significantly, compute RGB colors
        if app_mask.any():
            valid_xyz_color = xyz_sampled[app_mask]
            local_volume_indices_color, x_indices_left_color, y_indices_left_color, z_indices_left_color, x_indices_right_color, y_indices_right_color, z_indices_right_color = self.get_local_volume_indices(valid_xyz_color)
            x_image, y_image, z_image = triplanes[0][:,N_S:], triplanes[1][:,N_S:], triplanes[2][:,N_S:]
            color_coef = torch.zeros((8*valid_xyz_color.shape[0], n_c), device=valid_xyz_color.device) # [8*B, 27]
            local_volume_indices_color = local_volume_indices_color.reshape(-1, 3)
            # Vectorize the computation using einsum instead of nested loops
            # Reshape inputs for batch processing
            x_feats = x_image[local_volume_indices_color[:,0]] # [B, N*n_c]
            y_feats = y_image[local_volume_indices_color[:,1]] # [B, N*n_c] 
            z_feats = z_image[local_volume_indices_color[:,2]] # [B, N*n_c]
            
            # Reshape to separate N and n_c dimensions
            x_feats = x_feats.reshape(-1, n_c, N) # [B, n_c, N]
            y_feats = y_feats.reshape(-1, n_c, N) # [B, n_c, N]
            z_feats = z_feats.reshape(-1, n_c, N) # [B, n_c, N]
            
            # Compute product across N dimension for all components at once
            color_coef = torch.einsum('bcn,bcn,bcn->bc', x_feats, y_feats, z_feats) # [B, n_c]
            # for idx, (i, j, k) in enumerate(local_volume_indices_color):
            #     for p in range(n_c):
            #         color_coef[idx, p] = torch.sum(x_image[i,p*N:(p+1)*N]* y_image[j,p*N:(p+1)*N]* z_image[k,p*N:(p+1)*N], dim=-1)
            coords_expanded = self.coords[local_volume_indices_color[:,0], local_volume_indices_color[:,1], local_volume_indices_color[:,2]] #[8*B, 3]
            dirs_expanded = (coords_expanded.reshape(8, -1, 3) - rays_chunk[0, :3].reshape(1,-1,3)).reshape(-1,3) # [8xB, 3] # same camera for the batch
            base_dir = eval_sh_bases(2, dirs_expanded.reshape(-1, 3)) # [8xB, 9]
            app_chunk = torch.einsum('bi,bji->bj', base_dir, color_coef.reshape(-1, 3, 9)) # [8xB, 3]
            valid_rgbs = self.trilinear_interp_coords(app_chunk, valid_xyz_color[:, 0], valid_xyz_color[:, 1], valid_xyz_color[:, 2], 
                                         self.x_coords[x_indices_left_color], self.y_coords[y_indices_left_color], self.z_coords[z_indices_left_color], 
                                         self.x_coords[x_indices_right_color], self.y_coords[y_indices_right_color], self.z_coords[z_indices_right_color])
            
            # # Get appearance features
            # app_features = self.compute_appfeature(xyz_sampled[app_mask])
            # # Render RGB values using the chosen rendering module
            # valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            
            # Assign computed colors back to contributing points
            rgb[app_mask] = valid_rgbs
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

class TensorCP(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorCP, self).__init__(aabb, gridSize, device, **kargs)


    def init_svd_volume(self, res, device):
        self.density_line = self.init_one_svd(self.density_n_comp[0], self.gridSize, 0.2, device)
        self.app_line = self.init_one_svd(self.app_n_comp[0], self.gridSize, 0.2, device)
        self.basis_mat = torch.nn.Linear(self.app_n_comp[0], self.app_dim, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component, gridSize[vec_id], 1))))
        return torch.nn.ParameterList(line_coef).to(device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_densityfeature(self, xyz_sampled):

        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = F.grid_sample(self.density_line[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[1], coordinate_line[[1]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[2], coordinate_line[[2]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        sigma_feature = torch.sum(line_coef_point, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):

        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = F.grid_sample(self.app_line[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[1], coordinate_line[[1]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[2], coordinate_line[[2]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])

        return self.basis_mat(line_coef_point.T)
    

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
            app_line_coef[i] = torch.nn.Parameter(
                F.interpolate(app_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return density_line_coef, app_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_line, self.app_line = self.up_sampling_Vector(self.density_line, self.app_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)


        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx]) * 1e-3
        return total