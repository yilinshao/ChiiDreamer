import os
from dataclasses import dataclass
import numpy as np
import threestudio
import torch
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.utils import parse_optimizer
from threestudio.utils.loss import tv_loss
from threestudio.utils.typing import *
import math
import torch.nn.functional as F
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from torch_kdtree import build_kd_tree

from ..utils.np_utils.train import Runner
from ..geometry.gaussian_base import BasicPointCloud
from pytorch3d.ops import knn_points
from ..utils.sugar_utils.spherical_harmonics import SH2RGB

from ..sugar_scene.sugar_model import SuGaR
from ..sugar_scene.gs_model import PseudoGaussianSplattingWrapper
from ..sugar_scene.sugar_optimizer import OptimizationParams, SuGaROptimizer
from ..sugar_scene.sugar_densifier import SuGaRDensifier

# from ..geometry.gaussian_base import BasicPointCloud

'''
Here we assume the given condition are 2D layout boxes, and given in the CONFIG files. The layout is loaded in the initialization process.
Also we here apply the transformation for initialized 3D layout in this file for better explanability.
After initialization, the initialized 3D boxes are then transformed into optimizable parameters that controls the relative positions of given box.
NOTE@: Now we only consider depth and rotation as differentiable parameters that could be optimized.
TODO@: Implementing loading Layout 2D boxes and then transform into 3D boxes.
        CUSTOMIZED PARAMETERS:
        self.geometry.layout_2d: numpy->ndarray, the user given 2D layout boxes.
        self.geometry.box_configs: initialized 3D layout box.
        self.geometry.scaling_factors: for rescale
        self.geometry.mark_instances: for marking how many gaussians are there representing each instance.
HERE WE MENTION THAT: at the end of each iteration, we should update the given xyz.
'''


@threestudio.register("gaussian-splatting-chiidreamer-system")
class LayoutCHIGSSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        gaussian_checkpoint: str = None
        start_sdf_regularization_from: int = 9000
        reset_neighbors_every: int = 1000
        reset_dataset_every: int = 1000
        start_reset_neighbors_from: int = 7000
        loss_sdf_normal_start: int = 1000

        total_iter: int = 18000
        period_iter: int = 3000
        sdf_start: int = 1200
        gs_pull_start: int = 1500
        normal_start: int = 1700
        sdf_end: int = 2000
        sds_start: int = 2000

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()

        self.automatic_optimization = False
        # self.lpips = LPIPS(net='vgg').to(self.device)
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # split the prompt into list
        split_prompt = False
        prompt_str = self.cfg.prompt_processor.prompt
        if '.' in prompt_str:
            prompt_list = [p.strip() for p in prompt_str.split('.')]
            split_prompt = True

        self.prompt_processor_list = []
        self.prompt_utils_list = []
        if split_prompt:
            for prompt_item in prompt_list:
                self.cfg.prompt_processor['prompt'] = prompt_item
                prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
                prompt_utils = prompt_processor()

                self.prompt_processor_list.append(prompt_processor)
                self.prompt_utils_list.append(prompt_utils)
        else:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            self.prompt_utils = self.prompt_processor()

        self.obj_name_list = ['clock', 'hourglass']

        # init chiidreamer, first, load from gaussian checkpoint
        assert self.cfg.gaussian_checkpoint is not None
        self.load_gaussian_checkpoint(self.cfg.gaussian_checkpoint)

        # get random number generation and recover it after initialization, to maintain dataset loading order
        rng_state = torch.get_rng_state()

        # get position and color from trained 3DGS
        with torch.no_grad():
            print("Initializing model from trained 3DGS...")
            sh_levels = int(np.sqrt(self.geometry.get_features.shape[1]))
            points = self.geometry.get_xyz.detach().float().cuda()
            colors = SH2RGB(self.geometry.get_features[:, 0].detach().float().cuda())
        print(f"Point cloud generated. Number of points: {len(points)}")

        # param setting
        learnable_positions = True
        triangle_scale = 1.
        regularity_knn = 16  # 8 until now
        beta_mode = None
        freeze_gaussians = False
        o3d_mesh = None
        learn_surface_mesh_positions = False
        learn_surface_mesh_opacity = False
        learn_surface_mesh_scales = False
        n_gaussians_per_surface_triangle = 1
        nerfmodel = PseudoGaussianSplattingWrapper(
            device=self.geometry.device,
            height=512,
            width=512,
            cam_radius=1.732,
            gaussian_model=self.geometry)

        # init SuGaR model
        self.sugar = SuGaR(
            nerfmodel=nerfmodel,
            points=points,
            colors=colors,
            initialize=True,
            sh_levels=sh_levels,
            learnable_positions=learnable_positions,
            triangle_scale=triangle_scale,
            keep_track_of_knn=True,
            knn_to_track=regularity_knn,
            beta_mode=beta_mode,
            freeze_gaussians=freeze_gaussians,
            surface_mesh_to_bind=o3d_mesh,
            surface_mesh_thickness=None,
            learn_surface_mesh_positions=learn_surface_mesh_positions,
            learn_surface_mesh_opacity=learn_surface_mesh_opacity,
            learn_surface_mesh_scales=learn_surface_mesh_scales,
            n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
        )

        # init the rest 3DGS param
        with torch.no_grad():
            print("Initializing 3D gaussians from pre-trained 3D gaussians...")
            self.sugar._scales[...] = self.geometry._scaling.detach()
            self.sugar._quaternions[...] = self.geometry._rotation.detach()
            self.sugar.all_densities[...] = self.geometry._opacity.detach()
            self.sugar._sh_coordinates_dc[...] = self.geometry._features_dc.detach()
            self.sugar._sh_coordinates_rest[...] = self.geometry._features_rest.detach()

        # init SDF
        self.sugar.part_num = 1
        sugar_checkpoint_path = '.'  # fixme: redundant, checkpoint will not be set here
        self.sugar.neus_0 = Runner(sugar_checkpoint_path, None, part_num=self.sugar.part_num)
        # TODO: for objects_i in range(len(self.prompt_processor_list) - 2):
        self.sugar.neus_1 = Runner(sugar_checkpoint_path, None, part_num=self.sugar.part_num)

        print(f'\nSuGaR model has been initialized.')
        print(f'Number of parameters: {sum(p.numel() for p in self.sugar.parameters() if p.requires_grad)}')
        print(f'Checkpoints will be saved in {sugar_checkpoint_path}')
        print("\nModel parameters:")
        for name, param in self.sugar.named_parameters():
            print(name, param.shape, param.requires_grad)

        cameras_spatial_extent = self.sugar.get_cameras_spatial_extent()

        # ====================Initialize optimizer====================
        spatial_lr_scale = cameras_spatial_extent
        print("Using camera spatial extent as spatial_lr_scale:", spatial_lr_scale)
        position_lr_init = 0.00016
        position_lr_final = 0.0000016
        position_lr_delay_mult = 0.01
        position_lr_max_steps = 10000
        feature_lr = 0.0005
        opacity_lr = 0.05
        scaling_lr = 0.005
        rotation_lr = 0.001

        opt_params = OptimizationParams(
            position_lr_init=position_lr_init,
            position_lr_final=position_lr_final,
            position_lr_delay_mult=position_lr_delay_mult,
            position_lr_max_steps=position_lr_max_steps,
            feature_lr=feature_lr,
            opacity_lr=opacity_lr,
            scaling_lr=scaling_lr,
            rotation_lr=rotation_lr,
        )
        self.sugar_optimizer = SuGaROptimizer(self.sugar, opt_params, spatial_lr_scale=spatial_lr_scale)

        print("Optimizer initialized.")
        print("Optimizable parameters:")
        for param_group in self.sugar_optimizer.optimizer.param_groups:
            print(param_group['name'], param_group['lr'])

        # ====================Initialize densifier====================
        # do not use the densifier
        densify_grad_threshold = 0.0001  # 0.0002
        densify_screen_size_threshold = 20
        prune_opacity_threshold = 0.005
        densification_percent_distinction = 0.01
        self.gaussian_densifier = SuGaRDensifier(
            sugar_model=self.sugar,
            sugar_optimizer=self.sugar_optimizer,
            max_grad=densify_grad_threshold,
            min_opacity=prune_opacity_threshold,
            max_screen_size=densify_screen_size_threshold,
            scene_extent=cameras_spatial_extent,
            percent_dense=densification_percent_distinction,
        )
        print("Densifier initialized.")

        self.last_resample_iteration = self.last_reset_iteration = -1
        self.start_sdf_regularization_from = self.cfg.start_sdf_regularization_from

        torch.set_rng_state(rng_state)


    def calculate_collision(self, pc1, pc2):
        """
        point cloud distance
        theta: float, threshold for distinguishing the loss.
        """
        # KDTree cuda version
        tree1 = build_kd_tree(pc1)
        tree2 = build_kd_tree(pc2)

        # Check the nearest neighbour(1)
        distances1, ind1 = tree1.query(pc2, nr_nns_searches=1)
        print(distances1)
        distances2, ind2 = tree2.query(pc1, nr_nns_searches=1)

        return distances1, distances2, ind1, ind2

    def get_frontal_camera_batch(self):
        # customization of different parameters.
        batch_size = 1
        height: float = 224
        light_distance: float = 1.0
        # customization of different parameters.
        # By Colez.
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        elevation = torch.zeros(1).to('cuda')
        elevation_deg = elevation * math.pi / 180

        azimuth_deg: Float[Tensor, "B"]
        azimuth_deg = torch.zeros(1).to('cuda')
        azimuth = azimuth_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (torch.tensor([1.5]).to('cuda'))

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1, )

        # Not changed.
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions).to('cuda')
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                                   None, :
                                   ].repeat(batch_size, 1).to('cuda')

        # fixed fovy
        fovy_deg: Float[Tensor, "B"] = (torch.tensor([49.1]).to('cuda'))
        fovy = fovy_deg * math.pi / 180

        # here make this function static
        light_distances: Float[Tensor, "B"] = (
            torch.tensor([light_distance]).to('cuda')
        )

        # do not apply light perturb
        light_direction: Float[Tensor, "B 3"] = F.normalize(camera_positions, dim=-1)
        # get light position by scaling light direction by light distance
        light_positions: Float[Tensor, "B 3"] = (
            light_direction * light_distances[:, None]
        )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1).to('cuda')
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length

        directions_unit_focal = get_ray_directions(H=height, W=height, focal=1.0)

        focal_length: Float[Tensor, "B"] = 0.5 * height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[None, :, :, :].repeat(batch_size, 1, 1, 1).to(
            'cuda')
        directions[:, :, :, :2] = (directions[:, :, :, :2] / focal_length[:, None, None, None])

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(
            directions, c2w, keepdim=True, normalize=False
        )

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, height / height, 0.01, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx.to('cuda'))

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": height,
            "width": height,
            "fovy": fovy,
            "proj_mtx": proj_mtx,
        }

    def configure_optimizers(self):
        sugar_optim = self.sugar_optimizer.get_optimizer
        geometry_optim = self.geometry.optimizer
        if hasattr(self, "merged_optimizer"):
            return [sugar_optim, geometry_optim]
        else:
            if hasattr(self.cfg.optimizer, "name"):
                # if certain param (except the 3DGS) are required to optimize, then put them in the net optim, then merge
                net_optim = parse_optimizer(self.cfg.optimizer, self)
                geometry_optim = self.geometry.merge_optimizer(net_optim)
                self.merged_optimizer = True
            else:
                self.merged_optimizer = False
            return [sugar_optim, geometry_optim]

    def on_train_batch_start(self, batch, batch_idx):
        # Update learning rates
        udfnet_lr = self.sugar.neus_0.get_learning_rate_at_iteration(self.global_step, max_iter=20000)
        self.sugar_optimizer.update_learning_rate(self.global_step, sdfnet_lr=udfnet_lr)

    def forward(self, batch: Dict[str, Any], instance_id=-1, val_global_step=None) -> Dict[str, Any]:
        if val_global_step is None:
            global_step = self.global_step
        else:
            global_step = val_global_step
        if global_step == 0 or (global_step % self.cfg.period_iter) in range(self.cfg.sdf_start, self.cfg.sdf_end):
            # rasterizer setting
            current_sh_levels = sh_levels = 4
            compute_color_in_rasterizer = False
            use_same_scale_in_all_directions = False  # Should be False

            enforce_entropy_regularization = True
            white_bg = True

            # render the image
            # TODO: delete useless param in rasterizer
            outputs = self.sugar.render_image_gaussian_rasterizer(
                batch,
                verbose=False,
                comp_rgb_bg=self.background,
                material=self.material,
                bg_color=torch.Tensor([1.0, 1.0, 1.0]).to(self.sugar.device) if white_bg else None,
                sh_deg=current_sh_levels - 1,
                sh_rotations=None,
                compute_color_in_rasterizer=compute_color_in_rasterizer,
                compute_covariance_in_rasterizer=True,
                return_2d_radii=True,
                quaternions=None,
                use_same_scale_in_all_directions=use_same_scale_in_all_directions,
                return_opacities=enforce_entropy_regularization,
                use_pulled=False,
            )
            return outputs
        else:
            self.geometry.update_learning_rate(self.global_step)
            outputs = self.renderer.batch_forward(batch, instance_id)
            return outputs


    def training_step(self, batch, batch_idx):
        # ssyl===============select the gaussians for each instance=====>
        if len(self.prompt_utils_list) > 0:
            rand_num = np.random.rand()
            # rand_num = 0.7
            if rand_num < 0.33:
                instance_id = 0
                self.geometry.selected_instance_id = 0
                prompt_utils = self.prompt_utils_list[0]
            elif rand_num < 0.66:
                instance_id = 1
                self.geometry.selected_instance_id = 1
                prompt_utils = self.prompt_utils_list[1]
            else:
                instance_id = -1
                self.geometry.selected_instance_id = -1
                prompt_utils = self.prompt_utils_list[2]
        else:
            prompt_utils = self.prompt_utils

        # syl ==========================在只有instance渲染的过程中,挑一部分全部渲染,增强相互之间的语义
        if instance_id != -1:
            rand_num = np.random.rand()
            # rand_num = 0.1
            if rand_num > 0.8:
                render_instance_id = -1
            else:
                render_instance_id = instance_id
        else:
            render_instance_id = instance_id
        # syl ==========================只有instance渲染的过程中,挑一部分全部渲染,增强相互之间的语义

        loss_sdf = 0.0
        loss_sds = 0.0
        loss_gs = 0.0
        # step: scaling loss
        # scaling_loss = torch.abs(sugar.scaling.min(1)[0] - 1e-7).mean()
        clamped_scaling = torch.clamp(self.sugar.scaling.min(1)[0], min=1e-4)
        scaling_loss = torch.abs(clamped_scaling - 1e-4).mean()
        scaling_loss = 100 * scaling_loss
        self.log('train/scaling_loss', scaling_loss, on_step=True, on_epoch=True, prog_bar=True)

        # step: sdf loss

        # if iteration > 7000
        prune_hard_opacity_threshold = -0.1
        sugar_checkpoint_path = os.path.join(os.path.dirname(self.get_save_dir()), 'sugar')

        mark_accum = torch.cumsum(self.geometry.mark_instances, dim=0)
        obj_mask = []
        obj_1_mask = torch.zeros(self.sugar.points.shape[0])
        obj_2_mask = torch.zeros(self.sugar.points.shape[0])
        obj_1_mask[0: mark_accum[0]] = 1
        obj_mask.append(obj_1_mask.to(torch.bool))
        obj_2_mask[mark_accum[0]: mark_accum[1]] = 1
        obj_mask.append(obj_2_mask.to(torch.bool))

        # step: setup dataset, prune the gaussian before sdf loss
        if self.global_step % self.cfg.period_iter == 0:
            # TODO: not pruned
            print("Prunning Pointcloud Using Opacity...")
            prune_mask = (self.gaussian_densifier.model.strengths < prune_hard_opacity_threshold).squeeze()
            self.gaussian_densifier.prune_points(prune_mask)
            print('After Prunning: {} Gaussians Left.'.format(self.sugar.points.shape[0]))

            self.sugar.visual_point_cloud(iteration=0, checkpoint_path=sugar_checkpoint_path)

            # setup dataset
            loaded_gs_workdir = os.path.normpath(os.path.join(self.cfg.gaussian_checkpoint, '..', '..'))
            print(f"The initial SDF Datasets are stored in the loading folder: {loaded_gs_workdir}, "
                  f"\n not the working dir: {self.get_save_dir()}")

            save_dataset = True if self.global_step == 0 else False
            obj_0_points = self.sugar.points[0: mark_accum[0]].detach().cpu().numpy()
            self.sugar.neus_0.reset_datasets(
                os.path.join(loaded_gs_workdir, 'sugar'),
                obj_0_points,
                self.obj_name_list[0], iteration=self.global_step, scene_name='threestudio', save_dataset=save_dataset)

            obj_1_points = self.sugar.points[mark_accum[0]: mark_accum[1]].detach().cpu().numpy()
            self.sugar.neus_1.reset_datasets(
                os.path.join(loaded_gs_workdir, 'sugar'),
                obj_1_points,
                self.obj_name_list[1], iteration=self.global_step, scene_name='threestudio', save_dataset=save_dataset)


        # step: start sdf regularization
        if self.global_step % self.cfg.period_iter in range(self.cfg.sdf_start, self.cfg.sdf_end):
            opt = self.optimizers()[0]
            loss_sdf = loss_sdf + scaling_loss
            # # reset neighbors at start and at every interval
            # if ((self.global_step >= self.cfg.start_reset_neighbors_from) and
            #     ((self.global_step == self.start_sdf_regularization_from + 1) or
            #      (self.global_step % self.cfg.reset_neighbors_every == 0)) and self.global_step != self.last_reset_iteration):
            #     print("\n---INFO---\nResetting neighbors...")
            #     self.sugar.reset_neighbors()
            #     self.last_reset_iteration = self.global_step

            # train sdf
            cur_part_num = 1
            for i in range(2):
                neus_model = getattr(self.sugar, f'neus_{i}')
                dataset = getattr(neus_model, 'dataset' + str(cur_part_num))
                points, samples, point_gt, points_idx = dataset.get_train_data(10000)
                samples.requires_grad = True

                # sdf loss 1
                sdf_network = getattr(neus_model, 'sdf_network' + str(cur_part_num))
                gradients_sample = sdf_network.gradient(samples).squeeze()  # 5000x3
                udf_sample = sdf_network.sdf(samples)  # 5000x1
                grad_norm = F.normalize(gradients_sample, dim=1)  # 5000x3
                sample_moved = samples - grad_norm * udf_sample  # 5000x3
                # update the sdf network not the gs points
                ChamferDis = getattr(neus_model, 'ChamferDisL1')
                sdf_loss1 = ChamferDis(points.unsqueeze(0), sample_moved.unsqueeze(0))

                # sdf loss 2 = loss pull in the paper
                scaled_sample_moved = sample_moved * dataset.shape_scale + dataset.shape_center
                knn = knn_points(sample_moved[None], points[None], K=1)
                knn_idx = knn.idx[0, :, 0]
                # gaussian_inv_scaled_rotation = sugar.get_covariance(
                #     return_full_matrix=True, return_sqrt=True, inverse_scales=True, scaling_factor=-1, enlarge_minaxis=-1)
                gaussian_inv_scaled_rotation = self.sugar.get_covariance(
                    return_full_matrix=True, return_sqrt=True, inverse_scales=True, scaling_factor=100,
                    enlarge_minaxis=100)[obj_mask[i]]
                sugar_points_idx = torch.arange(obj_mask[i].sum(), device='cuda')[dataset.part_select_idx][dataset.downsample_idx]
                batch_selected_idx = sugar_points_idx[points_idx][knn_idx]
                closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[batch_selected_idx].detach().clone()
                surf_points = points[knn_idx].detach().clone() * dataset.shape_scale + dataset.shape_center

                shift = (
                        scaled_sample_moved - surf_points)  # NOTE: scaled_sample_moved = q, surf_points = /mu_j (no gradient, updating the sdf)
                warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
                neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
                neighbor_opacities = torch.exp(-1. / 2 * neighbor_opacities)
                sdf_loss2 = torch.abs(1 - neighbor_opacities)[neighbor_opacities > 0.9].mean()  # NOTE: loss pull
                # sdf_loss1 ~= 6e-3, sdf_loss2 ~= 1e-6
                loss_sdf = loss_sdf + 1.0 * sdf_loss1 + 100.0 * sdf_loss2
                self.log('train/sdf_loss', 1.0 * sdf_loss1, on_step=True, on_epoch=True, prog_bar=True)
                self.log('train/cov_loss', 100.0 * sdf_loss2, on_step=True, on_epoch=True, prog_bar=True)

                eikonal_loss = ((grad_norm.norm(dim=-1) - 1.0) ** 2).mean()
                loss_sdf = loss_sdf + eikonal_loss
                self.log('train/eikonal_loss', eikonal_loss, on_step=True, on_epoch=True, prog_bar=True)

                # ours: pull gs
                if self.global_step % self.cfg.period_iter >= self.cfg.gs_pull_start:  # delay for very large scene

                    # reset dataset every 500 iter
                    if self.global_step % self.cfg.reset_dataset_every == 0:
                        print('\n Recalculating Sample Points...')
                        obj_0_points = self.sugar.points[obj_mask[0]].detach().cpu().numpy()
                        self.sugar.neus_0.reset_datasets(
                            sugar_checkpoint_path,
                            obj_0_points,
                            self.obj_name_list[0], iteration=self.global_step, scene_name='threestudio',
                            save_dataset=False)

                        obj_1_points = self.sugar.points[obj_mask[1]].detach().cpu().numpy()
                        self.sugar.neus_1.reset_datasets(
                            sugar_checkpoint_path,
                            obj_1_points,
                            self.obj_name_list[1], iteration=self.global_step, scene_name='threestudio',
                            save_dataset=False)
                        self.last_resample_iteration = self.global_step

                    rescaled_sugar_points = (surf_points - dataset.shape_center) / dataset.shape_scale
                    _gradients_sample = sdf_network.gradient(rescaled_sugar_points).squeeze()
                    _udf_sample = sdf_network.sdf(rescaled_sugar_points)
                    _grad_norm = F.normalize(_gradients_sample, dim=1)  #
                    rescaled_sugar_points_moved = rescaled_sugar_points - _grad_norm * _udf_sample
                    sugar_points_moved = rescaled_sugar_points_moved * dataset.shape_scale + dataset.shape_center
                    sugar_points_diff = torch.norm(self.sugar.points[obj_mask[i]][batch_selected_idx] - sugar_points_moved.detach(), p=2,
                                                   dim=-1).mean()

                    # sugar_points_diff ~= 8e-4
                    loss_sdf = loss_sdf + 10.0 * sugar_points_diff  # NOTE: update the gs, pull them closer to sdf surface
                    self.log('train/points_diff', 10.0 * sugar_points_diff, on_step=True, on_epoch=True, prog_bar=True)

                # norm consistency
                if self.global_step % self.cfg.period_iter >= self.cfg.normal_start:   # delay for very large scene
                    sugar_normals = self.sugar.get_normals()[obj_mask[i]][batch_selected_idx]
                    surf_normals = _grad_norm.detach()
                    sugar_normal_loss = torch.abs(torch.sum(surf_normals * sugar_normals, -1).abs() - 1).mean()  # NOTE: update gs normal

                    if self.global_step > -1:   # TODO
                        gaussian_center_normals = sugar_normals.detach()
                        query_normal_loss = torch.abs(torch.sum(grad_norm * gaussian_center_normals, -1).abs() - 1).mean()  # NOTE: update sdf normal
                    else:
                        query_normal_loss = 0.

                    # sugar normal loss and query normal loss ~= 0.2
                    normal_loss = 0.1 * sugar_normal_loss + 0.01 * query_normal_loss
                    loss_sdf = loss_sdf + normal_loss

                    self.log('train/sugar_normal_loss', 0.1 * sugar_normal_loss, on_step=True, on_epoch=True, prog_bar=True)
                    self.log('train/query_normal_loss', 0.01 * query_normal_loss, on_step=True, on_epoch=True, prog_bar=True)

                # ours elastic potential energy
                # with torch.no_grad()?
                if self.global_step % self.cfg.period_iter >= 20:
                    # cp=complementary, 循环所有作用在物体 i 上面的力
                    for cp_i in range(len(obj_mask)):
                        if cp_i == i:
                            continue
                        cp_neus_model = getattr(self.sugar, f'neus_{cp_i}')
                        cp_dataset = getattr(cp_neus_model, 'dataset' + str(cur_part_num))
                        cp_sdf_network = getattr(cp_neus_model, 'sdf_network' + str(cur_part_num))

                        my_points = self.sugar.points[obj_mask[i]][sugar_points_idx].detach().cpu().numpy()
                        cp_selected_bool = np.all((my_points > cp_dataset.block_min) & (my_points < cp_dataset.block_max), axis=1)
                        cp_selected_idx = sugar_points_idx[cp_selected_bool]
                        my_points_in_cp_box = my_points[cp_selected_bool]

                        my_points_in_cp_box = my_points_in_cp_box - cp_dataset.shape_center.detach().cpu().numpy()
                        my_points_in_cp_box = my_points_in_cp_box / cp_dataset.shape_scale.detach().cpu().numpy()

                        my_points_sdf = cp_sdf_network.sdf(torch.from_numpy(my_points_in_cp_box).to(self.device).float())
                        collision_bool = my_points_sdf.squeeze(1).detach().cpu().numpy() < 0.02
                        collision_idx = cp_selected_idx[collision_bool]
                        my_collision_pts = my_points_in_cp_box[collision_bool]

                        print(f'Total number Obj{i} is {self.sugar.points[obj_mask[i]].shape[0]}, \
                        has {my_collision_pts.shape[0]} collision points.')

                        if self.global_step % 50 == 0:
                            # save point cloud
                            import trimesh
                            xyz_show = my_collision_pts
                            output_path = os.path.join(self.get_save_dir(), 'pointcloud')
                            os.makedirs(output_path, exist_ok=True)
                            trimesh.Trimesh(xyz_show).export(os.path.join(output_path, f'collision_points_of_obj{i}_' + str(self.global_step) + '.ply'))
                            print('Visualize Points OK.')

                        # calculate the energy
                        collision_gs_normal = self.sugar.get_normals()[obj_mask[i]][collision_idx]
                        sum_normal = collision_gs_normal.sum(dim=0)



            loss_sdf.backward()
            opt.step()

        else:  # train_sds_loss
            opt = self.optimizers()[1]
            loss_sds = loss_sds + scaling_loss
            if self.global_step % self.cfg.period_iter == self.cfg.sds_start:
                # from sugar.points to system.xyz
                with torch.no_grad():
                    self.geometry._xyz[...] = self.sugar._points.detach()  # NOTE: 这一步不会改变xyz的requires_grad等,只拷贝data
                    self.geometry._scaling[...] = self.sugar._scales.detach()
                    self.geometry._rotation[...] = self.sugar._quaternions.detach()
                    self.geometry._opacity[...] = self.sugar.all_densities.detach()
                    self.geometry._features_dc[...] = self.sugar._sh_coordinates_dc.detach()
                    self.geometry._features_rest[...] = self.sugar._sh_coordinates_rest.detach()

            if self.global_step % self.cfg.period_iter == self.cfg.period_iter - 1:
                # from system.xyz to sugar.points
                with torch.no_grad():
                    print("Initializing 3D gaussians from pre-trained 3D gaussians...")
                    self.sugar._points[...] = self.geometry._xyz.detach()
                    self.sugar._scales[...] = self.geometry._scaling.detach()
                    self.sugar._quaternions[...] = self.geometry._rotation.detach()
                    self.sugar.all_densities[...] = self.geometry._opacity.detach()
                    self.sugar._sh_coordinates_dc[...] = self.geometry._features_dc.detach()
                    self.sugar._sh_coordinates_rest[...] = self.geometry._features_rest.detach()

            out = self(batch, render_instance_id)  # outputs.
            guidance_inp = out["comp_rgb"]

            guidance_out = self.guidance(
                guidance_inp,
                prompt_utils,
                **batch,
                rgb_as_latents=False
            )

            self.log("gauss_num", int(self.geometry._xyz.shape[0]), on_step=True, on_epoch=True, prog_bar=True, logger=True)

            # sds loss
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss_sds += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

            visibility_filter = out["visibility_filter"]
            radii = out["radii"]
            viewspace_point_tensor = out["viewspace_points"]

            #  find instance filter
            # TODO: this should ALSO work in the train gaussian loss
            instance_filter = torch.zeros(self.geometry._xyz.shape[0], device=self.geometry._xyz.device)
            mark_accum = torch.cumsum(self.geometry.mark_instances, dim=0)
            if instance_id == 0:
                instance_filter[0: mark_accum[instance_id]] = 1
            elif instance_id == 1:
                instance_filter[mark_accum[instance_id - 1]: mark_accum[instance_id]] = 1
            else:
                instance_filter[:] = 1

            for i, view_filter in enumerate(visibility_filter):  # MV dream has 4 views (seems they are all the same)
                visibility_filter[i] = view_filter & (instance_filter != 0)

            loss_sds.backward(retain_graph=True)
            iteration = self.global_step

            self.geometry.update_states(
                iteration,
                visibility_filter,
                radii,
                viewspace_point_tensor,
            )

            # step: 清除其他inst的梯度 TODO: ALSO work in the train gaussian loss
            if instance_id != -1:
                mark_accum = torch.cumsum(self.geometry.mark_instances, dim=0).cpu()
                if instance_id == 0:
                    instance_range = np.arange(0, mark_accum[instance_id])
                else:
                    instance_range = np.arange(mark_accum[instance_id - 1], mark_accum[instance_id])
                zero_grad_inst(self.geometry, instance_range)

            opt.step()

        opt.zero_grad(set_to_none=True)
        train_gaussian_loss = False
        # gaussian related losses
        if train_gaussian_loss:
            loss_gs = 0.0
            if self.cfg.loss["lambda_position"] > 0.0:
                xyz_mean = self.sugar.points.norm(dim=-1)
                loss_position = xyz_mean.mean()
                self.log(f"train/loss_position", loss_position)
                loss_gs += self.C(self.cfg.loss["lambda_position"]) * loss_position

            if self.cfg.loss["lambda_opacity"] > 0.0:
                scaling = self.sugar.scaling.norm(dim=-1)
                loss_opacity = (
                    scaling.detach().unsqueeze(-1) * self.geometry.get_opacity
                ).sum()
                self.log(f"train/loss_opacity", loss_opacity)
                loss_gs += self.C(self.cfg.loss["lambda_opacity"]) * loss_opacity

            if self.cfg.loss["lambda_sparsity"] > 0.0:
                loss_sparsity = -(self.sugar.strengths - 0.5).pow(2).mean()
                self.log("train/loss_sparsity", loss_sparsity)
                loss_gs += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            if self.cfg.loss["lambda_scales"] > 0.0:
                scale_sum = torch.sum(self.sugar.scaling)
                self.log(f"train/scales", scale_sum)
                loss_gs += self.C(self.cfg.loss["lambda_scales"]) * scale_sum

            if self.cfg.loss["lambda_tv_loss"] > 0.0:
                # assert train_sds_loss
                loss_tv = self.C(self.cfg.loss["lambda_tv_loss"]) * tv_loss(
                    out["comp_rgb"].permute(0, 3, 1, 2)
                )
                self.log(f"train/loss_tv", loss_tv)
                loss_gs += loss_tv

            # =========== Normal losses =========== #

            if (
                out.__contains__("comp_depth")
                and self.cfg.loss["lambda_depth_tv_loss"] > 0.0
            ):
                loss_depth_tv = self.C(self.cfg.loss["lambda_depth_tv_loss"]) * (
                    tv_loss(out["comp_normal"].permute(0, 3, 1, 2))
                    + tv_loss(out["comp_depth"].permute(0, 3, 1, 2))
                )
                self.log(f"train/loss_depth_tv", loss_depth_tv)
                loss_gs += loss_depth_tv

            if self.cfg.loss["lambda_normal_smooth_loss"] > 0.0:
                if "comp_normal" not in out:
                    raise ValueError(
                        "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                    )
                normal = out["comp_normal"]
                loss_normal_smooth = self.C(
                    self.cfg.loss["lambda_normal_smooth_loss"]) * (
                    (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean() + ( \
                    normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean())
                self.log(f"train/loss_normal_smooth", loss_normal_smooth)
                loss_gs += loss_normal_smooth

            # =========== Normal losses =========== #
            for name, value in self.cfg.loss.items():
                self.log(f"train_params/{name}", self.C(value))

        if train_gaussian_loss:
            loss_gs.backward()

        return {"loss": loss_sdf + loss_sds}

    def validation_step(self, batch, batch_idx):

        save_instances = True
        if save_instances:
            prompt_list = [prompt_processor.prompt for prompt_processor in self.prompt_processor_list]
            saving_inst_name = [prompt_str.lower().replace(" ", "_") for prompt_str in prompt_list[: -1]]

            for i, inst_name in enumerate(saving_inst_name):
                out = self(batch, i, self.global_step - 1)
                self.save_image_grid(
                    f"it{self.global_step}-{inst_name}-{batch['index'][0]}.png",
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                    + (
                        [
                            {
                                "type": "rgb",
                                "img": out["comp_normal"][0],
                                "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                            }
                        ]
                        if "comp_normal" in out
                        else []
                    )
                    + (
                        [
                            {
                                "type": "rgb",
                                "img": out["comp_pred_normal"][0],
                                "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                            }
                        ]
                        if "comp_pred_normal" in out
                        else []
                    ),
                    name="validation_step",
                    step=self.global_step,
                )

        out = self(batch)

        # out = self(batch)
        # import pdb; pdb.set_trace()
        self.save_image_grid(
            f"it{self.global_step}-{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_pred_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_pred_normal" in out
                else []
            ),
            name="validation_step",
            step=self.global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        from ..extract_mesh import marching_cubes_sdf_part
        # save ply and mesh at the first step of test
        if batch["index"][0] == 0:
            iteration_for_sdf_dataset = (self.true_global_step // self.cfg.reset_dataset_every) * self.cfg.reset_dataset_every
            assert iteration_for_sdf_dataset != 0
            # 1. save ply
            save_path = self.get_save_path('point_cloud.ply')
            self.sugar.save_ply(checkpoint_path=os.path.dirname(save_path), iteration=self.true_global_step)

            # 2. save mesh by sdf network
            sugar_checkpoint_path = os.path.join(os.path.dirname(self.get_save_dir()), 'sugar')
            for i in range(2):
                neus_i = getattr(self.sugar, f'neus_{i}')
                if i == 0:
                    obj_i_points = self.sugar.points[0: self.get_accum[0]].detach().cpu().numpy()
                else:
                    obj_i_points = self.sugar.points[self.get_accum[0]: self.get_accum[1]].detach().cpu().numpy()

                neus_i.reset_datasets(
                    sugar_checkpoint_path, obj_i_points, self.obj_name_list[i], iteration_for_sdf_dataset,
                    scene_name='threestudio', save_dataset=False)

                evaluated_mesh = marching_cubes_sdf_part(
                    neus_i, obj_name=self.obj_name_list[i],
                    iteration=self.true_global_step,
                    checkpoint_path=os.path.dirname(save_path),
                    resolution=256,
                    vertex_color=False,
                    part=1, thres=0.002,
                    move_surf=False)

        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_pred_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_pred_normal" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )


    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        print('The optimized 3D layout depths:', self.geometry.get_layout_depths)

    def load_gaussian_checkpoint(self, ckpt_path) -> None:
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        num_pts = ckpt_dict["state_dict"]["geometry._xyz"].shape[0]
        pcd = BasicPointCloud(
            points=np.zeros((num_pts, 3)),
            colors=np.zeros((num_pts, 3)),
            normals=np.zeros((num_pts, 3)),
        )
        self.geometry.create_from_pcd(pcd, 10)
        self.geometry.training_setup()  # setup learning rate
        # super().on_load_checkpoint(ckpt_dict)
        # self.material.load_state_dict(ckpt_dict["state_dict"][""])
        self.load_state_dict(ckpt_dict["state_dict"])

    @property
    def get_accum(self):
        mark_accum = torch.cumsum(self.geometry.mark_instances, dim=0)
        return mark_accum

def zero_grad_inst(pc, instance_range):
    # note: 经过prune和densify之后,因为替换了optimizer中的 parameter, 梯度也会清零
    if pc._xyz.grad is None:
        return
    N = pc._xyz.grad.shape[0]
    keep_mask = torch.zeros(N, dtype=torch.bool, device=pc._xyz.grad.device)

    # 将需要保留的位置设为 True（注意：需要转换成 LongTensor）
    keep_mask[torch.from_numpy(instance_range).long().to(pc._xyz.grad.device)] = True

    pc._xyz.grad[~keep_mask] = 0
    pc._features_dc.grad[~keep_mask] = 0
    pc._features_rest.grad[~keep_mask] = 0
    pc._opacity.grad[~keep_mask] = 0
    pc._scaling.grad[~keep_mask] = 0
    pc._rotation.grad[~keep_mask] = 0


if __name__ == '__main__':
    obj_name_list = ['clock', 'hourglass']
    system_path = '/data/code/20022_layout3d/layout3d/threestudio/outputs/best_dreamer/An_antique_clock._A_fleeting_hourglass._A_clock_sits_next_to_an_hourglass@20251014-172202/save'
    sugar_ckpt_path = os.path.join(system_path, 'sugar')
