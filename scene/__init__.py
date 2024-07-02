#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from diff_gaussian_rasterization._C import sphere_ellipsoid_intersection, allocate_minimum_redundancy_value, find_minimum_projected_pixel_size
from simple_knn._C import distIndex2
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, quantise=False, half_float=False):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        ply_name = "point_cloud"
        if quantise:
            ply_name += "_quantised"
        if half_float:
            ply_name += "_half"
        ply_name += ".ply"
        self.gaussians.save_ply(os.path.join(point_cloud_path, ply_name), quantise, half_float)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def find_minimum_projected_pixel_size_python(self):
        # Initialise to a very high number
        pixel_sizes_world = 10000 * torch.ones_like(self.gaussians.get_opacity).detach()
        for camera in self.getTrainCameras():
            w2ndc_transform = camera.full_proj_transform
            ndc_centers_hom = torch.matmul(torch.cat((self.gaussians.get_xyz, torch.ones((self.gaussians.num_primitives, 1), device="cuda")), dim=1).unsqueeze(1), w2ndc_transform.unsqueeze(0)).squeeze()
            ndc_centers_hom /= ndc_centers_hom.clone()[:, -1:]

            depths = ndc_centers_hom[:, 2]
            # Frustum culling. This should utilise visibility filter/radii as in cuda
            mask = torch.logical_and(
                torch.logical_and(
                    torch.logical_and(ndc_centers_hom[:, 0:1] <= 1, ndc_centers_hom[:, 0:1] >= -1),
                    torch.logical_and(ndc_centers_hom[:, 1:2] <= 1, ndc_centers_hom[:, 1:2] >= -1)),
                torch.logical_and(ndc_centers_hom[:, 2:3] <= 1, ndc_centers_hom[:, 2:3] >= 0)).squeeze()
            
            p_hom = torch.zeros_like(ndc_centers_hom)
            p_hom[:, 0 if camera.image_width > camera.image_height else 1] = min(2/camera.image_width, 2/camera.image_height)
            p_hom[:, 2] = depths
            p_hom[:, 3] = torch.ones(self.gaussians.num_primitives, device="cuda")
            
            p_hom_zero = torch.zeros_like(ndc_centers_hom)
            p_hom_zero[:, 2] = depths
            p_hom_zero[:, 3] = torch.ones(self.gaussians.num_primitives, device="cuda")
            
            # NDC [-1, 1] x [-1, 1] -> Pixel space [0, W] x [0, H]
            # [x, y, depth, 1] [x', y, depth, 1] -> [x_proj, y_proj, depth_proj, 1] - [x'_proj, y_proj, depth_proj, 1] -> [dx_proj, 0, 0, 1]
            # [dx, 0, depth, 1] [0, dy, depth, 1]

            p_hom[mask] = p_hom[mask] @ w2ndc_transform.inverse().unsqueeze(0)
            p_hom[mask] /= p_hom[mask][:, -1:]

            p_hom_zero[mask] = p_hom_zero[mask] @ w2ndc_transform.inverse().unsqueeze(0)
            p_hom_zero[mask] /= p_hom_zero[mask][:, -1:]

            pixel_sizes_world[mask] = torch.min(pixel_sizes_world[mask], torch.norm((p_hom[mask] - p_hom_zero[mask])[:, :3], dim=1, keepdim=True))
        return pixel_sizes_world
    
    def calculate_redundancy_metric(self, pixel_scale=1.0, num_neighbours=30):
        cameras = self.getTrainCameras()
        # Get minimum projected pixel size
        cube_size = find_minimum_projected_pixel_size(
            torch.stack([camera.full_proj_transform for camera in cameras], dim=0),
            torch.stack([camera.inverse_full_proj_transform for camera in cameras], dim=0),
            self.gaussians._xyz,
            torch.tensor([camera.image_height for camera in cameras], device="cuda", dtype=torch.int32),
            torch.tensor([camera.image_width for camera in cameras], device="cuda", dtype=torch.int32)
        )
        
        scaled_pixel_size = cube_size * pixel_scale
        half_diagonal = scaled_pixel_size * torch.sqrt(torch.tensor([3], device="cuda")) / 2 
        
        # Find neighbours as candidates for the intersection test
        _, indices = distIndex2(self.gaussians.get_xyz, num_neighbours)
        indices = indices.view(-1, num_neighbours)
        
        # Do the intersection check
        redundancy_metrics, intersection_mask = sphere_ellipsoid_intersection(self.gaussians._xyz,
                                                                              self.gaussians.get_scaling,
                                                                              self.gaussians.get_rotation,
                                                                              indices,
                                                                              half_diagonal,
                                                                              num_neighbours)
        # We haven't counted count for the primitive at the center of each sphere, so add 1 to everything
        redundancy_metrics += 1
        
        indices = torch.cat((torch.arange(self.gaussians.num_primitives, device="cuda", dtype=torch.int).view(-1, 1), indices), dim=1)
        intersection_mask = torch.cat((torch.ones_like(self.gaussians._opacity, device="cuda", dtype=bool), intersection_mask), dim=1)
        
        min_redundancy_metrics = allocate_minimum_redundancy_value(redundancy_metrics, indices, intersection_mask, num_neighbours+1)[0]
        return min_redundancy_metrics, cube_size