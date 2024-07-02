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
from argparse import ArgumentParser

class SceneGroup:
    def __init__(self, scene_names, images_arg=""):
        self._scene_names = scene_names
        self._images_arg = images_arg
        self._source = None

mipnerf360_outdoor_scenegroup = SceneGroup(["bicycle", "flowers", "garden", "stump", "treehill"], "-i images_4")
mipnerf360_indoor_scenegroup = SceneGroup(["room", "counter", "kitchen", "bonsai"], "-i images_2")
tanks_and_temples_scenegroup = SceneGroup(["truck", "train"])
deep_blending_scenegroup = SceneGroup(["drjohnson", "playroom"])

scene_group_list = []
scene_group_list.append(mipnerf360_outdoor_scenegroup)
scene_group_list.append(mipnerf360_indoor_scenegroup)
scene_group_list.append(tanks_and_temples_scenegroup)
scene_group_list.append(deep_blending_scenegroup)

configuration = {}
configuration["high_sh_sparsity"] = "--store_grads --lambda_sh_sparsity=0.1"
configuration["sh_sparsity"] = "--store_grads --lambda_sh_sparsity=0.01"
configuration["cull_SH"] = "--store_grads --cull_SH 15000 --std_threshold=0.04"
configuration["mercy_points"] = "--mercy_points --prune_dead_points --store_grads --lambda_alpha_regul=0.001"

# Paper experiments
# Baseline
configuration["baseline"] = ""
# Just point culling
configuration["mercy_points"] = " ".join([configuration["mercy_points"], "--mercy_type=redundancy_opacity_opacity"])
# Ours
configuration["full_final"] = " ".join([configuration["high_sh_sparsity"], configuration["cull_SH"], configuration["mercy_points"], "--std_threshold=0.04 --cdist_threshold=6 --mercy_type=redundancy_opacity_opacity"])
# Opacity culling
configuration["mercy_type_opacity"] = " ".join([configuration["high_sh_sparsity"], configuration["cull_SH"], configuration["mercy_points"], "--std_threshold=0.04 --cdist_threshold=6 --mercy_type=opacity"])
# Redundancy Random culling
configuration["mercy_type_redundancy_random"] = " ".join([configuration["high_sh_sparsity"], configuration["cull_SH"], configuration["mercy_points"], "--std_threshold=0.04 --cdist_threshold=6 --mercy_type=redundancy_random"])
# Redundancy with low opacity culling
configuration["mercy_type_redundancy_opacity"] = " ".join([configuration["high_sh_sparsity"], configuration["cull_SH"], configuration["mercy_points"], "--std_threshold=0.04 --cdist_threshold=6 --mercy_type=redundancy_opacity"])
# High compression
configuration["high_compression"] = " ".join([configuration["high_sh_sparsity"], configuration["mercy_points"], "--std_threshold=0.06 --cdist_threshold=8 --cull_SH 15000 --mercy_minimum=2 --mercy_type=redundancy_opacity_opacity"])
# Low compression
configuration["low_compression"] = " ".join([configuration["high_sh_sparsity"], configuration["mercy_points"], "--cull_SH 15000", "--std_threshold=0.01", "--cdist_threshold=1 --mercy_type=redundancy_opacity_opacity"])

all_scene_names = [scene for scene_group in scene_group_list for scene in scene_group._scene_names]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--skip_measure_fps", action="store_true", help="Argument passed to render.py")
parser.add_argument("--output_path", default="./eval", type=str)
parser.add_argument("--experiments", "-e", default=["full_final"], nargs="+", choices=list(configuration.keys()), type=str)
parser.add_argument("--scenes", "-s", default=all_scene_names, nargs="+", choices=all_scene_names, type=str)
parser.add_argument('--mipnerf360', "-m360", required=False, default="/m360", type=str)
parser.add_argument("--tanksandtemples", "-tat", required=False, default="/tat", type=str)
parser.add_argument("--deepblending", "-db", required=False, default="/db", type=str)
args = parser.parse_args()

mipnerf360_outdoor_scenegroup._source = args.mipnerf360
mipnerf360_indoor_scenegroup._source = args.mipnerf360
tanks_and_temples_scenegroup._source = args.tanksandtemples
deep_blending_scenegroup._source = args.deepblending

for scene_group in scene_group_list:
    for scene in scene_group._scene_names:
        if scene in args.scenes:
            for experiment in args.experiments:
                output_path = f"{args.output_path}/{scene}/{experiment}"
                if not args.skip_training:
                    common_args = " --quiet --eval --test_iterations -1 "
                    os.system(f"python train.py -s {scene_group._source}/{scene} {scene_group._images_arg} -m {output_path} {common_args} {configuration[experiment]}")

                if not args.skip_rendering:
                    common_args = f" --quiet --eval --skip_train {'--skip_measure_fps' if args.skip_measure_fps else ''}"
                    os.system(f"python render.py --iteration 30000 -s {scene_group._source}/{scene} -m {output_path} {common_args}")

                if not args.skip_metrics:
                    os.system(f"python metrics.py -m {output_path}")
