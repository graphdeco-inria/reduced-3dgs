import torch
from simple_knn._C import distIndex2
from scene import Scene, GaussianModel
import numpy as np
from gaussian_renderer import render
from utils.loss_utils import l1_loss
from utils.image_utils import psnr
import os
import pandas as pd
from diff_gaussian_rasterization._C import calculate_colours_variance
import math
pd.options.display.float_format = '{:,.4f}'.format

from argparse import ArgumentParser

models_configuration = {
    'baseline': {
        'quantised': False,
        'half_float': False,
        'name': 'point_cloud.ply'
        },
    'quantised': {
        'quantised': True,
        'half_float': False,
        'name': 'point_cloud_quantised.ply'
        },
    'quantised_half': {
        'quantised': True,
        'half_float': True,
        'name': 'point_cloud_quantised_half.ply'
        },
}

def get_metrics(experiment_path, pcd_name, iteration=30000):
    metrics = pd.read_json(os.path.join(experiment_path, 'results.json'))[f'{pcd_name}_{iteration}'].T
    try:
        metrics['FPS'] = pd.read_json(os.path.join(experiment_path, 'fps_results.json'))[f'{pcd_name}_{iteration}'].T['FPS']
    except:
        metrics['FPS'] = 0
    return metrics

    
def memory_results(gaussians, experiment_path, scene_name, experiment_name, index, pcd_name, baseline_points=None, half_float=False, quantised=False, skip_compare_baseline=False):
    if quantised:
        experiment_name += "_quantised"
    if half_float:
        experiment_name += "_half"

    byte_size = 1
    float_size = 4 * byte_size
    half_float_size = 2 * byte_size
    P = gaussians.num_primitives

    points_0 = (gaussians._degrees == 0).sum().cpu().numpy()
    points_1 = (gaussians._degrees == 1).sum().cpu().numpy()
    points_2 = (gaussians._degrees == 2).sum().cpu().numpy()
    points_3 = (gaussians._degrees == 3).sum().cpu().numpy()

    colour_values = (points_0 + points_1 * 4 + points_2 * 9 + points_3 * 16) * 3
    sh_values = (points_1 * 3 + points_2 * 8 + points_3 * 15) * 3
    xyz_values = P * 3
    rest_values = P * 8

    colour_memory = colour_values * float_size
    xyz_memory = xyz_values * float_size
    rest_memory = rest_values * float_size
    sh_memory = sh_values * float_size
    total_memory = xyz_memory + colour_memory + rest_memory

    if quantised:
        # 256 is the number of clusters. A bit unimportant as long as it is less than one byte
        colour_memory = colour_values * byte_size + 256 * 16 * float_size
        rest_memory = rest_values * byte_size + 256 * 8 * float_size
        
        sh_memory = sh_values * byte_size + 256 * 15 * float_size
        total_memory = xyz_memory + colour_memory + rest_memory
        if half_float:
            xyz_memory = xyz_values * half_float_size
            total_memory = xyz_memory + colour_memory + rest_memory
    
    elif half_float:
        colour_memory = colour_values * half_float_size
        xyz_memory = xyz_values * half_float_size
        rest_memory = rest_values * half_float_size
        sh_memory = sh_values * half_float_size
        total_memory = colour_memory + xyz_memory + rest_memory

    metrics = get_metrics(experiment_path, pcd_name)

    return pd.Series([scene_name,
                      experiment_name,
                      metrics['LPIPS'],
                      metrics['PSNR'],
                      metrics['SSIM'],
                      metrics['FPS'],
                      total_memory.item(),
                      P,
                      total_memory.item() / (baseline_points * 59 * float_size) if not skip_compare_baseline else float('nan'),
                      (baseline_points * 59 * float_size) / total_memory.item() if not skip_compare_baseline else float('nan'),
                      P/baseline_points if not skip_compare_baseline else float('nan'),
                      total_memory / (P * 59 * float_size),
                      points_0 / P,
                      points_1 / P,
                      points_2 / P,
                      points_3 / P,
                      sh_memory / (P * 45 * float_size),
                      xyz_memory / total_memory,
                      sh_memory / total_memory,
                      colour_memory / total_memory,
                      rest_memory / total_memory
                      ], index=index, name=f'{scene_name}/{experiment_name}')
    

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--eval_folder", required=False, default="/data/graphdeco/user/ppapanto/i3d_output", type=str)
parser.add_argument("--scenes_folder", required=False, default="/data/graphdeco/user/ppapanto/scenes", type=str)
parser.add_argument("--models",
                        help="Types of models to test",
                        choices=models_configuration.keys(),
                        default=['baseline', 'quantised_half'],
                        nargs="+")
args, _ = parser.parse_known_args()

feature_names = ["scene", "experiment", "LPIPS", "PSNR", "SSIM", "FPS", "memory",
                 "n_points", "%memory vs baseline", "memory_gain vs baseline",
                 "%points vs baseline", "%memory reduction", "%points 0 bands",
                 "%points 1 band", "%points 2 bands", "%points 3 bands", "%sh memory vs original sh",
                 "%xyz_memory vs total", "%sh_memory vs total", "%colour memory vs total", "%rest memory vs total"]
df = pd.DataFrame(columns=feature_names)
gaussians = GaussianModel(3)

# For every scene in the evaluation folder iterate over every experiment
for scene_name in os.listdir(args.eval_folder):
    print(scene_name)
    experiment_names = os.listdir(os.path.join(args.eval_folder, scene_name))
    
    # If baseline is not an experiment, no comparison with it will be made
    if 'baseline' not in experiment_names:
        skip_compare_baseline = True
        baseline_points = None
    else:
        skip_compare_baseline = False
        experiment_names.remove('baseline')
        experiment_names = ['baseline'] + experiment_names
    
    for experiment_name in experiment_names:
        for model in args.models:
            name = models_configuration[model]['name']
            quantised = models_configuration[model]['quantised']
            half_float = models_configuration[model]['half_float']

            scene_path = os.path.join(args.scenes_folder, scene_name)
            experiment_path = os.path.join(args.eval_folder, scene_name, experiment_name)
            if 'results.json' not in os.listdir(experiment_path):
                print(f"No results found for {scene_name} {experiment_name}! Run metrics.py")
                continue
            
            gaussians.load_ply(os.path.join(experiment_path, "point_cloud", "iteration_30000", name), quantised=quantised, half_float=half_float)

            if experiment_name == "baseline":
                baseline_points = gaussians.num_primitives
            
            df = df.append(memory_results(gaussians, experiment_path, scene_name, experiment_name, feature_names, name, baseline_points, quantised=quantised, half_float=half_float, skip_compare_baseline=skip_compare_baseline))

with open("results_final.json", "w") as f:
    f.write(df.to_json())
