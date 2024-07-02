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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

losses = ["l1_loss", "ssim_loss", "alpha_regul", "sh_sparsity_loss", "total_loss", "iter_time"]
dens_statistic_dict = {"n_points_cloned": 0, "n_points_split": 0, "n_points_mercied": 0, "n_points_pruned": 0, "redundancy_threshold": 0, "opacity_threshold": 0}

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, args.variable_sh_bands)
    loss_aggregator = {loss : 0 for loss in losses}

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    fine_tune_start = opt.iterations
    if len(args.cull_SH) != 0 or args.mercy_points:
        fine_tune_start = opt.iterations - 3000

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                with torch.no_grad():
                    net_image_bytes = None
                    custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            with torch.no_grad():
                gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, lambda_sh_sparsity=args.lambda_sh_sparsity)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if args.lambda_alpha_regul == 0:
            Lalpha_regul = torch.tensor([0.], device=image.device)
        else:
            points_opacity = gaussians.get_opacity[visibility_filter]
            Lalpha_regul = points_opacity.abs().mean()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        Lssim = 1.0 - ssim(image, gt_image)
        # Just for logging reasons. Isn't actually used. Can be removed safely
        sh_sparsity_loss = args.lambda_sh_sparsity * gaussians._features_rest.detach()[visibility_filter].abs().mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim + Lalpha_regul * args.lambda_alpha_regul
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, Lssim, Lalpha_regul, sh_sparsity_loss, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_aggregator, dens_statistic_dict)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, dens_statistic_dict, args.store_grads)
            
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            elif args.prune_dead_points and iteration % opt.densification_interval == 0:
                gaussians.prune(1/255, scene.cameras_extent, None, dens_statistic_dict)

            if args.mercy_points and iteration % (args.mercy_interval*opt.densification_interval) == 0 and iteration <= fine_tune_start and \
            (iteration >= opt.densify_until_iter or iteration % opt.opacity_reset_interval != 0):
                gaussians._splatted_num_accum, _ = scene.calculate_redundancy_metric(pixel_scale = args.box_size)
                gaussians._splatted_num_accum = gaussians._splatted_num_accum.unsqueeze(1)
                scene.gaussians.mercy_points(dens_statistic_dict, args.lambda_mercy, args.mercy_minimum, args.mercy_type)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # Delete low opacity dead points
                if args.prune_dead_points:
                    gaussians.prune(1/255, scene.cameras_extent, None, dens_statistic_dict)
                scene.save(iteration)

            if iteration in args.cull_SH:
                print("\n[ITER {}] Pruning SH Bands".format(iteration))
                gaussians.cull_sh_bands(scene.getTrainCameras(), threshold=args.cdist_threshold*np.sqrt(3)/255, std_threshold=args.std_threshold)

    scene.save(iteration)
    scene.gaussians.produce_clusters(store_dict_path=scene.model_path)        
    scene.save(iteration, quantise=True)
    scene.save(iteration, quantise=True, half_float=True)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, Lssim, Lalpha_regul, sh_sparsity_loss, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_aggregator, density_statistics_dict):
    if iteration % args.densification_interval == 0:
        (average_l1_loss,
        average_ssim_loss,
        average_alpha_regul,
        average_sh_sparsity_loss,
        average_total_loss,
        average_iter_time) = (loss_aggregator["l1_loss"]/args.densification_interval,
                              loss_aggregator["ssim_loss"]/args.densification_interval,
                              loss_aggregator["alpha_regul"]/args.densification_interval,
                              loss_aggregator["sh_sparsity_loss"]/args.densification_interval,
                              loss_aggregator["total_loss"]/args.densification_interval,
                              loss_aggregator["iter_time"]/args.densification_interval)
        if tb_writer:
            tb_writer.add_scalar('train_loss_patches/l1_loss', average_l1_loss, iteration)
            tb_writer.add_scalar('train_loss_patches/ssim_loss', average_ssim_loss, iteration)
            tb_writer.add_scalar('train_loss_patches/alpha_regul', average_alpha_regul, iteration)
            tb_writer.add_scalar('train_loss_patches/sh_sparsity_loss', average_sh_sparsity_loss, iteration)
            tb_writer.add_scalar('train_loss_patches/total_loss', average_total_loss, iteration)
            tb_writer.add_scalar('iter_time', average_iter_time, iteration)
            tb_writer.add_scalar('total_points/points_cloned', density_statistics_dict['n_points_cloned'], iteration)
            tb_writer.add_scalar('total_points/points_split', density_statistics_dict['n_points_split'], iteration)
            tb_writer.add_scalar('total_points/points_mercied', density_statistics_dict['n_points_mercied'], iteration)
            tb_writer.add_scalar('total_points/points_mercied_%', density_statistics_dict['n_points_mercied'] / scene.gaussians.num_primitives, iteration)
            tb_writer.add_scalar('total_points/points_pruned', density_statistics_dict['n_points_pruned'], iteration)
            tb_writer.add_scalar('total_points/points_pruned_%', density_statistics_dict['n_points_pruned'] / scene.gaussians.num_primitives, iteration)
            tb_writer.add_scalar('total_points/redundancy_threshold', density_statistics_dict['redundancy_threshold'], iteration)
            tb_writer.add_scalar('total_points/opacity_threshold', density_statistics_dict['opacity_threshold'], iteration)
        density_statistics_dict['n_points_cloned'] = 0
        density_statistics_dict['n_points_split'] = 0
        density_statistics_dict['n_points_mercied'] = 0
        density_statistics_dict['n_points_pruned'] = 0
        loss_aggregator["l1_loss"] = 0
        loss_aggregator["ssim_loss"] = 0
        loss_aggregator["alpha_regul"] = 0
        loss_aggregator["sh_sparsity_loss"] = 0
        loss_aggregator["total_loss"] = 0
        loss_aggregator["iter_time"] = 0
    else:
        loss_aggregator["l1_loss"] += Ll1.detach().item()
        loss_aggregator["ssim_loss"] += Lssim.detach().item()
        loss_aggregator["alpha_regul"] += Lalpha_regul.detach().item()
        loss_aggregator["sh_sparsity_loss"] += sh_sparsity_loss.detach().item()
        loss_aggregator["total_loss"] += loss.detach().item()
        loss_aggregator["iter_time"] += elapsed

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.num_primitives, iteration)
            tb_writer.add_scalar('total_points/points_0_bands_%', (scene.gaussians._degrees == 0).sum() / scene.gaussians.num_primitives, iteration)
            tb_writer.add_scalar('total_points/points_1_bands_%', (scene.gaussians._degrees == 1).sum() / scene.gaussians.num_primitives, iteration)
            tb_writer.add_scalar('total_points/points_2_bands_%', (scene.gaussians._degrees == 2).sum() / scene.gaussians.num_primitives, iteration)
            tb_writer.add_scalar('total_points/points_3_bands_%', (scene.gaussians._degrees == 3).sum() / scene.gaussians.num_primitives, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--cull_SH", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
