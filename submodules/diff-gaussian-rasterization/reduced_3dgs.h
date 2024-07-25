/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef REDUCED_3DGS_H_INCLUDED
#define REDUCED_3DGS_H_INCLUDED

#include <vector>
#include <functional>
#include <torch/extension.h>

namespace Reduced3DGS
{
	std::tuple<torch::Tensor, torch::Tensor>
	kmeans(
		const torch::Tensor &values,
		const torch::Tensor &centers,
		const float tol,
		const int max_iterations);

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
	calculateColourVariance(
		const torch::Tensor &cam_positions,
		const torch::Tensor &means3D,
		const torch::Tensor &opacity,
		const torch::Tensor &scales,
		const torch::Tensor &rotations,
		const torch::Tensor &cam_viewmatrices,
		const torch::Tensor &cam_projmatrices,
		const torch::Tensor &tan_fovxs,
		const torch::Tensor &tan_fovys,
		const torch::Tensor &image_height,
		const torch::Tensor &image_width,
		const torch::Tensor &sh,
		const torch::Tensor &degrees,
		const int deg);

	std::tuple<torch::Tensor, torch::Tensor> intersectionTest(
		const torch::Tensor &means3D,
		const torch::Tensor &scales,
		const torch::Tensor &rotations,
		const torch::Tensor &neighbours_indices,
		const torch::Tensor &sphere_radius,
		const int knn);

	std::tuple<torch::Tensor>
	assignFinalRedundancyValue(
		const torch::Tensor &redundancyValues,
		const torch::Tensor &neighbours_indices,
		const torch::Tensor &intersection_mask,
		const int knn);

	torch::Tensor
	calculatePixelSize(
		const torch::Tensor &w2ndc_transforms,
		const torch::Tensor &w2ndc_transforms_inverse,
		const torch::Tensor &means3D,
		const torch::Tensor &image_height,
		const torch::Tensor &image_width);
};

#endif