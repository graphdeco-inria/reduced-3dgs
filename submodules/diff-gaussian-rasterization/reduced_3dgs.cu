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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>
#include <glm/glm.hpp>
#include "reduced_3dgs.h"
#include "reduced_3dgs/kmeans.h"
#include "reduced_3dgs/redundancy_score.h"
#include "reduced_3dgs/sh_culling.h"	

#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "cuda_rasterizer/auxiliary.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include "cuda_rasterizer/forward.h"
using namespace torch::indexing;

std::function<char *(size_t N)> resizeFunctional(torch::Tensor &t);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Reduced3DGS::calculateColourVariance(
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
	const int max_sh_deg)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3)
	{
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);
	int rendered = 0;
	int M = 0;
	if (P != 0)
	{
		if (sh.size(0) != 0)
		{
			M = sh.size(1);
		}
	}

	torch::Tensor colours = torch::zeros({P, max_sh_deg + 1, 3}, means3D.options());
	torch::Tensor colourDistance = torch::zeros({P, 1}, means3D.options());
	torch::Tensor colourDistancesAccum = torch::zeros({P, max_sh_deg}, means3D.options());

	torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

	torch::Tensor wSum = torch::zeros({P, 1}, means3D.options());
	torch::Tensor wSumSq = torch::zeros({P, 1}, means3D.options());
	torch::Tensor mean = torch::zeros({P, 1, 3}, means3D.options());
	torch::Tensor variance = torch::zeros({P, 1, 3}, means3D.options());

	glm::vec3 *cam_positionsAlias = (glm::vec3 *)cam_positions.contiguous().data<float>();
	for (int i = 0; i < cam_positions.size(0); ++i)
	{
		const int H = image_height.index({i}).item<int>();
		const int W = image_width.index({i}).item<int>();
		const float tan_fovx = tan_fovxs.index({i}).item<float>();
		const float tan_fovy = tan_fovys.index({i}).item<float>();

		torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
		torch::Tensor out_density = torch::full({1, H, W}, 0.0, float_opts);
		torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
		torch::Tensor density3D = torch::full({P, 1}, 0.0, float_opts);
		torch::Tensor out_touched_pixels = torch::full({P, 1}, 0, int_opts);
		torch::Tensor out_transmittance = torch::full({P, 1}, 0.f, float_opts);

		torch::Device device(torch::kCUDA);
		torch::TensorOptions options(torch::kByte);
		torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
		torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
		torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
		std::function<char *(size_t)> geomFunc = resizeFunctional(geomBuffer);
		std::function<char *(size_t)> binningFunc = resizeFunctional(binningBuffer);
		std::function<char *(size_t)> imgFunc = resizeFunctional(imgBuffer);

        // Unused variables, since we use forward only for preprocess
        torch::Tensor background = torch::zeros({3}, means3D.options());
        torch::Tensor colors = torch::empty({0}, means3D.options());
        torch::Tensor cov3D_precomp = torch::empty({0}, means3D.options());

		out_touched_pixels.zero_();
		out_transmittance.zero_();

        // Used to find the average transmittance and visibility mask
        // Background colour is set as 0 as we don't care about its value
		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P, degrees.contiguous().data<int>(), M,
			background.contiguous().data<float>(),
			W, H,
			means3D.contiguous().data<float>(),
			sh.contiguous().data_ptr<float>(),
			colors.contiguous().data<float>(),
			opacity.contiguous().data<float>(),
			scales.contiguous().data_ptr<float>(),
			1.f,
			rotations.contiguous().data_ptr<float>(),
			cov3D_precomp.contiguous().data<float>(),
			cam_viewmatrices.index({i}).contiguous().data<float>(),
			cam_projmatrices.index({i}).contiguous().data<float>(),
			cam_positions.index({i}).contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			false,
			out_color.contiguous().data<float>(),
			out_touched_pixels.contiguous().data<int>(),
			out_transmittance.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			true,
			false);

		present = radii > 0;

		colours.zero_();

		out_transmittance /= out_touched_pixels.max(torch::ones({P, 1}, means3D.options()));
		wSum += out_transmittance;
		wSumSq += out_transmittance.pow(2);

        calculateColour(P,
                                                      degrees.contiguous().data<int>(),
                                                      M,
                                                      (glm::vec3 *)means3D.contiguous().data<float>(),
                                                      cam_positionsAlias[i],
                                                      sh.contiguous().data<float>(),
                                                      (glm::vec3 *)colours.contiguous().data<float>());
        colours.index_put_({present.logical_not()}, 0);

		for (int curDegree = 0; curDegree < max_sh_deg; curDegree++)
		{
			// Colour distance up to deg-th band (full colour) vs up to curDegree-th band (DC)
			colourDistance =
				(colours.index({Slice(None), Slice(max_sh_deg, None)}) -
				 colours.index({Slice(None), Slice(curDegree, curDegree + 1)}))
					.pow(2)
					.sum(2)
					.sqrt();
			colourDistance.index_put_({colourDistance.isnan()}, 0.f);
			colourDistancesAccum.index_put_(
				{Slice(None), Slice(curDegree, curDegree + 1)},
				colourDistancesAccum.index({Slice(None), Slice(curDegree, curDegree + 1)}) +
					out_transmittance * colourDistance);
		}

		// Calculate rolling average and variance
		auto colour = colours.index({Slice(None), Slice(max_sh_deg, None)});
		auto mean_old = mean;
		auto coefficient = out_transmittance / wSum;
		coefficient.index_put_({coefficient.isnan()}, 0.f);
		mean.index_put_(
			{present},
			mean_old.index({present}) +
				coefficient.index({present}).view({-1, 1, 1}) *
					(colour.index({present}) - mean_old.index({present})));
		variance.index_put_(
			{present},
			variance.index({present}) +
				out_transmittance.index({present}).view({-1, 1, 1}) *
					(colour.index({present}) - mean_old.index({present})) *
					(colour.index({present}) - mean.index({present})));
	}

	// Return average_distances, colour_variances and mean_colour
	return std::make_tuple(colourDistancesAccum / wSum, variance / wSum.view({-1, 1, 1}), mean);
}

std::tuple<torch::Tensor, torch::Tensor>
Reduced3DGS::intersectionTest(
    const torch::Tensor &means3D,
    const torch::Tensor &scales,
    const torch::Tensor &rotations,
    const torch::Tensor &neighbours_indices,
    const torch::Tensor &sphere_radius,
    const int knn)
{
    const int P = means3D.size(0);
    auto float_opts = means3D.options().dtype(torch::kFloat32);
    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto bool_opts = means3D.options().dtype(torch::kBool);

    torch::Tensor rotation_matrices = torch::zeros({P, 3, 3}, float_opts);
    torch::Tensor redundancy_values = torch::zeros({P, 1}, int_opts);
    torch::Tensor intersection_mask = torch::zeros({P, knn}, bool_opts);

    buildRotationMatrix(P,
                        (float4 *)(rotations.contiguous().data<float>()),
                        (glm::mat3 *)rotation_matrices.contiguous().data<float>());
    sphereEllipsoidIntersection(P,
                                (glm::vec3 *)means3D.contiguous().data<float>(),
                                (glm::vec3 *)scales.contiguous().data<float>(),
                                (glm::mat3 *)rotation_matrices.contiguous().data<float>(),
                                neighbours_indices.contiguous().data<int>(),
                                sphere_radius.contiguous().data<float>(),
                                redundancy_values.contiguous().data<int>(),
                                intersection_mask.contiguous().data<bool>(),
                                knn);

    return std::make_tuple(redundancy_values, intersection_mask);
}

torch::Tensor Reduced3DGS::calculatePixelSize(
    const torch::Tensor &w2ndc_transforms,
    const torch::Tensor &w2ndc_transforms_inverse,
    const torch::Tensor &means3D,
    const torch::Tensor &image_height,
    const torch::Tensor &image_width)
{
    const int P = means3D.size(0);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Tensor pixel_values = torch::full({P, 1}, 10000, float_opts);

    for (int i = 0; i < w2ndc_transforms.size(0); ++i)
    {
        transformCentersNDC(
            P,
            (glm::vec3 *)means3D.contiguous().data<float>(),
            (glm::mat4 *)w2ndc_transforms.index({i}).contiguous().data<float>(),
            (glm::mat4 *)w2ndc_transforms_inverse.index({i}).contiguous().data<float>(),
            image_height.index({i}).item<int>(),
            image_width.index({i}).item<int>(),
            pixel_values.contiguous().data<float>());
    }
    return pixel_values;
}

// This function assigns for each point the minimum redundancy value out
// of all its intersecting neighbours
std::tuple<torch::Tensor>
Reduced3DGS::assignFinalRedundancyValue(
    const torch::Tensor &redundancy_values,
    const torch::Tensor &neighbours_indices,
    const torch::Tensor &intersection_mask,
    const int knn)
{
    const int P = redundancy_values.size(0);
    auto int_opts = redundancy_values.options().dtype(torch::kInt32);
    torch::Tensor minimum_redundancy_values = torch::full({P, 1}, P, int_opts);

    findMinimumRedundancyValue(P,
                               redundancy_values.contiguous().data<int>(),
                               neighbours_indices.contiguous().data<int>(),
                               intersection_mask.contiguous().data<bool>(),
                               minimum_redundancy_values.contiguous().data<int>(),
                               knn);
    return std::make_tuple(minimum_redundancy_values);
}

// Works with 256 centers 1 dimensional data only
std::tuple<torch::Tensor, torch::Tensor>
Reduced3DGS::kmeans(
	const torch::Tensor &values,
	const torch::Tensor &centers,
	const float tol,
	const int max_iterations)
{
	const int n_values = values.size(0);
	const int n_centers = centers.size(0);
	torch::Tensor ids = torch::zeros({n_values, 1}, values.options().dtype(torch::kInt32));
	torch::Tensor new_centers = torch::zeros({n_centers}, values.options().dtype(torch::kFloat32));
	torch::Tensor old_centers = torch::zeros({n_centers}, values.options().dtype(torch::kFloat32));
	new_centers = centers.clone();
	torch::Tensor center_sizes = torch::zeros({n_centers}, values.options().dtype(torch::kInt32));

	for (int i = 0; i < max_iterations; ++i)
	{
		updateIds(
			values.contiguous().data<float>(),
			ids.contiguous().data<int>(),
			new_centers.contiguous().data<float>(),
			n_values,
			n_centers);

		old_centers = new_centers.clone();
		new_centers.zero_();
		center_sizes.zero_();

		updateCenters(
			values.contiguous().data<float>(),
			ids.contiguous().data<int>(),
			new_centers.contiguous().data<float>(),
			center_sizes.contiguous().data<int>(),
			n_values,
			n_centers);

		new_centers = new_centers / center_sizes;
		new_centers.index_put_({new_centers.isnan()}, 0.f);
		float center_shift = (old_centers - new_centers).abs().sum().item<float>();
		if (center_shift < tol)
			break;
	}

	updateIds(
		values.contiguous().data<float>(),
		ids.contiguous().data<int>(),
		new_centers.contiguous().data<float>(),
		n_values,
		n_centers);

	return std::make_tuple(ids, new_centers);
}