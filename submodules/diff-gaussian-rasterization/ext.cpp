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

#include <torch/extension.h>
#include "rasterize_points.h"
#include "reduced_3dgs.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians_variableSH_bands", &RasterizeGaussiansVariableSHBandsCUDA);
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("calculate_colours_variance", &Reduced3DGS::calculateColourVariance);
  m.def("sphere_ellipsoid_intersection", &Reduced3DGS::intersectionTest);
  m.def("allocate_minimum_redundancy_value", &Reduced3DGS::assignFinalRedundancyValue);
  m.def("find_minimum_projected_pixel_size", &Reduced3DGS::calculatePixelSize);
  m.def("kmeans_cuda", &Reduced3DGS::kmeans);
}