#ifndef REDUNDANCY_SCORE_H_INCLUDED
#define REDUNDANCY_SCORE_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

void findMinimumRedundancyValue(
    const int P,
    const int *redundancy_values,
    const int *neighbours_indices,
    const bool *intersection_mask,
    int *minimum_redundancy_values,
    const int knn);

void transformCentersNDC(
    const int P,
    const glm::vec3 *centers,
    const glm::mat4 *projmatrix,
    const glm::mat4 *inverse_projmatrix,
    const int image_height,
    const int image_width,
    float *pixel_sizes);

void sphereEllipsoidIntersection(
    const int P,
    const glm::vec3 *means3D,
    const glm::vec3 *scales,
    const glm::mat3 *R,
    const int *neighbours_indices,
    const float *sphere_radius,
    int *redundancy_values,
    bool *intersection_mask,
    const int knn);

void buildRotationMatrix(
    const int P,
    const float4 *rotations,
    glm::mat3 *rotations_matrices);

#endif