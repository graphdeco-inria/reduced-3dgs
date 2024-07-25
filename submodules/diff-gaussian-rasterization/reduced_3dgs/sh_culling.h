#ifndef SH_CULLING_H_INCLUDED
#define SH_CULLING_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

void calculateColour(
    const int P,
    const int *degs,
    int max_coeffs,
    const glm::vec3 *means3D,
    const glm::vec3 &campos,
    const float *shs,
    glm::vec3 *colours);

#endif