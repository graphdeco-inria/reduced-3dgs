#include "sh_culling.h"
#include "../cuda_rasterizer/auxiliary.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ void computeColorFromSH(const int idx, const int *degs, int max_coeffs, const glm::vec3 *means, glm::vec3 campos, const float *shs, glm::vec3 *out_colours)
{
    // The implementation is loosely based on code for
    // "Differentiable Point-Based Radiance Fields for
    // Efficient View Synthesis" by Zhang et al. (2022)
    glm::vec3 pos = means[idx];
    glm::vec3 dir = pos - campos;
    dir = dir / glm::length(dir);

    glm::vec3 *sh = ((glm::vec3 *)shs) + idx * max_coeffs;
    glm::vec3 result = SH_C0 * sh[0];
    result += 0.5f;

    const int deg = degs[idx];

    out_colours[idx * 4 + 0] = glm::max(result, 0.0f);
    if (deg == 0)
        return;

    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];
    out_colours[idx * 4 + 1] = glm::max(result, 0.0f);
    if (deg == 1)
        return;

    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    result = result +
             SH_C2[0] * xy * sh[4] +
             SH_C2[1] * yz * sh[5] +
             SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
             SH_C2[3] * xz * sh[7] +
             SH_C2[4] * (xx - yy) * sh[8];

    out_colours[idx * 4 + 2] = glm::max(result, 0.0f);
    if (deg == 2)
        return;

    result = result +
             SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
             SH_C3[1] * xy * z * sh[10] +
             SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
             SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
             SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
             SH_C3[5] * z * (xx - yy) * sh[14] +
             SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
    out_colours[idx * 4 + 3] = glm::max(result, 0.0f);

    return;
}

__global__ void calculateColourCUDA(
    const int P,
    const int *degs,
    int max_coeffs,
    const glm::vec3 *means3D,
    const glm::vec3 &campos,
    const float *shs,
    glm::vec3 *colours)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;
    computeColorFromSH(idx, degs, max_coeffs, means3D, campos, shs, colours);
    return;
}

void calculateColour(
    const int P,
    const int *degs,
    int max_coeffs,
    const glm::vec3 *means3D,
    const glm::vec3 &campos,
    const float *shs,
    glm::vec3 *colours)
{
    calculateColourCUDA<<<(P + 255) / 256, 256>>>(P,
                                                  degs,
                                                  max_coeffs,
                                                  means3D,
                                                  campos,
                                                  shs,
                                                  colours);
}