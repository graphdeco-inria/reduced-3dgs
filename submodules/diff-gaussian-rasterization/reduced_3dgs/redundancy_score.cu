#include "redundancy_score.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


__global__ void findMinimumRedundancyValueCUDA(
    const int P,
    const int *redundancy_values,
    const int *neighbours_indices,
    const bool *intersection_mask,
    int *minimum_redundancy_values,
    const int knn)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;
    const bool *curr_intersection_mask = intersection_mask + idx * knn;
    const int *curr_neighbours_indices = neighbours_indices + idx * knn;
    for (int i = 0; i < knn; i++)
    {
        if (curr_intersection_mask[i])
        {
            int neighbour_idx = curr_neighbours_indices[i];
            atomicMin(&minimum_redundancy_values[neighbour_idx], redundancy_values[idx]);
        }
    }
}

void findMinimumRedundancyValue(const int P,
                                const int *redundancy_values,
                                const int *neighbours_indices,
                                const bool *intersection_mask,
                                int *minimum_redundancy_values,
                                const int knn)
{
    findMinimumRedundancyValueCUDA<<<(P + 255) / 256, 256>>>(P,
                                                             redundancy_values,
                                                             neighbours_indices,
                                                             intersection_mask,
                                                             minimum_redundancy_values,
                                                             knn);
}


__global__ void transformCentersNDCCUDA(
    const int P,
    const glm::vec3 *centers,
    const glm::mat4 *projmatrix,
    const glm::mat4 *inverse_projmatrix,
    const int image_height,
    const int image_width,
    float *pixel_sizes)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Transform point by projecting
    glm::vec4 p_orig = glm::vec4(centers[idx], glm::vec1(1.f));
    glm::vec4 p_hom = *projmatrix * p_orig;

    float p_w = 1.0f / (p_hom.w + 0.0000001f);

    // vec4 to vec3 discards the last component
    // Might be better done with swizzling
    glm::vec3 p_proj = glm::vec3(p_hom) * p_w;
    float depth = p_proj.z;

    // Our NDC ranges from -1 1 for x and y and 0 to 1 for z
    bool isInside = glm::all(glm::lessThanEqual(p_proj, glm::vec3(1.f))) && glm::all(glm::greaterThanEqual(p_proj, glm::vec3(-1.f, -1.f, 0.f)));

    // Take two points that have a pixel wide distance, inverse project them and calculate the final distance
    if (isInside)
    {
        glm::vec3 p_proj_end(0.f);
        if (image_width > image_height)
            p_proj_end.x = 2.f / image_width;
        else
            p_proj_end.y = 2.f / image_height;
        p_proj_end.z = depth;

        glm::vec3 p_proj_start(0.f);
        p_proj_start.z = depth;

        glm::vec4 p_orig_end = *inverse_projmatrix * glm::vec4(p_proj_end, glm::vec1(1.f));

        p_w = 1.f / (p_orig_end.w + 0.0000001f);
        glm::vec3 p_orig_end_norm = glm::vec3(p_orig_end) * p_w;

        glm::vec4 p_orig_start = *inverse_projmatrix * glm::vec4(p_proj_start, glm::vec1(1.f));
        p_w = 1.f / (p_orig_start.w + 0.0000001f);
        glm::vec3 p_orig_start_norm = glm::vec3(p_orig_start) * p_w;

        glm::vec3 difference = p_orig_end_norm - p_orig_start_norm;
        pixel_sizes[idx] = min(pixel_sizes[idx], glm::length(difference));
    }
}

void transformCentersNDC(
    const int P,
    const glm::vec3 *centers,
    const glm::mat4 *projmatrix,
    const glm::mat4 *inverse_projmatrix,
    const int image_height,
    const int image_width,
    float *pixel_sizes)
{
    transformCentersNDCCUDA<<<(P + 255) / 256, 256>>>(
        P,
        centers,
        projmatrix,
        inverse_projmatrix,
        image_height,
        image_width,
        pixel_sizes);
}


__global__ void sphereEllipsoidIntersectionCUDA(
    const int P,
    const glm::vec3 *means3D,
    const glm::vec3 *scales,
    const glm::mat3 *R,
    const int *neighbours_indices,
    const float *sphere_radius,
    int *redundancy_values,
    bool *intersection_mask,
    const int knn)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    const glm::vec3 curr_xyz = means3D[idx];
    const int *curr_neighbours = neighbours_indices + idx * knn;
    bool *curr_intersection_mask = intersection_mask + idx * knn;
    const float curr_radius = sphere_radius[idx];
    // get neighbours
    for (int i = 0; i < knn; ++i)
    {
        // get scales of neighbour
        // get rotations of neighbour
        // intersection test if yes ++1
        int neighbour_id = curr_neighbours[i];
        glm::vec3 difference = curr_xyz - means3D[neighbour_id];
        glm::vec3 augmented_neighbour_scales = scales[neighbour_id] + glm::vec3(curr_radius);
        const glm::mat3 &neighbour_rotation = R[idx];

        // Change of basis: x_old = A x_new -> x_new = A^T x_old for orthonormal
        // equivalent to left multiplication
        glm::vec3 difference_neigbhours_coordinate_system = difference * neighbour_rotation;

        if (glm::dot(glm::pow(difference_neigbhours_coordinate_system, glm::vec3(2)), glm::vec3(1) / glm::pow(augmented_neighbour_scales, glm::vec3(2))) < 1)
        {
            redundancy_values[idx]++;
            curr_intersection_mask[i] = true;
        }
    }
    return;
}

void sphereEllipsoidIntersection(
    const int P,
    const glm::vec3 *means3D,
    const glm::vec3 *scales,
    const glm::mat3 *R,
    const int *neighbours_indices,
    const float *sphere_radius,
    int *redundancy_values,
    bool *intersection_mask,
    const int knn)
{
    sphereEllipsoidIntersectionCUDA<<<(P + 255) / 256, 256>>>(
        P,
        means3D,
        scales,
        R,
        neighbours_indices,
        sphere_radius,
        redundancy_values,
        intersection_mask,
        knn);
}


__global__ void buildRotationMatrixCUDA(
    const int P,
    const float4 *rotations,
    glm::mat3 *rotations_matrices)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    const float4 &q = rotations[idx];
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    // Compute rotation matrix from quaternion
    rotations_matrices[idx] = glm::mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y + r * z), 2.f * (x * z - r * y),
        2.f * (x * y - r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + r * x),
        2.f * (x * z + r * y), 2.f * (y * z - r * x), 1.f - 2.f * (x * x + y * y));
}

void buildRotationMatrix(
    const int P,
    const float4 *rotations,
    glm::mat3 *rotations_matrices)
{
    buildRotationMatrixCUDA<<<(P + 255) / 256, 256>>>(P, rotations, rotations_matrices);
}