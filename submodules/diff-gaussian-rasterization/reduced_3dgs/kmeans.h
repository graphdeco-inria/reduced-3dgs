#ifndef KMEANS_H_INCLUDED
#define KMEANS_H_INCLUDED

// This function finds the centroid value based on the points that are
// classified as belonding to the respective class
void updateCenters(
    const float *values,
    const int *ids,
    float *centers,
    int *center_sizes,
    const int n_values,
    const int n_centers);

// This function finds the closest centroid for each point
void updateIds(
    const float *values,
    int *ids,
    const float *centers,
    const int n_values,
    const int n_centers);

#endif