extern "C" __global__ void pairwise_distance(
    const float* points,
    float* distances,
    int num_points,
    int dims
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= num_points || j >= num_points) {
        return;
    }

    float dist_sq = 0.0f;
    for (int k = 0; k < dims; ++k) {
        float diff = points[i * dims + k] - points[j * dims + k];
        dist_sq += diff * diff;
    }

    distances[i * num_points + j] = sqrtf(dist_sq);
}
