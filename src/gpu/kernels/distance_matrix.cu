extern "C" __global__ void compute_distances(
    const float* points, // flattened [x,y,z, x,y,z...]
    float* dists,        // Output: flattened N*N float distances
    int n,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;

    int i = idx / n;
    int j = idx % n;

    if (i >= j) return; // Symmetric, only calc upper triangle

    float dx = points[i*3 + 0] - points[j*3 + 0];
    float dy = points[i*3 + 1] - points[j*3 + 1];
    float dz = points[i*3 + 2] - points[j*3 + 2];

    float dist_sq = dx*dx + dy*dy + dz*dz;
    float dist = sqrtf(dist_sq);
    
    // We output the REAL distance if within threshold
    // If outside, we output infinity (or max) to effectively cull it later
    if (dist <= threshold) {
        dists[idx] = dist;
        dists[j * n + i] = dist; // Symmetric write
    } else {
        dists[idx] = 999999.0f; // Infinity sentinel
        dists[j * n + i] = 999999.0f;
    }
}
