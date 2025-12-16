extern "C" __global__ void compute_distances(
    const float* points, // flattened [x,y,z, x,y,z...]
    unsigned char* adj,  // flattened N*N
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
    
    if (dist_sq <= threshold * threshold) {
        adj[idx] = 1;
        adj[j * n + i] = 1; // Symmetric write
    }
}










