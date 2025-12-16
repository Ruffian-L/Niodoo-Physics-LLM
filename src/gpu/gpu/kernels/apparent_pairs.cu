/**
 * CUDA kernel for identifying apparent pairs in persistent homology
 * An apparent pair is a simplex-cofacet pair that can be matched without global reduction
 * This pre-processing step eliminates ~90% of columns in typical Rips complexes
 */

extern "C" __global__ void find_apparent_pairs(
    const int* __restrict__ col_ptr,     // CSC column pointers
    const int* __restrict__ row_idx,     // CSC row indices  
    int* __restrict__ apparent_pairs,    // Output: apparent_pairs[i] = j means (i,j) is a pair
    const int num_cols
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_cols) return;
    
    // Check if this column has exactly one entry (a cofacet)
    const int col_start = col_ptr[tid];
    const int col_end = col_ptr[tid + 1];
    const int col_nnz = col_end - col_start;
    
    if (col_nnz == 1) {
        // This simplex has exactly one cofacet
        const int cofacet_idx = row_idx[col_start];
        
        // Try to claim this as an apparent pair
        // If cofacet_idx hasn't been paired yet, pair it with tid
        atomicCAS(&apparent_pairs[cofacet_idx], -1, tid);
    }
}

/**
 * Mark columns that are part of apparent pairs as cleared
 * This prevents them from being processed in the main reduction
 */
extern "C" __global__ void mark_apparent_cleared(
    const int* __restrict__ apparent_pairs,
    bool* __restrict__ is_cleared,
    const int num_cols
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_cols) return;
    
    if (apparent_pairs[tid] >= 0) {
        // This column is part of an apparent pair
        is_cleared[tid] = true;
        is_cleared[apparent_pairs[tid]] = true;
    }
}
