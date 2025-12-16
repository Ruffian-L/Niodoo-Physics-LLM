// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

extern "C" __device__ int printf(const char* format, ...);

__device__ int get_max_row(const int* data, int len) {
    // Assumes sorted descending.
    if (len == 0) return -1;
    return data[0];
}

// -----------------------------------------------------------------------------
// Kernel 1: Apparent Pairs
// -----------------------------------------------------------------------------
// Identifies simplex-cofacet pairs (sigma, tau) where tau is the ONLY cofacet of sigma.
// This is a pre-processing step to reduce matrix density.

extern "C" __global__ void apparent_pairs_kernel(
    const int* col_ptr,
    const int* row_idx,
    int* pivots,      // Output: pivots[row] = col (if paired)
    int* is_cleared, // Output: is_cleared[col] = 1 (if paired)
    int num_cols
) {
    // Placeholder: In a real implementation, we need the coboundary matrix.
    // For now, this kernel does nothing, leaving all columns to be reduced by the lock-free solver.
    // This is correct but slower.
}

// -----------------------------------------------------------------------------
// Parallel Merge Helpers
// -----------------------------------------------------------------------------

__device__ int binary_search_desc(const int* data, int len, int val) {
    int l = 0;
    int r = len;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (data[mid] > val) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

__device__ int binary_search_desc_strict(const int* data, int len, int val) {
    int l = 0;
    int r = len;
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (data[mid] >= val) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

__device__ int parallel_merge(int* dest, const int* A, int lenA, const int* B, int lenB) {
    int tid = threadIdx.x % 32;
    int total_len = lenA + lenB;

    // Process A
    for (int i = tid; i < lenA; i += 32) {
        int val = A[i];
        int rankB = binary_search_desc(B, lenB, val);
        dest[i + rankB] = val;
    }
    
    // Process B
    for (int i = tid; i < lenB; i += 32) {
        int val = B[i];
        int rankA = binary_search_desc_strict(A, lenA, val);
        dest[rankA + i] = val;
    }
    
    __syncwarp();

    // 3. Mark Duplicates (Parallel)
    // dest is sorted descending. Duplicates are adjacent.
    for (int idx = tid; idx < total_len - 1; idx += 32) {
        if (dest[idx] == dest[idx + 1]) {
            dest[idx] = -1;
            dest[idx + 1] = -1;
        }
    }
    __syncwarp();

    // 4. Compact (Parallel)
    int write_idx = 0;
    
    for (int base = 0; base < total_len; base += 32) {
        int idx = base + tid;
        int val = (idx < total_len) ? dest[idx] : -1;
        int keep = (val != -1);
        
        unsigned mask = __ballot_sync(0xFFFFFFFF, keep);
        int local_rank = __popc(mask & ((1 << tid) - 1));
        
        if (keep) {
            dest[write_idx + local_rank] = val;
        }
        
        write_idx += __popc(mask);
    }
    
    return write_idx;
}

// -----------------------------------------------------------------------------
// Kernel 2: Lock-Free Reduction
// -----------------------------------------------------------------------------

extern "C" __global__ void lock_free_kernel(
    int* pivots,           // [num_rows] -1 if empty, else col_idx
    const int* col_ptr,    // [num_cols + 1]
    const int* row_idx,    // [nnz]
    int num_cols,
    int num_rows,
    // Heap for fill-in
    int* heap_data,        // Massive array for new columns
    int* heap_head,        // Atomic counter
    int heap_capacity,
    // Current column state
    int* col_heads,        // [num_cols] index into heap_data OR -1 if original
    int* col_lens          // [num_cols] length of column
) {
    // Warp-per-column strategy
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= num_cols) return;

    int my_col_idx = warp_id;
    
    // Initialize column state
    int curr_head = col_heads[my_col_idx];
    int curr_len = col_lens[my_col_idx];
    
    // Pointer to the data of the current column
    const int* my_data_ptr;
    if (curr_head == -1) {
        // Original data
        my_data_ptr = &row_idx[col_ptr[my_col_idx]];
    } else {
        // Heap data
        my_data_ptr = &heap_data[curr_head];
    }

    int loop_count = 0;
    while (true) {
        loop_count++;
        if (loop_count > 10000) {
            if (lane_id == 0) printf("Col %d stuck in loop\n", my_col_idx);
            break;
        }
        // 1. Find Pivot
        // We assume sorted descending, so pivot is the first element.
        int pivot = -1;
        if (curr_len > 0) {
            // Only lane 0 reads, then broadcast
            if (lane_id == 0) {
                pivot = my_data_ptr[0];
            }
        }
        pivot = __shfl_sync(0xFFFFFFFF, pivot, 0);

        if (pivot == -1) {
            // Column is empty
            break;
        }

        // 2. Attempt to claim pivot
        int owner = -1;
        if (lane_id == 0) {
            // atomicCAS(address, compare, val)
            owner = atomicCAS(&pivots[pivot], -1, my_col_idx);
        }
        owner = __shfl_sync(0xFFFFFFFF, owner, 0);

        if (owner == -1) {
            // Success! We claimed the pivot.
            break;
        } else if (owner == my_col_idx) {
            // We already own it (shouldn't happen in this loop structure unless re-entry)
            break;
        } else {
            // Failure! Collision with 'owner'.
            // We must add column 'owner' to 'my_col'.
            
            // Get owner's data
            int owner_head = col_heads[owner];
            int owner_len = col_lens[owner];
            const int* owner_data_ptr;
            
            if (owner_head == -1) {
                owner_data_ptr = &row_idx[col_ptr[owner]];
            } else {
                owner_data_ptr = &heap_data[owner_head];
            }
            
            // 3. Merge (Add) Columns
            int new_capacity = curr_len + owner_len;
            int new_head_idx = -1;
            
            if (lane_id == 0) {
                new_head_idx = atomicAdd(heap_head, new_capacity);
            }
            new_head_idx = __shfl_sync(0xFFFFFFFF, new_head_idx, 0);
            
            if (new_head_idx + new_capacity >= heap_capacity) {
                // OOM
                return; 
            }
            
            int* new_data_ptr = &heap_data[new_head_idx];
            
            // Parallel Merge
            int new_len = parallel_merge(new_data_ptr, my_data_ptr, curr_len, owner_data_ptr, owner_len);
            
            // Broadcast new_len (parallel_merge returns same value on all threads)
            new_len = __shfl_sync(0xFFFFFFFF, new_len, 0);
            
            // Update state
            if (lane_id == 0) {
                col_heads[my_col_idx] = new_head_idx;
                col_lens[my_col_idx] = new_len;
            }
            
            curr_head = new_head_idx;
            curr_len = new_len;
            my_data_ptr = new_data_ptr;
            
            __threadfence(); 
        }
    }
}
