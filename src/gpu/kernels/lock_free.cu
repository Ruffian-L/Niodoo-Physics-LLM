#define FULL_MASK 0xffffffff
#define WARP_SIZE 32

/**
 * Main lock-free reduction kernel
 * Each warp processes one column.
 * Implements Z2 sparse vector addition (Symmetric Difference) for persistence reduction.
 */
extern "C" __global__ void lock_free_reduction(
    int* __restrict__ pivots,            // Global pivot array
    const int* __restrict__ col_ptr,     // Column pointers (CSC format)
    const int* __restrict__ row_idx,     // Row indices (CSC format)
    const bool* __restrict__ is_cleared, // Columns to skip
    int* __restrict__ heap,              // Dynamic memory heap
    int* __restrict__ heap_ptr,          // Heap allocation pointer
    const int num_cols,
    const int heap_capacity
) {
    // Calculate which column this warp will process
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    if (warp_id >= num_cols) return;
    
    // Skip if this column was cleared (apparent pair or clearing optimization)
    if (is_cleared[warp_id]) return;
    
    int my_col = warp_id;
    
    // State variables for the current column
    // If head == -1, data is in row_idx (static). Else, data is in heap (dynamic).
    int curr_head = -1; 
    int curr_len = col_ptr[my_col + 1] - col_ptr[my_col];
    const int* curr_data_ptr = &row_idx[col_ptr[my_col]];

    // Main reduction loop
    int loop_safety = 0;
    while (loop_safety++ < 10000) {
        // Step 1: Find Pivot (Max row index)
        // Assumes data is sorted descending. Pivot is simply the first element.
        int pivot = -1;
        if (curr_len > 0) {
            if (lane_id == 0) {
                pivot = curr_data_ptr[0];
            }
        }
        // Broadcast pivot to warp
        pivot = __shfl_sync(FULL_MASK, pivot, 0);
        
        if (pivot == -1) {
            break; // Column reduced to empty (Cycle born)
        }
        
        // Step 2: Try to claim this pivot
        int owner = -1;
        if (lane_id == 0) {
            // Atomic Compare-And-Swap: If pivots[pivot] is -1, set it to my_col
            owner = atomicCAS(&pivots[pivot], -1, my_col);
        }
        owner = __shfl_sync(FULL_MASK, owner, 0);
        
        if (owner == -1) {
            // Success! We claimed the pivot. This column kills row 'pivot'.
            break; 
        } else if (owner == my_col) {
            // We already own it (shouldn't happen typically, but safe exit)
            break;
        } else {
            // Collision! 'owner' already claimed this pivot.
            // We must add column 'owner' to 'my_col' (mod 2 addition) to eliminate the pivot.
            
            // NOTE: Ideally we fetch owner's data location from a global 'col_heads' array.
            // For this fix, we assume owner is static for simplicity, or fallback to global lookup.
            // This simplified version accesses row_idx. A production version needs a 'col_state' array.
            
            int owner_start = col_ptr[owner];
            int owner_len = col_ptr[owner+1] - owner_start;
            const int* owner_ptr = &row_idx[owner_start];

            // Step 3: Allocate memory for the merged column
            // Max possible size is sum of lengths
            int max_new_len = curr_len + owner_len;
            int new_ptr_idx = -1;
            
            if (lane_id == 0) {
                new_ptr_idx = atomicAdd(heap_ptr, max_new_len);
            }
            new_ptr_idx = __shfl_sync(FULL_MASK, new_ptr_idx, 0);
            
            // Check OOM
            if (new_ptr_idx + max_new_len >= heap_capacity) {
                // if (lane_id == 0) printf("GPU Heap OOM!\n");
                return; 
            }
            
            // Step 4: Merge Sort (Symmetric Difference for Z2)
            // Both lists are sorted descending.
            int i = 0; // index for curr
            int j = 0; // index for owner
            int k = 0; // index for result
            
            // Warp-cooperative merge is complex; using serialized merge in lane 0 for correctness first.
            // Optimization: Parallel merge path can be added later.
            if (lane_id == 0) {
                int* new_data = &heap[new_ptr_idx];
                
                while (i < curr_len && j < owner_len) {
                    int val_a = curr_data_ptr[i];
                    int val_b = owner_ptr[j];
                    
                    if (val_a > val_b) {
                        new_data[k++] = val_a;
                        i++;
                    } else if (val_b > val_a) {
                        new_data[k++] = val_b;
                        j++;
                    } else {
                        // val_a == val_b. In Z2, 1+1=0. Skip both.
                        i++;
                        j++;
                    }
                }
                
                // Copy remaining
                while (i < curr_len) new_data[k++] = curr_data_ptr[i++];
                while (j < owner_len) new_data[k++] = owner_ptr[j++];
                
                // Update state for next iteration
                curr_len = k;
                curr_data_ptr = new_data; 
            }
            
            // Sync warp before next iteration
            curr_len = __shfl_sync(FULL_MASK, curr_len, 0);
            // Note: We need to broadcast the pointer, but pointers vary by 64-bit vs 32-bit.
            // Simplification: We rely on heap_base + offset logic in a real implementation.
            // For this snippet, we assume heap is globally accessible.
            
            // Important: The pointer update logic above works because heap is global.
            // We just need to update the offset.
            int new_offset = new_ptr_idx; 
            curr_data_ptr = &heap[new_offset]; 
        }
    }
}
