#[cfg(test)]
mod tests {
    use crate::gpu::lophat::CudaLockFreeAlgo;
    use lophat::algorithms::DecompositionAlgo;
    use cudarc::driver::CudaDevice;
    use std::sync::Arc;

    #[test]
    fn test_gpu_lock_free_simple() {
        if !crate::gpu::cuda_available() {
            println!("Skipping GPU test: CUDA not available");
            return;
        }

        let dev = CudaDevice::new(0).expect("Failed to get CUDA device");
        // We can use new directly, or init via trait if we want to test trait fully.
        // But new is fine.
        let algo = CudaLockFreeAlgo::new(dev);

        // Simple triangle boundary matrix
        // 0: []
        // 1: []
        // 2: []
        // 3: [0, 1]
        // 4: [1, 2]
        // 5: [0, 2]
        // 6: [3, 4, 5] (boundary of triangle 012)
        
        let cols = vec![
            vec![], 
            vec![], 
            vec![], 
            vec![1, 0], // sorted descending
            vec![2, 1], 
            vec![2, 0], 
            vec![5, 4, 3]
        ];

        use lophat::columns::VecColumn;
        let cols_iter = cols.into_iter().map(|c| {
            let pivot = c.iter().max().cloned().unwrap_or(0);
            VecColumn::from((pivot, c))
        });
        let decomp = algo.add_cols(cols_iter).decompose();
        
        let pivots = decomp.pivots;
        println!("Pivots: {:?}", pivots);
        
        // Expected:
        // 0,1,2 are empty.
        // 3 reduces to pivot 1? Or 0?
        // Standard reduction:
        // 3: low=1. Pivot[1] = 3.
        // 4: low=2. Pivot[2] = 4.
        // 5: low=2. Collision with 4. Add 4 to 5.
        //    5 = [2,0] + [2,1] = [1,0].
        //    low=1. Collision with 3. Add 3 to 5.
        //    5 = [1,0] + [1,0] = [].
        //    5 is empty.
        // 6: low=5. Pivot[5] = 6? No, 5 is empty. 
        //    Wait, 5 was reduced to empty. So 5 is not a pivot.
        //    6 has boundary [5,4,3].
        //    5 is empty? No, column 5 is empty. Row 5 is not.
        //    Boundary of 6 is 3+4+5.
        //    In matrix terms:
        //    Col 3 has pivot 1.
        //    Col 4 has pivot 2.
        //    Col 5 reduces to 0.
        //    Col 6: low=5.
        //    Is 5 a pivot? No.
        //    So Pivot[5] = 6.
        
        // Resulting pivots array (size num_rows=6? or 7?):
        // Indices: 0 1 2 3 4 5
        // Values: -1 3 4 -1 -1 6
        
        // Let's check.
        assert_eq!(pivots[1], 3);
        assert_eq!(pivots[2], 4);
        assert_eq!(pivots[5], 6);
    }
}
