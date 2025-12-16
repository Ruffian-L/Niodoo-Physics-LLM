use super::MatrixDecomposer;
use std::collections::BTreeSet; // Sorted set for easy Symmetric Difference

pub struct CpuDecomposer {
    /// The R matrix (reduced boundary matrix).
    /// Stored as sparse columns (sorted vectors of row indices).
    matrix: Vec<BTreeSet<usize>>,
    /// Lookup table: low_row_index -> col_index
    /// Maps a pivot (row) to the column that kills it.
    pivots: Vec<Option<usize>>,
}

impl CpuDecomposer {
    pub fn new(boundary_matrix: Vec<Vec<usize>>) -> Self {
        let _num_cols = boundary_matrix.len();
        let max_row = boundary_matrix.iter()
            .flatten()
            .max()
            .copied()
            .unwrap_or(0);

        // Convert input Vec<Vec> to BTreeSet for easier set ops
        let matrix: Vec<BTreeSet<usize>> = boundary_matrix
            .into_iter()
            .map(|col| col.into_iter().collect())
            .collect();

        Self {
            matrix,
            pivots: vec![None; max_row + 1],
        }
    }
}

impl MatrixDecomposer for CpuDecomposer {
    fn get_pivot(&self, col_idx: usize) -> Option<usize> {
        // In PH, the "pivot" is usually the maximum index (the "youngest" simplex)
        self.matrix[col_idx].iter().next_back().copied()
    }

    fn add_entries(&mut self, target_idx: usize, source_idx: usize) {
        // Column Addition in Z2 is Symmetric Difference (XOR)
        // We have to clone source to avoid borrowing issues if not careful, 
        // but BTreeSet makes union/diff easy.
        
        let source_col = self.matrix[source_idx].clone();
        let target_col = &mut self.matrix[target_idx];

        for row in source_col {
            if target_col.contains(&row) {
                target_col.remove(&row); // 1 + 1 = 0
            } else {
                target_col.insert(row);  // 0 + 1 = 1
            }
        }
    }

    fn get_r_col(&self, col_idx: usize) -> Vec<usize> {
        self.matrix[col_idx].iter().copied().collect()
    }

    /// Standard PH Reduction Algorithm
    fn reduce(&mut self) {
        let num_cols = self.matrix.len();

        for j in 0..num_cols {
            // While R[j] is not empty
            while let Some(pivot_row) = self.get_pivot(j) {
                // Check if this pivot is already "owned" by a previous column
                if let Some(k) = self.pivots[pivot_row] {
                    // If owned by k, we must add column k to j to eliminate the pivot
                    self.add_entries(j, k);
                } else {
                    // Pivot is unique! We claim it.
                    self.pivots[pivot_row] = Some(j);
                    break; // Column j is now reduced
                }
            }
        }
    }
}






