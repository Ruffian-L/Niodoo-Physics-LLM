use candle_core::{Result, Tensor};

pub struct AntigravityOp;

impl AntigravityOp {
    /// Applies the force calculation using Matrix Broadcasting (O(N^2)).
    /// Supports both CPU and CUDA via Candle.
    pub fn calculate_forces(
        positions: &Tensor,
        charges: &Tensor,
        masses: &Tensor,
        g_sem: f32,
    ) -> Result<Tensor> {
        let n = positions.dim(0)?;
        // 1. Batching to avoid OOM
        // N=17000^2 is too big (3GB+ per tensor). We chunk 'i'.
        let chunk_size = 16;
        let mut force_chunks = Vec::new();

        for i in (0..n).step_by(chunk_size) {
            let len = (n - i).min(chunk_size);

            // Slice current batch [Batch, 3]
            let pos_chunk = positions.narrow(0, i, len)?;
            let chg_chunk = charges.narrow(0, i, len)?;
            let mass_chunk = masses.narrow(0, i, len)?;

            // Prepare Broadcast [Batch, 1, 3] vs [1, N, 3]
            let pos_i = pos_chunk.unsqueeze(1)?;
            let pos_j = positions.unsqueeze(0)?;

            // diff: r_ij = pos_j - pos_i
            let diff = pos_j.broadcast_sub(&pos_i)?; // [Batch, N, 3]

            // Distance Squared
            let dist_sq = diff.sqr()?.sum_keepdim(2)?; // [Batch, N, 1]
            let soft_dist_sq = (dist_sq + 0.1)?; // Significant softening (10cm) for stability
            let dist = soft_dist_sq.sqrt()?;

            // Charge Interaction
            let chg_i = chg_chunk.unsqueeze(1)?;
            let chg_j = charges.unsqueeze(0)?;
            let chg_prod = (chg_i.broadcast_mul(&chg_j))?.sum_keepdim(2)?; // [Batch, N, 1]

            // Mass Product
            let m_i = mass_chunk.reshape((len, 1, 1))?;
            let m_j = masses.reshape((1, n, 1))?;
            let m_prod = m_i.broadcast_mul(&m_j)?;

            // Force Magnitude
            let g_tensor = Tensor::new(g_sem, positions.device())?
                .reshape((1, 1, 1))?
                .broadcast_as(m_prod.shape())?;

            let force_mag = ((g_tensor * m_prod)? / soft_dist_sq)?;

            // Direction
            let dir = diff.broadcast_div(&dist)?;

            // Sign
            let sign = chg_prod.sign()?;

            // force = dir * mag * sign
            let force_matrix = dir.broadcast_mul(&force_mag)?.broadcast_mul(&sign)?;

            // Sum Forces [Batch, N, 3] -> [Batch, 3]
            let batch_force = force_matrix.sum(1)?;
            force_chunks.push(batch_force);
        }

        // Concatenate chunks
        let total_forces = Tensor::cat(&force_chunks, 0)?;

        Ok(total_forces)
    }
}
