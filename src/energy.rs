use crate::sheaf::SheafGraph;
use candle_core::{Module, Result};

pub fn compute_sheaf_energy(graph: &SheafGraph) -> Result<f32> {
    let mut total_energy = 0.0;

    // Iterate over all edges.
    // We iterate restrictions and only process if u < v to avoid double counting.
    for ((u, v), restriction_uv) in &graph.restrictions {
        // Only process each edge once
        if *u >= *v {
            continue;
        }

        let restriction_vu = match graph.restrictions.get(&(*v, *u)) {
            Some(r) => r,
            None => continue, // Should not happen if graph is consistent
        };

        let x_u = match graph.nodes.get(u) {
            Some(n) => &n.x,
            None => continue,
        };
        let x_v = match graph.nodes.get(v) {
            Some(n) => &n.x,
            None => continue,
        };

        let proj_u = restriction_uv.linear.forward(x_u)?;
        let proj_v = restriction_vu.linear.forward(x_v)?;

        let diff = (proj_u - proj_v)?;
        let sq_norm = diff.sqr()?.sum_all()?.to_scalar::<f32>()?;

        total_energy += sq_norm;
    }

    Ok(total_energy)
}
