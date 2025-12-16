use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module};
use std::collections::HashMap;

#[derive(Clone)]
pub struct Stalk {
    pub x: Tensor,
}

#[derive(Clone)]
pub struct RestrictionMap {
    pub linear: Linear,
}

pub struct SheafGraph {
    pub nodes: HashMap<u64, Stalk>,
    // Stores B_uv for edge u->v. Key is (u, v).
    pub restrictions: HashMap<(u64, u64), RestrictionMap>,
    pub adjacency: HashMap<u64, Vec<u64>>,
    pub device: Device,
}

impl SheafGraph {
    pub fn new(device: Device) -> Self {
        Self {
            nodes: HashMap::new(),
            restrictions: HashMap::new(),
            adjacency: HashMap::new(),
            device,
        }
    }

    pub fn add_node(&mut self, id: u64, x: Tensor) {
        self.nodes.insert(id, Stalk { x });
        self.adjacency.entry(id).or_default();
    }

    pub fn add_edge(&mut self, u: u64, v: u64, b_uv: Linear, b_vu: Linear) {
        self.restrictions
            .insert((u, v), RestrictionMap { linear: b_uv });
        self.restrictions
            .insert((v, u), RestrictionMap { linear: b_vu });

        self.adjacency.entry(u).or_default().push(v);
        self.adjacency.entry(v).or_default().push(u);
    }

    pub fn diffusion_step(&mut self, alpha: f64) -> Result<()> {
        let mut updates: HashMap<u64, Tensor> = HashMap::new();

        // Calculate updates for all nodes
        for (&u, stalk_u) in &self.nodes {
            let neighbors = match self.adjacency.get(&u) {
                Some(n) => n,
                None => continue,
            };

            let mut gradient_sum = Tensor::zeros_like(&stalk_u.x)?;

            for &v in neighbors {
                let stalk_v = self.nodes.get(&v).unwrap();

                let b_uv = &self.restrictions.get(&(u, v)).unwrap().linear;
                let b_vu = &self.restrictions.get(&(v, u)).unwrap().linear;

                // B_uv * x_u
                let proj_u = b_uv.forward(&stalk_u.x)?;
                // B_vu * x_v
                let proj_v = b_vu.forward(&stalk_v.x)?;

                // diff = B_uv x_u - B_vu x_v
                let diff = (proj_u - proj_v)?;

                // Back project: B_uv^T * diff
                // In candle, Linear computes x @ w.t() + b.
                // So the weight matrix W is (out_dim, in_dim).
                // To apply the transpose of the linear map (which is x @ W^T),
                // we want to multiply by W (not transposed).
                // diff is (1, 64). W is (64, 64).
                // diff @ W -> (1, 64).
                let back_proj = diff.matmul(b_uv.weight())?;

                gradient_sum = (gradient_sum + back_proj)?;
            }

            // x_new = x_old - alpha * sum(...)
            let update = (gradient_sum * alpha)?;
            updates.insert(u, update);
        }

        // Apply updates
        for (u, update) in updates {
            let node = self.nodes.get_mut(&u).unwrap();
            node.x = (node.x.clone() - update)?;
        }

        Ok(())
    }
}
