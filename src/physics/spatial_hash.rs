use nalgebra::Vector3;
use std::collections::HashMap;

pub struct SpatialHashGrid {
    cell_size: f32,
    grid: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialHashGrid {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            grid: HashMap::new(),
        }
    }

    // CRITICAL FIX: Reuse memory instead of reallocating
    pub fn clear(&mut self) {
        for bucket in self.grid.values_mut() {
            bucket.clear();
        }
    }

    pub fn insert(&mut self, pos: &Vector3<f32>, id: usize) {
        let key = (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
            (pos.z / self.cell_size).floor() as i32,
        );
        self.grid.entry(key).or_default().push(id);
    }

    pub fn query_radius(&self, pos: &Vector3<f32>, radius: f32) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let center_key = (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
            (pos.z / self.cell_size).floor() as i32,
        );

        let range = (radius / self.cell_size).ceil() as i32;

        for x in -range..=range {
            for y in -range..=range {
                for z in -range..=range {
                    let key = (center_key.0 + x, center_key.1 + y, center_key.2 + z);
                    if let Some(bucket) = self.grid.get(&key) {
                        neighbors.extend(bucket);
                    }
                }
            }
        }
        neighbors
    }
}
