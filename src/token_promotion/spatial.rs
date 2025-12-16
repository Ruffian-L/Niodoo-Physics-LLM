//! Niodoo-TCS: Topological Cognitive System
//! Copyright (c) 2025 Jason Van Pham

use std::collections::HashMap;

use crate::memory_system::MemorySystem;

/// Lightweight spatial hash for grouping memory spheres by proximity.
#[derive(Debug, Default)]
pub struct SpatialHash {
    cell_size: f32,
    buckets: HashMap<(i32, i32, i32), Vec<u64>>, // u64 is payload_id
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cell_size,
            buckets: HashMap::new(),
        }
    }

    /// Map a 3D position to an integer bucket.
    pub fn position_to_bucket(&self, position: &[f32; 3]) -> (i32, i32, i32) {
        let to_bucket = |value: f32| -> i32 { (value / self.cell_size).floor() as i32 };
        (
            to_bucket(position[0]),
            to_bucket(position[1]),
            to_bucket(position[2]),
        )
    }

    /// Populate the hash from the current memory spheres.
    pub fn rebuild_from_memory(&mut self, memory_system: &MemorySystem) {
        self.buckets.clear();
        // Iterate geometries and map to buckets
        // We assume geometries and semantics are aligned by index, and payload_id is in semantics
        for (idx, geom) in memory_system.storage.geometries.iter().enumerate() {
            if idx < memory_system.storage.payload_ids.len() {
                let payload_id = memory_system.storage.payload_ids[idx];
                let pos = [geom.position[0], geom.position[1], geom.position[2]];
                let bucket = self.position_to_bucket(&pos);
                self.buckets.entry(bucket).or_default().push(payload_id);
            }
        }
    }

    /// Retrieve references to the identifiers stored in a bucket.
    pub fn bucket(&self, bucket: &(i32, i32, i32)) -> Option<&[u64]> {
        self.buckets.get(bucket).map(|ids| ids.as_slice())
    }
}
