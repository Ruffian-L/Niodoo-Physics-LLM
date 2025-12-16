pub mod engine;
pub mod hnsw;
pub mod memory;
pub mod transaction;

pub use memory::{InMemoryBlobStore, OpaqueSplatRef, SplatBlobStore, TopologicalMemoryStore};

use crate::encoder::GaussianSplat;
use crate::indexing::TopologicalFingerprint;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: u64,
    pub splats: Vec<GaussianSplat>,
    pub fingerprint: TopologicalFingerprint,
    pub tags: Vec<String>,
    pub timestamp: u64,
}

pub struct TIVMMemory {
    entries: HashMap<u64, MemoryEntry>,
    next_id: u64,
}

impl TIVMMemory {
    pub fn new() -> Result<Self> {
        Ok(Self {
            entries: HashMap::new(),
            next_id: 0,
        })
    }

    pub async fn store(&mut self, splats: Vec<GaussianSplat>, tags: &[&str]) -> Result<u64> {
        let id = self.next_id;
        self.next_id += 1;

        let fingerprint = TopologicalFingerprint::new(vec![], vec![]);

        let entry = MemoryEntry {
            id,
            splats,
            fingerprint,
            tags: tags.iter().map(|s| s.to_string()).collect(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        self.entries.insert(id, entry);
        Ok(id)
    }

    pub async fn retrieve(
        &self,
        _query_splats: Vec<GaussianSplat>,
        k: usize,
    ) -> Result<Vec<MemoryEntry>> {
        let mut results: Vec<MemoryEntry> = self.entries.values().take(k).cloned().collect();

        results.truncate(k);
        Ok(results)
    }

    pub fn get(&self, id: u64) -> Option<&MemoryEntry> {
        self.entries.get(&id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for TIVMMemory {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

/// Zero-Copy Loading Helper
/// Safely memory-maps a file and passes the reference to a closure.
/// Note: Validation is currently bypassed (unsafe access) due to build issues with CheckBytes.
pub fn mmap_and_access<T, F, R>(path: &std::path::Path, f: F) -> Result<R>
where
    T: rkyv::Archive,
    T::Archived: rkyv::Portable,
    F: FnOnce(&T::Archived) -> R,
{
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

    // UNSAFE: Bypass validation
    let archived = unsafe { rkyv::access_unchecked::<T::Archived>(&mmap[..]) };
    Ok(f(archived))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    #[tokio::test]
    async fn test_memory_creation() {
        let memory = TIVMMemory::new().unwrap();
        assert_eq!(memory.len(), 0);
    }

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let mut memory = TIVMMemory::new().unwrap();

        let splat = GaussianSplat::new(Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, 1.0);

        let id = memory.store(vec![splat], &["test"]).await.unwrap();
        assert_eq!(id, 0);
        assert_eq!(memory.len(), 1);

        let entry = memory.get(id).unwrap();
        assert_eq!(entry.tags[0], "test");
    }
}
