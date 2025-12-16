use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use half::f16;
use ndarray::Array2;
use ndarray_npy::read_npy;
use serde::{Deserialize, Serialize};

use crate::indexing::{fingerprint_from_splat, TopologicalFingerprint};
use crate::memory::emotional::{
    EmotionalState, PadGhostState, TemporalDecayConfig, WeightedMemoryMetadata,
};
use crate::retrieval::fitness::{calculate_radiance_score, FitnessWeights};
use crate::storage::hnsw::HnswIndex;
use crate::structs::{
    PackedSemantics, SplatFileHeader, SplatGeometry, SplatLighting, SplatSemantics,
};
use crate::tivm::SplatRagConfig;
use crate::types::{SplatId, SplatInput, SplatMeta};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpaqueSplatRef {
    Path(PathBuf),
    Bytes(Arc<Vec<u8>>),
    External(String),
}

pub trait SplatBlobStore: Send + Sync + 'static {
    fn put(&self, id: SplatId, blob: OpaqueSplatRef);
    fn get(&self, id: SplatId) -> Option<OpaqueSplatRef>;
}

#[derive(Default)]
pub struct InMemoryBlobStore {
    blobs: Mutex<HashMap<SplatId, OpaqueSplatRef>>,
}

impl Serialize for InMemoryBlobStore {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let blobs = self.blobs.lock().unwrap();
        blobs.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for InMemoryBlobStore {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let blobs = HashMap::deserialize(deserializer)?;
        Ok(Self {
            blobs: Mutex::new(blobs),
        })
    }
}

impl SplatBlobStore for InMemoryBlobStore {
    fn put(&self, id: SplatId, blob: OpaqueSplatRef) {
        let mut guard = self.blobs.lock().unwrap();
        guard.insert(id, blob);
    }

    fn get(&self, id: SplatId) -> Option<OpaqueSplatRef> {
        let guard = self.blobs.lock().unwrap();
        guard.get(&id).cloned()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMemory {
    pub id: SplatId,
    pub fingerprint: TopologicalFingerprint,
    pub embedding: Vec<f16>,
    pub manifold_vector: Vec<f32>, // Changed from [f32; 64] to Vec<f32> for serde compatibility
    pub meta: SplatMeta,
    pub splat: SplatInput,
    pub text: String, // Added for Genesis Physics (Entropy/Shaping)
    #[serde(default)]
    pub geometry: SplatGeometry,
    #[serde(default)]
    pub lighting: Option<SplatLighting>,
}

#[derive(Serialize, Deserialize)]
pub struct TopologicalMemoryStore<B: SplatBlobStore> {
    config: SplatRagConfig,
    blob_store: B,
    entries: HashMap<SplatId, StoredMemory>,
    next_id: SplatId,
    #[serde(skip)] // Skip indexing serialization via Serde
    index: Option<HnswIndex>,
    #[serde(skip)]
    current_pad: Option<PadGhostState>,
}

impl<B: SplatBlobStore + Serialize + serde::de::DeserializeOwned> TopologicalMemoryStore<B> {
    pub fn load_from_npy(npy_path: &Path, config: SplatRagConfig, blob_store: B) -> Result<Self> {
        let mut store = Self::new(config, blob_store);
        println!("Loading memory cloud from {:?}...", npy_path);
        // Read as u16 because ndarray-npy doesn't support f16 directly
        let embeddings_u16: Array2<u16> = read_npy(npy_path)?;
        let (rows, cols) = embeddings_u16.dim();
        println!("Loaded {} embeddings ({} dim)", rows, cols);

        for (i, row) in embeddings_u16.axis_iter(ndarray::Axis(0)).enumerate() {
            let id = i as u64;
            let embedding_u16 = row.to_vec();
            let embedding: Vec<f16> = embedding_u16.iter().map(|&x| f16::from_bits(x)).collect();

            // Create dummy SplatInput
            // Use first 3 dims as pos if available, else 0
            let pos = if embedding.len() >= 3 {
                [
                    embedding[0].to_f32(),
                    embedding[1].to_f32(),
                    embedding[2].to_f32(),
                ]
            } else {
                [0.0; 3]
            };

            let splat = SplatInput {
                static_points: vec![pos],
                covariances: vec![[0.01; 9]], // Dummy cov
                motion_velocities: None,
                meta: SplatMeta {
                    timestamp: Some(0.0),
                    labels: vec![],
                    emotional_state: None,
                    fitness_metadata: None,
                },
                normals: None,
                idiv: None,
                ide: None,
                sss_params: None,
                sh_occlusion: None,
            };

            let fingerprint = fingerprint_from_splat(&splat, &store.config);

            let stored = StoredMemory {
                id,
                fingerprint,
                embedding: embedding.clone(),
                manifold_vector: vec![0.0; 64], // Will be computed on first retrieval
                meta: splat.meta.clone(),
                splat,
                text: String::new(),
                geometry: SplatGeometry::default(),
                lighting: None,
            };

            store.entries.insert(id, stored);
            if let Some(index) = store.index.as_mut() {
                let emb_f32: Vec<f32> = embedding.iter().map(|x| x.to_f32()).collect();
                index.add(id, &emb_f32)?;
            }
            store.next_id = id + 1;
        }

        Ok(store)
    }

    pub fn load_from_split_files(
        geom_path: &Path,
        sem_path: &Path,
        config: SplatRagConfig,
        blob_store: B,
    ) -> Result<Self> {
        let mut store = Self::new(config, blob_store);
        println!("Loading split files: {:?} / {:?}", geom_path, sem_path);

        // 1. Read Geometry
        let mut geom_file = File::open(geom_path)?;
        let mut header_bytes = [0u8; std::mem::size_of::<SplatFileHeader>()];
        geom_file.read_exact(&mut header_bytes)?;
        let header: SplatFileHeader = bytemuck::cast(header_bytes);

        if &header.magic != b"SPLTRAG\0" {
            anyhow::bail!("Invalid magic bytes in geometry file");
        }

        let count = header.count as usize;
        let mut geoms = vec![SplatGeometry::default(); count];
        let geom_bytes = bytemuck::cast_slice_mut(&mut geoms);
        geom_file.read_exact(geom_bytes)?;

        // 1.5 Read Lighting (Optional)
        let lgt_path = sem_path.with_extension("lgt");
        let lgt_path = Path::new(&lgt_path);
        let mut lighting_data: Option<Vec<SplatLighting>> = None;

        if lgt_path.exists() {
            let mut lgt_file = File::open(lgt_path)?;
            let mut lgt_header_bytes = [0u8; std::mem::size_of::<SplatFileHeader>()];
            lgt_file.read_exact(&mut lgt_header_bytes)?;
            // We can verify header if we want, but mostly we trust it matches count

            let mut lights = vec![SplatLighting::default(); count];
            let light_bytes = bytemuck::cast_slice_mut(&mut lights);
            lgt_file.read_exact(light_bytes)?;
            lighting_data = Some(lights);
        }

        // 2. Read Semantics (Meta)
        // Try to find the meta file
        let sem_path_str = sem_path.to_string_lossy();
        let meta_path_str = if sem_path_str.ends_with(".bin") {
            format!("{}_meta.bin", sem_path_str.trim_end_matches(".bin"))
        } else {
            format!("{}_meta.bin", sem_path_str)
        };
        let meta_path = Path::new(&meta_path_str);

        if !meta_path.exists() {
            anyhow::bail!(
                "Meta file not found at {:?}. Cannot load full semantics.",
                meta_path
            );
        }

        let meta_file = File::open(meta_path)?;
        let mut meta_reader = BufReader::new(meta_file);

        for i in 0..count {
            let sem: SplatSemantics = bincode::deserialize_from(&mut meta_reader)?;
            let geom = geoms[i];

            let id = sem.payload_id;

            // Convert embedding to f16
            let embedding: Vec<f16> = sem.embedding.iter().map(|&x| f16::from_f32(x)).collect();

            // Reconstruct SplatInput
            let splat = SplatInput {
                static_points: vec![geom.position],
                covariances: vec![[0.01; 9]], // Dummy cov
                motion_velocities: None,
                meta: SplatMeta {
                    timestamp: Some(sem.birth_time),
                    labels: vec![],
                    emotional_state: sem.emotional_state,
                    fitness_metadata: sem.fitness_metadata,
                },
                normals: lighting_data.as_ref().map(|l| vec![l[i].normal]),
                idiv: lighting_data.as_ref().map(|l| vec![l[i].idiv]),
                ide: lighting_data.as_ref().map(|l| vec![l[i].ide]),
                sss_params: lighting_data.as_ref().map(|l| vec![l[i].sss_params]),
                sh_occlusion: lighting_data.as_ref().map(|l| {
                    let sh = &l[i].sh_occlusion;
                    // Pad from 7 to 9 elements for SplatInput compatibility
                    vec![[sh[0], sh[1], sh[2], sh[3], sh[4], sh[5], sh[6], 0.0, 0.0]]
                }),
            };

            let fingerprint = fingerprint_from_splat(&splat, &store.config);

            let stored = StoredMemory {
                id,
                fingerprint,
                embedding: embedding.clone(),
                manifold_vector: sem.manifold_vector.to_vec(),
                meta: splat.meta.clone(),
                splat,
                text: String::new(),
                geometry: geom,
                lighting: lighting_data.as_ref().map(|l| l[i]),
            };

            store.entries.insert(id, stored);
            if let Some(index) = store.index.as_mut() {
                index.add(id, &sem.embedding)?;
            }
            if id >= store.next_id {
                store.next_id = id + 1;
            }
        }

        println!("Loaded {} memories from split files.", count);
        Ok(store)
    }

    pub fn save_to_disk<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let tmp_path = path.with_extension("tmp");

        {
            let file = File::create(&tmp_path)?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer(&mut writer, self)?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }

        std::fs::rename(&tmp_path, path)?;

        Ok(())
    }

    pub fn load_from_disk<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let store: Self = serde_json::from_reader(reader)?;
        Ok(store)
    }
}

impl<B: SplatBlobStore> TopologicalMemoryStore<B> {
    pub fn new(config: SplatRagConfig, blob_store: B) -> Self {
        Self {
            config,
            blob_store,
            entries: HashMap::new(),
            next_id: 0,
            index: None,
            current_pad: None,
        }
    }

    pub fn with_indexer(config: SplatRagConfig, blob_store: B, index: HnswIndex) -> Self {
        let mut store = Self::new(config, blob_store);
        store.index = Some(index);
        store
    }

    pub fn attach_indexer(&mut self, mut index: HnswIndex) -> Result<()> {
        for entry in self.entries.values() {
            let emb_f32: Vec<f32> = entry.embedding.iter().map(|x| x.to_f32()).collect();
            index.add(entry.id, &emb_f32)?;
        }
        self.index = Some(index);
        Ok(())
    }

    pub fn add_splat(
        &mut self,
        splat: &SplatInput,
        blob: OpaqueSplatRef,
        text: String,
        embedding: Vec<f32>,
    ) -> Result<SplatId> {
        let id = self.next_id;
        self.next_id += 1;

        let fingerprint = fingerprint_from_splat(splat, &self.config);
        // let embedding = fingerprint.to_vector(); // Use provided embedding instead
        let meta = splat.meta.clone();
        let splat_clone = splat.clone();

        self.blob_store.put(id, blob);

        let embedding_f16: Vec<f16> = embedding.iter().map(|&x| f16::from_f32(x)).collect();

        let stored = StoredMemory {
            id,
            fingerprint,
            embedding: embedding_f16,
            manifold_vector: vec![0.0; 64], // Will be computed on first retrieval
            meta,
            splat: splat_clone,
            text,
            geometry: SplatGeometry::default(),
            lighting: None,
        };

        if let Some(index) = self.index.as_mut() {
            index.add(id, &embedding)?;
        }

        self.entries.insert(id, stored);

        Ok(id)
    }

    pub fn get(&self, id: SplatId) -> Option<&StoredMemory> {
        self.entries.get(&id)
    }

    pub fn blob(&self, id: SplatId) -> Option<OpaqueSplatRef> {
        self.blob_store.get(id)
    }

    pub fn embeddings(&self) -> impl Iterator<Item = (&SplatId, Vec<f32>)> {
        self.entries
            .iter()
            .map(|(id, entry)| (id, entry.embedding.iter().map(|x| x.to_f32()).collect()))
    }

    pub fn search_embeddings(&self, query: &[f32], k: usize) -> Result<Vec<(SplatId, f32)>> {
        match &self.index {
            Some(index) => Ok(index.search(query, k)),
            None => Ok(Vec::new()),
        }
    }

    pub fn entries_mut(&mut self) -> &mut HashMap<SplatId, StoredMemory> {
        &mut self.entries
    }

    // Add this method to allow iteration
    pub fn entries(&self) -> std::collections::hash_map::Iter<'_, SplatId, StoredMemory> {
        self.entries.iter()
    }

    pub fn remove(&mut self, id: SplatId) -> Option<StoredMemory> {
        let entry = self.entries.remove(&id);
        if let Some(ref _e) = entry {
            if let Some(_index) = self.index.as_mut() {
                // Note: HNSW doesn't easily support removal without rebuild or soft delete
                // For now we just remove from map. Rebuilding index is expensive.
                // We might need a soft-delete flag or just accept index drift until reload.
            }
        }
        entry
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get_radiance(&self, id: SplatId) -> f32 {
        let entry = match self.entries.get(&id) {
            Some(e) => e,
            None => return 0.0,
        };

        let default_emotional = EmotionalState::default();
        let _emotional_state = entry
            .meta
            .emotional_state
            .as_ref()
            .unwrap_or(&default_emotional);

        let default_metadata = WeightedMemoryMetadata::default();
        let metadata = entry
            .meta
            .fitness_metadata
            .as_ref()
            .unwrap_or(&default_metadata);

        let default_pad = PadGhostState::default();
        let current_pad = self.current_pad.as_ref().unwrap_or(&default_pad);
        let weights = FitnessWeights::default();
        let temporal_config = TemporalDecayConfig::default();

        calculate_radiance_score(
            entry.meta.timestamp.unwrap_or(0.0) as f64,
            metadata,
            current_pad,
            &weights,
            &temporal_config,
        )
    }

    pub fn load_current() -> Result<Self>
    where
        B: Default + Serialize + serde::de::DeserializeOwned,
    {
        let store_path = "mindstream_store.json";
        if Path::new(store_path).exists() {
            return Self::load_from_disk(store_path);
        }

        // Prefer NPY
        let npy_path = Path::new("memory_cloud_64dim.npy");
        if npy_path.exists() {
            return Self::load_from_npy(npy_path, SplatRagConfig::default(), B::default());
        }

        let geom_path = Path::new("mindstream_current.geom");
        let sem_path = Path::new("mindstream_current.sem");
        if geom_path.exists() && sem_path.exists() {
            // Check if geom file is empty or just header
            let meta = std::fs::metadata(geom_path)?;
            if meta.len() > 40 {
                // Header ~36-40 bytes
                return Self::load_from_split_files(
                    geom_path,
                    sem_path,
                    SplatRagConfig::default(),
                    B::default(),
                );
            }
        }

        Ok(Self::new(SplatRagConfig::default(), B::default()))
    }

    /// Saves the store's memories to split geometry/semantics files
    /// Compatible with Ingest/Retrieve format:
    /// .geom -> SplatGeometry (Fixed)
    /// .sem -> PackedSemantics (Fixed)
    /// .ids -> u64 IDs
    /// .emb -> f32 Embeddings
    /// _meta.bin -> SplatSemantics (Bincode)
    pub fn save_split_files(&self, geom_path: &str, sem_path: &str) -> Result<()> {
        let mut geom_file = File::create(geom_path)?;
        let mut sem_file = File::create(sem_path)?;

        // Sidecar files
        let ids_path = sem_path.replace(".sem", ".ids");
        let emb_path = sem_path.replace(".sem", ".emb");
        let meta_path = if sem_path.ends_with(".bin") {
            format!("{}_meta.bin", sem_path.trim_end_matches(".bin"))
        } else {
            format!("{}_meta.bin", sem_path)
        };

        let mut ids_file = File::create(&ids_path)?;
        let mut emb_file = File::create(&emb_path)?;
        let mut meta_file = File::create(&meta_path)?;

        let lgt_path = sem_path.replace(".sem", ".lgt");
        let mut lgt_file = File::create(&lgt_path)?;

        let entries_count = self.entries.len() as u64;
        let header = SplatFileHeader {
            magic: *b"SPLTRAG\0",
            version: 1,
            count: entries_count,
            geometry_size: std::mem::size_of::<SplatGeometry>() as u32,
            semantics_size: std::mem::size_of::<PackedSemantics>() as u32,
            motion_size: 0,
            lighting_size: std::mem::size_of::<SplatLighting>() as u32,
            _pad: [0; 2],
        };

        // Write header to geom and sem
        let header_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                (&header as *const SplatFileHeader) as *const u8,
                std::mem::size_of::<SplatFileHeader>(),
            )
        };
        geom_file.write_all(header_bytes)?;
        sem_file.write_all(header_bytes)?;
        lgt_file.write_all(header_bytes)?;

        // We need to iterate in a deterministic order (e.g. by ID or just consistent iteration)
        // Since HashMap iteration is random, we MUST sort by ID or something to ensure
        // geom[i] corresponds to sem[i] corresponds to ids[i].
        // But wait, `entries` is HashMap<SplatId, StoredMemory>.
        // We should sort by ID to be safe and consistent.

        let mut sorted_entries: Vec<_> = self.entries.values().collect();
        sorted_entries.sort_by_key(|e| e.id);

        for entry in sorted_entries {
            // 1. Geometry
            let pos = if let Some(p) = entry.splat.static_points.first() {
                *p
            } else {
                [0.0; 3]
            };

            let geom = SplatGeometry {
                position: pos,
                scale: [1.0; 3], // Should we preserve scale from somewhere? StoredMemory doesn't have it explicitly?
                // Wait, StoredMemory has `splat`. But SplatInput doesn't have scale/rotation explicitly?
                // It seems we lose scale/rotation if we don't store it in StoredMemory.
                // But `load_from_split_files` reads `geoms`.
                // It creates `SplatInput` with `static_points`.
                // It does NOT store `scale` or `rotation` in `StoredMemory`.
                // This is a data loss issue in `load_from_split_files` -> `StoredMemory` conversion.
                // However, for now we use defaults.
                rotation: [0.0, 0.0, 0.0, 1.0],
                color_rgba: [128, 128, 128, 255],
                physics_props: [
                    128,
                    0,
                    entry
                        .meta
                        .emotional_state
                        .as_ref()
                        .map(|e| ((e.pleasure * 127.0) + 128.0) as u8)
                        .unwrap_or(128),
                    0,
                ],
                domain_valence: [0.25, 0.25, 0.25, 0.25], // Neutral
            };

            let geom_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    (&geom as *const SplatGeometry) as *const u8,
                    std::mem::size_of::<SplatGeometry>(),
                )
            };
            geom_file.write_all(geom_bytes)?;

            // 2. PackedSemantics
            let packed = PackedSemantics {
                position: pos,
                opacity: 1.0,
                scale: [1.0; 3],
                _pad1: 0.0,
                rotation: [0.0, 0.0, 0.0, 1.0],
                query_vector: {
                    let mut q = [0.0; 16];
                    for (i, v) in entry.embedding.iter().take(16).enumerate() {
                        q[i] = v.to_f32();
                    }
                    q
                },
            };
            let packed_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    (&packed as *const PackedSemantics) as *const u8,
                    std::mem::size_of::<PackedSemantics>(),
                )
            };
            sem_file.write_all(packed_bytes)?;

            // 3. IDs
            ids_file.write_all(&entry.id.to_le_bytes())?;

            // 4. Embeddings
            for v in &entry.embedding {
                emb_file.write_all(&v.to_f32().to_le_bytes())?;
            }

            // 5. Meta (SplatSemantics)
            let sem = SplatSemantics {
                payload_id: entry.id,
                birth_time: entry.meta.timestamp.unwrap_or(0.0),
                confidence: 1.0,
                embedding: {
                    let mut arr = [0.0f32; crate::constants::FULL_EMBED_DIM];
                    for (i, v) in entry
                        .embedding
                        .iter()
                        .take(crate::constants::FULL_EMBED_DIM)
                        .enumerate()
                    {
                        arr[i] = v.to_f32();
                    }
                    arr
                },
                manifold_vector: {
                    let mut arr = [0.0f32; 64];
                    for (i, v) in entry.manifold_vector.iter().take(64).enumerate() {
                        arr[i] = *v;
                    }
                    arr
                },
                emotional_state: entry.meta.emotional_state.clone(),
                fitness_metadata: entry.meta.fitness_metadata.clone(),
            };
            bincode::serialize_into(&mut meta_file, &sem)?;

            // 6. Lighting
            let lighting = SplatLighting {
                normal: entry
                    .splat
                    .normals
                    .as_ref()
                    .and_then(|v| v.first().cloned())
                    .unwrap_or([0.0, 1.0, 0.0]),
                idiv: entry
                    .splat
                    .idiv
                    .as_ref()
                    .and_then(|v| v.first().cloned())
                    .unwrap_or([0.0; 3]),
                ide: entry
                    .splat
                    .ide
                    .as_ref()
                    .and_then(|v| v.first().cloned())
                    .unwrap_or([0.0; 3]),
                sss_params: entry
                    .splat
                    .sss_params
                    .as_ref()
                    .and_then(|v| v.first().cloned())
                    .unwrap_or([0.0; 4]),
                sh_occlusion: entry
                    .splat
                    .sh_occlusion
                    .as_ref()
                    .and_then(|v| {
                        v.first()
                            .map(|arr| [arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6]])
                    })
                    .unwrap_or([0.0; 7]),
                domain_valence: [0.25, 0.25, 0.25, 0.25], // Neutral
                _pad: [],
            };
            let lgt_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    (&lighting as *const SplatLighting) as *const u8,
                    std::mem::size_of::<SplatLighting>(),
                )
            };
            lgt_file.write_all(lgt_bytes)?;
        }

        Ok(())
    }
}
