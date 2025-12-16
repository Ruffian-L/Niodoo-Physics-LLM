use crate::config::SplatMemoryConfig;
use crate::storage::transaction::SplatTransaction;
use crate::structs::{
    PackedSemantics, SplatFileHeader, SplatGeometry, SplatLighting, SplatManifest,
    SplatManifestEntry, SplatSemanticsV2,
};
use anyhow::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::mem;
use std::path::Path;

pub struct SplatStorage {
    // In-memory storage (SoA)
    pub geometries: Vec<SplatGeometry>,
    pub semantics: Vec<PackedSemantics>,
    pub lighting: Vec<SplatLighting>,
    pub manifest: HashMap<u64, SplatManifestEntry>,

    // O(1) lookup for payload_id -> index in SoA arrays
    pub id_to_index: HashMap<u64, usize>,

    // Parallel arrays for ID and Embedding
    pub payload_ids: Vec<u64>,
    pub embeddings: Vec<Vec<f32>>,
    pub rvq_indices: Vec<[u16; crate::encoder::rvq_candle::NUM_QUANTIZERS]>, // V2: Discrete codes - POST-COSINE

    // Phoneme Index: payload_id -> (start_byte_offset, count)
    pub phoneme_index: HashMap<u64, (u64, u64)>,

    pub next_payload_id: u64,

    // Paths
    pub geom_path: String,
    pub sem_path: String,
    pub lgt_path: String,
    pub manifest_path: String,
    pub phoneme_path: String,
    pub phoneme_index_path: String,
    pub emb_path: String,
    pub rvq_path: String, // V2
    pub ids_path: String,
}

impl SplatStorage {
    pub fn new(base_path: &str, manifest_path: &str) -> Result<Self> {
        let geom_path = format!("{}.splat", base_path);
        let sem_path = format!("{}.sem", base_path);
        let lgt_path = format!("{}.lgt", base_path);
        let phoneme_path = format!("{}_phonemes.bin", base_path);
        let phoneme_index_path = format!("{}_phoneme_index.json", base_path);
        let emb_path = format!("{}.emb", base_path);
        let rvq_path = format!("{}.rvq", base_path);
        let ids_path = format!("{}.ids", base_path);

        let mut storage = Self {
            geometries: Vec::new(),
            semantics: Vec::new(),
            lighting: Vec::new(),
            manifest: HashMap::new(),
            id_to_index: HashMap::new(),
            payload_ids: Vec::new(),
            embeddings: Vec::new(),
            rvq_indices: Vec::new(),
            phoneme_index: HashMap::new(),
            next_payload_id: 0,
            geom_path,
            sem_path,
            lgt_path,
            manifest_path: manifest_path.to_string(),
            phoneme_path,
            phoneme_index_path,
            emb_path,
            rvq_path,
            ids_path,
        };

        storage.load()?;
        Ok(storage)
    }

    fn load(&mut self) -> Result<()> {
        // Load Geometry
        if Path::new(&self.geom_path).exists() {
            let mut file = File::open(&self.geom_path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;

            let header_size = mem::size_of::<SplatFileHeader>();
            let start_offset = if buffer.len() >= header_size && &buffer[0..8] == b"SPLTRAG\0" {
                header_size
            } else {
                0
            };

            let size = mem::size_of::<SplatGeometry>();
            if size > 0 && buffer.len() >= start_offset {
                let count = (buffer.len() - start_offset) / size;
                self.geometries = unsafe {
                    std::slice::from_raw_parts(
                        buffer[start_offset..].as_ptr() as *const SplatGeometry,
                        count,
                    )
                    .to_vec()
                };
            }
        }

        // Load Semantics
        if Path::new(&self.sem_path).exists() {
            let mut file = File::open(&self.sem_path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;

            let header_size = mem::size_of::<SplatFileHeader>();
            if buffer.len() >= header_size {
                let data_slice = &buffer[header_size..];
                let item_size = mem::size_of::<PackedSemantics>();
                if item_size > 0 {
                    let count = data_slice.len() / item_size;
                    self.semantics = unsafe {
                        std::slice::from_raw_parts(
                            data_slice.as_ptr() as *const PackedSemantics,
                            count,
                        )
                        .to_vec()
                    };
                }
            }
        }

        // Load IDs
        if Path::new(&self.ids_path).exists() {
            let mut file = File::open(&self.ids_path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            let count = buffer.len() / 8;
            self.payload_ids = unsafe {
                std::slice::from_raw_parts(buffer.as_ptr() as *const u64, count).to_vec()
            };
        }

        // Load Embeddings
        if Path::new(&self.emb_path).exists() {
            let mut file = File::open(&self.emb_path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            if !self.payload_ids.is_empty() {
                let total_floats = buffer.len() / 4;
                let dim = total_floats / self.payload_ids.len();
                let floats: &[f32] = bytemuck::cast_slice(&buffer);
                self.embeddings = floats.chunks(dim).map(|c| c.to_vec()).collect();
            }
        }

        // Load RVQ Indices (V2)
        if Path::new(&self.rvq_path).exists() {
            let mut file = File::open(&self.rvq_path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            if !buffer.is_empty() {
                // Each entry is 16 * u16 = 32 bytes - POST-COSINE
                let entry_size = 32;
                let count = buffer.len() / entry_size;
                let ptr =
                    buffer.as_ptr() as *const [u16; crate::encoder::rvq_candle::NUM_QUANTIZERS];
                self.rvq_indices = unsafe { std::slice::from_raw_parts(ptr, count).to_vec() };
            }
        }

        // Load Manifest
        if Path::new(&self.manifest_path).exists() {
            let is_json = self.manifest_path.ends_with(".json");
            let mut loaded = false;

            if !is_json {
                let file = File::open(&self.manifest_path)?;
                let reader = std::io::BufReader::new(file);

                // Try Bincode first
                if let Ok(m) = bincode::deserialize_from::<_, SplatManifest>(reader) {
                    for entry in m.entries {
                        self.manifest.insert(entry.id, entry);
                    }
                    loaded = true;
                }
            }

            if !loaded {
                let file = File::open(&self.manifest_path)?;
                let reader = std::io::BufReader::new(file);

                // Try JSON SplatManifest
                if let Ok(m) = serde_json::from_reader::<_, SplatManifest>(reader) {
                    for entry in m.entries {
                        self.manifest.insert(entry.id, entry);
                    }
                    loaded = true;
                }
            }

            if !loaded {
                let file = File::open(&self.manifest_path)?;
                let reader = std::io::BufReader::new(file);
                let legacy: HashMap<u64, String> =
                    serde_json::from_reader(reader).unwrap_or_default();
                for (k, v) in legacy {
                    self.manifest.insert(
                        k,
                        SplatManifestEntry {
                            id: k,
                            text: v,
                            birth_time: 0.0,
                            valence_history: vec![],
                            initial_valence: 0,
                            tags: vec![],
                        },
                    );
                }
            }
            self.next_payload_id = self.manifest.keys().max().copied().unwrap_or(0) + 1;
        }

        // Load Phoneme Index
        if Path::new(&self.phoneme_index_path).exists() {
            let file = File::open(&self.phoneme_index_path)?;
            if let Ok(idx) = serde_json::from_reader(file) {
                self.phoneme_index = idx;
            }
        }

        // Rebuild ID to Index
        for (i, &id) in self.payload_ids.iter().enumerate() {
            self.id_to_index.insert(id, i);
        }

        Ok(())
    }

    pub fn persist_batch(
        &mut self,
        batch: Vec<(
            u64,
            String,
            SplatGeometry,
            crate::structs::PackedSemantics,
            SplatSemanticsV2,
            SplatLighting,
            Vec<f32>,
        )>,
        _config: &SplatMemoryConfig,
    ) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }

        let mut geom_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&self.geom_path)?;
        let mut sem_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&self.sem_path)?;
        let mut lgt_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&self.lgt_path)?;

        // Check and write headers if empty
        if geom_file.metadata()?.len() == 0 {
            let header = SplatFileHeader {
                magic: *b"SPLTRAG\0",
                version: 2, // Manifesto V2
                count: 0,
                geometry_size: mem::size_of::<SplatGeometry>() as u32,
                semantics_size: mem::size_of::<PackedSemantics>() as u32,
                motion_size: 0,
                lighting_size: mem::size_of::<SplatLighting>() as u32,
                _pad: [0; 2],
            };
            geom_file.write_all(bytemuck::bytes_of(&header))?;
        }

        if sem_file.metadata()?.len() == 0 {
            let header = SplatFileHeader {
                magic: *b"SPLTRAG\0",
                version: 1,
                count: 0,
                geometry_size: mem::size_of::<SplatGeometry>() as u32,
                semantics_size: mem::size_of::<PackedSemantics>() as u32,
                motion_size: 0,
                lighting_size: mem::size_of::<SplatLighting>() as u32,
                _pad: [0; 2],
            };
            sem_file.write_all(bytemuck::bytes_of(&header))?;
        }

        if lgt_file.metadata()?.len() == 0 {
            let header = SplatFileHeader {
                magic: *b"SPLTRAG\0",
                version: 1,
                count: 0,
                geometry_size: mem::size_of::<SplatGeometry>() as u32,
                semantics_size: mem::size_of::<PackedSemantics>() as u32,
                motion_size: 0,
                lighting_size: mem::size_of::<SplatLighting>() as u32,
                _pad: [0; 2],
            };
            lgt_file.write_all(bytemuck::bytes_of(&header))?;
        }
        let mut phoneme_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&self.phoneme_path)?;
        let mut emb_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&self.emb_path)?;
        let mut rvq_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&self.rvq_path)?;
        let mut ids_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&self.ids_path)?;

        let mut transaction = SplatTransaction::begin(
            &mut geom_file,
            &mut sem_file,
            &mut lgt_file,
            &mut phoneme_file,
            &mut emb_file,
            &mut rvq_file,
        )?;
        let initial_phoneme_offset = transaction.phoneme_start;

        let write_result = (|| -> Result<()> {
            for (_id, _txt, geom, sem, sem_v2, lgt, _emb) in &batch {
                // Write Geometry
                transaction.geom_file.write_all(bytemuck::bytes_of(geom))?;

                // Write Semantics (Packed)
                transaction.sem_file.write_all(bytemuck::bytes_of(sem))?;

                // Write Lighting
                transaction.lgt_file.write_all(bytemuck::bytes_of(lgt))?;

                // Write RVQ (V2)
                transaction
                    .rvq_file
                    .write_all(bytemuck::bytes_of(&sem_v2.rvq_indices))?;
            }
            Ok(())
        })();

        match write_result {
            Ok(_) => transaction.commit()?,
            Err(e) => {
                transaction.rollback()?;
                return Err(e);
            }
        }

        // Update In-Memory State
        let mut current_phoneme_offset = initial_phoneme_offset;

        for (id, txt, geom, sem, sem_v2, lgt, embedding) in batch {
            // Append to IDs file
            ids_file.write_all(&id.to_le_bytes())?;

            self.manifest.insert(
                id,
                SplatManifestEntry {
                    id,
                    text: txt,
                    birth_time: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs_f64(),
                    valence_history: vec![],
                    initial_valence: 0,
                    tags: vec![],
                },
            );
            self.geometries.push(geom);
            self.payload_ids.push(id);
            self.embeddings.push(embedding);

            let idx = self.semantics.len();
            self.id_to_index.insert(id, idx);
            self.semantics.push(sem);
            self.lighting.push(lgt);
            self.rvq_indices.push(sem_v2.rvq_indices);

            self.next_payload_id = self.next_payload_id.max(id + 1);
        }

        self.save_manifest()?;
        self.save_phoneme_index()?;

        Ok(())
    }

    pub fn save_manifest(&self) -> Result<()> {
        let mf = File::create(&self.manifest_path)?;
        let mut writer = std::io::BufWriter::new(mf);
        let entries: Vec<_> = self.manifest.values().cloned().collect();
        let manifest_struct = SplatManifest { entries };

        if self.manifest_path.ends_with(".json") {
            serde_json::to_writer(writer, &manifest_struct)?;
        } else {
            bincode::serialize_into(&mut writer, &manifest_struct)?;
        }
        Ok(())
    }

    pub fn save_phoneme_index(&self) -> Result<()> {
        let pf = File::create(&self.phoneme_index_path)?;
        serde_json::to_writer(pf, &self.phoneme_index)?;
        Ok(())
    }

    pub fn save_all(&self) -> Result<()> {
        // Atomic save: write to .tmp then rename
        let geom_tmp = format!("{}.tmp", self.geom_path);
        let sem_tmp = format!("{}.tmp", self.sem_path);
        let lgt_tmp = format!("{}.tmp", self.lgt_path);
        let emb_tmp = format!("{}.tmp", self.emb_path);
        let rvq_tmp = format!("{}.tmp", self.rvq_path);
        let ids_tmp = format!("{}.tmp", self.ids_path);

        let header = SplatFileHeader {
            magic: *b"SPLTRAG\0",
            version: 2, // Manifesto V2
            count: self.geometries.len() as u64,
            geometry_size: mem::size_of::<SplatGeometry>() as u32,
            semantics_size: mem::size_of::<PackedSemantics>() as u32,
            motion_size: 0,
            lighting_size: mem::size_of::<SplatLighting>() as u32,
            _pad: [0; 2],
        };

        // 1. Write Geometry
        {
            let mut f = File::create(&geom_tmp)?;
            f.write_all(bytemuck::bytes_of(&header))?;
            for g in &self.geometries {
                f.write_all(bytemuck::bytes_of(g))?;
            }
        }

        // 2. Write Semantics
        {
            let mut f = File::create(&sem_tmp)?;
            f.write_all(bytemuck::bytes_of(&header))?;
            for s in &self.semantics {
                f.write_all(bytemuck::bytes_of(s))?;
            }
        }

        // 2.5 Write Lighting
        {
            let mut f = File::create(&lgt_tmp)?;
            f.write_all(bytemuck::bytes_of(&header))?;
            for l in &self.lighting {
                f.write_all(bytemuck::bytes_of(l))?;
            }
        }

        // 3.5 Write RVQ
        {
            let mut f = File::create(&rvq_tmp)?;
            for r in &self.rvq_indices {
                f.write_all(bytemuck::bytes_of(r))?;
            }
        }

        // 4. Write IDs
        {
            let mut f = File::create(&ids_tmp)?;
            for id in &self.payload_ids {
                f.write_all(&id.to_le_bytes())?;
            }
        }

        // 5. Rename all
        std::fs::rename(&geom_tmp, &self.geom_path)?;
        std::fs::rename(&sem_tmp, &self.sem_path)?;
        std::fs::rename(&lgt_tmp, &self.lgt_path)?;
        std::fs::rename(&rvq_tmp, &self.rvq_path)?;
        std::fs::rename(&ids_tmp, &self.ids_path)?;

        // Also save manifest and phoneme index
        self.save_manifest()?;
        self.save_phoneme_index()?;

        Ok(())
    }
    pub fn add_splat(
        &mut self,
        input: &crate::types::SplatInput,
        blob: crate::storage::OpaqueSplatRef,
        text: String,
        embedding: Vec<f32>,
    ) -> Result<()> {
        let id = self.next_payload_id;
        self.next_payload_id += 1;

        // Convert SplatInput to internal structs
        let geom = SplatGeometry {
            position: [
                input.static_points[0][0],
                input.static_points[0][1],
                input.static_points[0][2],
            ],
            scale: [input
                .covariances
                .first()
                .map(|c| c[0].sqrt())
                .unwrap_or(1.0); 3], // Approx scale
            rotation: [0.0, 0.0, 0.0, 1.0],
            color_rgba: [255, 255, 255, 255],
            physics_props: [
                0, // Entropy
                0, // Anisotropy
                input
                    .idiv
                    .as_ref()
                    .map(|v| (v[0][0] * 127.0) as u8)
                    .unwrap_or(0), // Valence approx
                0,
            ],
            domain_valence: [0.25, 0.25, 0.25, 0.25], // Neutral: will be classified at ingestion
        };

        let sem = PackedSemantics {
            position: geom.position,
            opacity: 1.0,
            scale: geom.scale,
            _pad1: 0.0,
            rotation: geom.rotation,
            query_vector: [0.0; 16],
        };

        let lighting = SplatLighting {
            normal: input
                .normals
                .as_ref()
                .map(|v| v[0])
                .unwrap_or([0.0, 1.0, 0.0]),
            idiv: input.idiv.as_ref().map(|v| v[0]).unwrap_or([0.0; 3]),
            ide: input.ide.as_ref().map(|v| v[0]).unwrap_or([0.0; 3]),
            sss_params: input.sss_params.as_ref().map(|v| v[0]).unwrap_or([0.0; 4]),
            sh_occlusion: input
                .sh_occlusion
                .as_ref()
                .map(|v| {
                    [
                        v[0][0], v[0][1], v[0][2], v[0][3], v[0][4], v[0][5], v[0][6],
                    ]
                })
                .unwrap_or([0.0; 7]),
            domain_valence: [0.25, 0.25, 0.25, 0.25], // Neutral - will be classified later
            _pad: [],
        };

        // Persist
        // We need to construct dummy V2 semantics for add_splat
        let sem_v2 = crate::structs::SplatSemanticsV2 {
            payload_id: id,
            birth_time: 0.0,
            confidence: 0.0,
            rvq_indices: [0; crate::encoder::rvq_candle::NUM_QUANTIZERS], // POST-COSINE dummy
            coarse_mass: 0,
            domain_valence: [0.25; 4],
            manifold_vector: [0.0; 64],
            emotional_state: None,
            fitness_metadata: None,
        };
        let batch = vec![(id, text, geom, sem, sem_v2, lighting, embedding)];
        self.persist_batch(batch, &SplatMemoryConfig::default())?;

        Ok(())
    }

    // Helper for retrieving blob (text)
    pub fn blob(&self, id: u64) -> Option<crate::storage::OpaqueSplatRef> {
        self.manifest
            .get(&id)
            .map(|e| crate::storage::OpaqueSplatRef::External(e.text.clone()))
    }

    pub fn entries(&self) -> std::collections::hash_map::Iter<u64, SplatManifestEntry> {
        self.manifest.iter()
    }

    pub fn entries_mut(&mut self) -> std::collections::hash_map::IterMut<u64, SplatManifestEntry> {
        self.manifest.iter_mut()
    }

    pub fn get(&self, id: u64) -> Option<&SplatManifestEntry> {
        self.manifest.get(&id)
    }

    pub fn remove(&mut self, id: u64) -> Option<SplatManifestEntry> {
        // This is complex because we need to remove from parallel arrays.
        // For now, just remove from manifest to mark as deleted.
        // Full compaction is needed for arrays.
        self.manifest.remove(&id)
    }
}
