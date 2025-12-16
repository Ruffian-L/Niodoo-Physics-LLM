use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Represents a row in the Brysbaert Concreteness dataset.
/// We use serde aliases to handle potential variations in header naming.
#[derive(Debug, Deserialize)]
struct BrysbaertRow {
    #[serde(alias = "Word", alias = "word")]
    word: String,

    // The dataset uses "Conc.M" for the mean rating.
    // We map this strictly to the mass value.
    #[serde(alias = "Conc.M", alias = "conc_m")]
    concreteness_mean: f32,
}

/// Represents a row in the NRC-VAD Lexicon.
#[derive(Debug, Deserialize)]
struct VadRow {
    #[serde(alias = "Word", alias = "word")]
    word: String,

    #[serde(alias = "Valence", alias = "valence")]
    valence: f32,

    #[serde(alias = "Arousal", alias = "arousal")]
    arousal: f32,

    #[serde(alias = "Dominance", alias = "dominance")]
    dominance: f32,
}

/// The PhysicsRegistry serves as the central database for particle properties.
/// It ensures that every entity in the simulation has both Mass and Charge.
pub struct PhysicsRegistry {
    // Maps a word to its inertial mass scalar.
    pub mass_map: HashMap<String, f32>,

    // Maps a word to its charge vector.
    pub charge_map: HashMap<String, [f32; 3]>,

    // The computational device (CPU/CUDA) where tensors will be allocated.
    device: Device,
}

impl PhysicsRegistry {
    /// Initializes a new, empty registry on the specified device.
    pub fn new(device: &Device) -> Self {
        Self {
            mass_map: HashMap::new(),
            charge_map: HashMap::new(),
            device: device.clone(),
        }
    }

    /// Loads and intersects the datasets.
    ///
    /// # Arguments
    /// * `mass_path` - Path to the Brysbaert Concreteness CSV.
    /// * `vad_path` - Path to the NRC-VAD Lexicon CSV.
    pub fn load_datasets<P: AsRef<Path>>(&mut self, mass_path: P, vad_path: P) -> Result<()> {
        // Step 1: Load Mass Data
        // We use a temporary map to hold the raw mass data before intersection.
        let mut temp_mass_map: HashMap<String, f32> = HashMap::new();

        println!("Loading mass data from {:?}", mass_path.as_ref());
        let mass_file = File::open(mass_path).context("Failed to open Mass dataset")?;
        let mut mass_rdr = csv::ReaderBuilder::new()
            .delimiter(b'\t') // Brysbaert is typically TSV
            .has_headers(true)
            .from_reader(mass_file);

        for result in mass_rdr.deserialize() {
            let record: BrysbaertRow = result.context("Failed to parse mass row")?;

            // Normalization: Transform [1.0, 5.0] -> [0.1, 1.0]
            // Formula: normalized = 0.1 + (raw - 1.0) * (0.9 / 4.0)
            let raw = record.concreteness_mean;
            let normalized_mass = 0.1 + (raw - 1.0) * 0.225;

            // Store as lowercase to ensure case-insensitive matching
            temp_mass_map.insert(record.word.to_lowercase(), normalized_mass);
        }
        println!("Ingested {} raw mass entries.", temp_mass_map.len());

        // Step 2: Load Charge Data and Intersect
        println!("Loading VAD data from {:?}", vad_path.as_ref());
        let vad_file = File::open(vad_path).context("Failed to open VAD dataset")?;
        let mut vad_rdr = csv::ReaderBuilder::new()
            .delimiter(b'\t') // NRC-VAD is often TSV; change to b',' if CSV
            .has_headers(false) // Verified: No headers in file
            .from_reader(vad_file);

        for result in vad_rdr.deserialize() {
            // Need relax on parsing because sometimes rows are weird
            match result {
                Ok(record) => {
                    let record: VadRow = record;
                    let word = record.word.to_lowercase();

                    // Intersection Check: Does this word exist in our mass map?
                    if let Some(&mass) = temp_mass_map.get(&word) {
                        // If yes, we add it to the permanent registry.
                        self.mass_map.insert(word.clone(), mass);

                        // Normalization: Transform [0.0, 1.0] -> [-0.5, 0.5]
                        let q_v = record.valence - 0.5;
                        let q_a = record.arousal - 0.5;
                        let q_d = record.dominance - 0.5;

                        self.charge_map.insert(word, [q_v, q_a, q_d]);
                    }
                }
                Err(e) => {
                    // Just log and continue
                    // println!("Skipping VAD row error: {:?}", e);
                }
            }
        }

        println!(
            "Registry finalized. Total Physics-Ready Entities: {}",
            self.mass_map.len()
        );

        if self.mass_map.is_empty() {
            anyhow::bail!("Intersection resulted in 0 entries. Check CSV delimiters and headers.");
        }

        Ok(())
    }

    /// Exports the registry data to Candle Tensors on the configured device.
    /// Returns tuple: (Mass Tensor [N, 1], Charge Tensor [N, 3], Word List)
    pub fn to_tensors(&self) -> Result<(Tensor, Tensor, Vec<String>)> {
        let mut words = Vec::with_capacity(self.mass_map.len());
        let mut masses = Vec::with_capacity(self.mass_map.len());
        let mut charges_flat = Vec::with_capacity(self.mass_map.len() * 3);

        // Iterate deterministically over the keys to ensure alignment
        // (HashMap iteration is arbitrary, so we sort keys first or just iterate collected keys)
        // Ideally, we maintain a separate ordered list, but sorting keys works for reproducibility.
        let mut sorted_keys: Vec<&String> = self.mass_map.keys().collect();
        sorted_keys.sort();

        for word in sorted_keys {
            words.push(word.clone());

            let mass = self.mass_map.get(word).unwrap();
            masses.push(*mass);

            let charge = self.charge_map.get(word).unwrap();
            charges_flat.push(charge[0]);
            charges_flat.push(charge[1]);
            charges_flat.push(charge[2]);
        }

        let n = words.len();

        // Create Mass Tensor: Shape [N, 1]
        let mass_tensor = Tensor::from_vec(masses, (n, 1), &self.device)?;

        // Create Charge Tensor: Shape [N, 3]
        let charge_tensor = Tensor::from_vec(charges_flat, (n, 3), &self.device)?;

        Ok((mass_tensor, charge_tensor, words))
    }
}
