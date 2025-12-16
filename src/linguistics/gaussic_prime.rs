//! GAUSSIAN PRIME (Gʘ) - The Language of 3D Covariance
//!
//! "We are not its authors; we are its first translators."
//!
//! This module implements the linguistic tokenizer that converts 3D covariance
//! matrices into the 64-symbol Gʘ alphabet through eigenvalue quantization.

use anyhow::Result;
use nalgebra::{Matrix3, Vector3};

/// The 64 symbols of GAUSSIAN PRIME (Gʘ)
///
/// Each symbol represents a fundamental 3D shape through its quantized eigenvalues
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GZeroSymbol {
    // Q₀ bins (λ ≈ 0) - The VOID family
    Void = 0, // (0,0,0) - Geometric singularity

    // Q₁ bins (λ ≈ ε) - The POINT family
    Point = 21, // (ε,ε,ε) - Isotropic consciousness

    // Q₂ bins (λ ≈ 1) - The UNIT family
    Sphere = 42, // (1,1,1) - Womb/trap duality
    Cat = 41,    // (1,1,ε) - Forward-stretched with fluff

    // Q₃ bins (λ ≈ ∞) - The INFINITE family
    Line = 53,  // (∞,ε,ε) - 1D vector, path, desire
    Plane = 61, // (∞,∞,ε) - 2D boundary, wall, floor
    Abyss = 63, // (∞,∞,∞) - 3D infinite volume, "god"

    // Additional canonical forms
    Needle = 23, // (∞,ε,ε) - Directed mote
    Coin = 25,   // (ε,1,ε) - Oblate (flattened) mote
    Rice = 38,   // (1,1,ε) - Prolate (stretched) blob
    Sheet = 40,  // (1,1,0) - Defined 2D surface
    Pillar = 43, // (∞,1,1) - Stretched sphere
    Shield = 46, // (1,∞,1) - Flattened sphere
    Tube = 54,   // (∞,1,ε) - 1D path with volume
    Beam = 58,   // (∞,1,1) - Thick 1D path, warmth
    Slab = 62,   // (∞,∞,1) - 2D boundary with thickness
}

impl GZeroSymbol {
    /// Get the semantic meaning of this symbol
    pub fn meaning(&self) -> &'static str {
        match self {
            GZeroSymbol::Void => "singularity, nothingness, silence",
            GZeroSymbol::Point => "isotropic mote, 'I', consciousness",
            GZeroSymbol::Sphere => "isotropic enclosure, womb/trap duality",
            GZeroSymbol::Cat => "anisotropic form, forward-stretched with fluff",
            GZeroSymbol::Line => "1D vector, path, desire, directedness",
            GZeroSymbol::Plane => "2D boundary, wall, floor, containment",
            GZeroSymbol::Abyss => "3D infinite volume, context, 'god'",
            GZeroSymbol::Needle => "directed mote, sharp focus",
            GZeroSymbol::Coin => "oblate (flattened) mote, pressed form",
            GZeroSymbol::Rice => "prolate (stretched) blob, elongated",
            GZeroSymbol::Sheet => "defined 2D surface, membrane",
            GZeroSymbol::Pillar => "stretched sphere, columnar form",
            GZeroSymbol::Shield => "flattened sphere, protective barrier",
            GZeroSymbol::Tube => "1D path with volume, hollow form",
            GZeroSymbol::Beam => "thick 1D path, warmth, energy",
            GZeroSymbol::Slab => "2D boundary with thickness, plate",
        }
    }

    /// Get the canonical eigenvalue triplet for this symbol
    pub fn eigenvalues(&self) -> (f32, f32, f32) {
        match self {
            GZeroSymbol::Void => (0.0, 0.0, 0.0),
            GZeroSymbol::Point => (0.1, 0.1, 0.1),
            GZeroSymbol::Sphere => (1.0, 1.0, 1.0),
            GZeroSymbol::Cat => (1.0, 1.0, 0.1),
            GZeroSymbol::Line => (100.0, 0.1, 0.1),
            GZeroSymbol::Plane => (100.0, 100.0, 0.1),
            GZeroSymbol::Abyss => (100.0, 100.0, 100.0),
            GZeroSymbol::Needle => (100.0, 0.1, 0.1),
            GZeroSymbol::Coin => (0.1, 1.0, 0.1),
            GZeroSymbol::Rice => (1.0, 1.0, 0.1),
            GZeroSymbol::Sheet => (1.0, 1.0, 0.0),
            GZeroSymbol::Pillar => (100.0, 1.0, 1.0),
            GZeroSymbol::Shield => (1.0, 100.0, 1.0),
            GZeroSymbol::Tube => (100.0, 1.0, 0.1),
            GZeroSymbol::Beam => (100.0, 1.0, 1.0),
            GZeroSymbol::Slab => (100.0, 100.0, 1.0),
        }
    }
}

/// The Gʘ Tokenizer - Rosetta Stone for 3D covariance
///
/// Converts 3x3 covariance matrices into Gʘ symbols through eigenvalue quantization
#[derive(Clone)]
pub struct GZeroTokenizer {
    /// Quantization thresholds for logarithmic bins
    epsilon_threshold: f32,
    unit_threshold: f32,
    large_threshold: f32,
}

impl GZeroTokenizer {
    /// Create a new tokenizer with default logarithmic quantization
    pub fn new() -> Self {
        Self {
            // Logarithmic quantization bins (Section 1.2)
            epsilon_threshold: 0.5, // Q₁: ε ≈ 0.01-0.5
            unit_threshold: 5.0,    // Q₂: 1 ≈ 0.5-5.0
            large_threshold: 100.0, // Q₃: ∞ ≈ >5.0
        }
    }

    /// The core linguistic function: covariance → Gʘ symbol
    ///
    /// Implements the "Rosetta Stone" logic from Section 1.2
    pub fn covariance_to_symbol(&self, cov: &Matrix3<f32>) -> Result<GZeroSymbol> {
        // Step 1: Extract eigenvalues (the "phonemes")
        let eig = cov.symmetric_eigen();
        let mut eigenvalues: Vec<f32> = eig.eigenvalues.iter().map(|&v| v.max(0.0)).collect();

        // Step 2: Canonicalize - sort eigenvalues (discard orientation)
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Step 3: Logarithmic quantization (the "Logarithmic Imperative")
        let q = |x: f32| -> u8 {
            if x <= 0.01 {
                0 // Q₀: VOID bin
            } else if x <= self.epsilon_threshold {
                1 // Q₁: POINT bin (ε)
            } else if x <= self.unit_threshold {
                2 // Q₂: UNIT bin (1)
            } else {
                3 // Q₃: LARGE bin (∞)
            }
        };

        // Step 4: Pack into 6-bit symbol ID
        let q1 = q(eigenvalues[0]); // Smallest eigenvalue
        let q2 = q(eigenvalues[1]); // Middle eigenvalue
        let q3 = q(eigenvalues[2]); // Largest eigenvalue

        let symbol_id = ((q3 << 4) | (q2 << 2) | q1) as u8;

        // Step 5: Map to canonical Gʘ symbol
        let symbol = match symbol_id {
            0 => GZeroSymbol::Void,
            21 => GZeroSymbol::Point,
            41 => GZeroSymbol::Cat,
            42 => GZeroSymbol::Sphere,
            53 => GZeroSymbol::Line,
            61 => GZeroSymbol::Plane,
            63 => GZeroSymbol::Abyss,
            23 => GZeroSymbol::Needle,
            25 => GZeroSymbol::Coin,
            38 => GZeroSymbol::Rice,
            40 => GZeroSymbol::Sheet,
            43 => GZeroSymbol::Pillar,
            46 => GZeroSymbol::Shield,
            54 => GZeroSymbol::Tube,
            58 => GZeroSymbol::Beam,
            62 => GZeroSymbol::Slab,
            _ => GZeroSymbol::Void, // Default to void for unmapped symbols
        };

        Ok(symbol)
    }

    /// Reverse: Gʘ symbol → covariance matrix
    ///
    /// For the Gʘ Compiler (Section 4.2) - generate 3D scenes from language
    pub fn symbol_to_covariance(&self, symbol: GZeroSymbol) -> Matrix3<f32> {
        let (lambda1, lambda2, lambda3) = symbol.eigenvalues();

        // Create diagonal matrix with eigenvalues
        // (Orientation is handled by syntax/position, not the symbol itself)
        Matrix3::new(lambda1, 0.0, 0.0, 0.0, lambda2, 0.0, 0.0, 0.0, lambda3)
    }

    /// Parse a Gʘ "word" from 3D Gaussian parameters
    ///
    /// Implements the full linguistic decomposition from Section 2.1
    pub fn parse_gaussian_word(
        &self,
        cov: &Matrix3<f32>,
        position: &Vector3<f32>,
        opacity: f32,
        color: &[f32; 3],
    ) -> GZeroWord {
        let symbol = self.covariance_to_symbol(cov).unwrap_or(GZeroSymbol::Void);

        GZeroWord {
            symbol,
            position: *position,
            opacity,
            base_color: *color,
            // Note: Spherical harmonics would be handled separately for "tone"
        }
    }
}

/// A complete Gʘ "word" - the linguistic unit of 3D meaning
///
/// From Section 2.1: "Word = one Gaussian"
#[derive(Debug, Clone)]
pub struct GZeroWord {
    /// The Gʘ symbol (covariance → noun/object)
    pub symbol: GZeroSymbol,
    /// 3D position (spatial grammar: "at", "in", "above")
    pub position: Vector3<f32>,
    /// Opacity (punctuation: emphasis, whisper, redaction)
    pub opacity: f32,
    /// Base color (adjective: mood, tone)
    pub base_color: [f32; 3],
}

impl GZeroWord {
    /// Get the linguistic interpretation of this word
    pub fn interpret(&self) -> String {
        let symbol_meaning = self.symbol.meaning();
        let position_desc = format!(
            "at ({:.1}, {:.1}, {:.1})",
            self.position.x, self.position.y, self.position.z
        );
        let opacity_desc = match self.opacity {
            1.0 => "emphatic",
            0.5..=0.9 => "suggestive",
            0.1..=0.4 => "whispered",
            0.0 => "redacted",
            _ => "muted",
        };

        format!(
            "({} {} {} in {:?} tone)",
            symbol_meaning, position_desc, opacity_desc, self.base_color
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;

    #[test]
    fn test_cat_symbol() {
        let tokenizer = GZeroTokenizer::new();

        // CAT covariance: (1, 1, 0.1) - forward-stretched with fluff
        let cat_cov = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1);

        let symbol = tokenizer.covariance_to_symbol(&cat_cov).unwrap();
        assert_eq!(symbol, GZeroSymbol::Cat);
        assert_eq!(
            symbol.meaning(),
            "anisotropic form, forward-stretched with fluff"
        );
    }

    #[test]
    fn test_line_symbol() {
        let tokenizer = GZeroTokenizer::new();

        // LINE covariance: (∞, ε, ε) - 1D vector/path
        let line_cov = Matrix3::new(100.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1);

        let symbol = tokenizer.covariance_to_symbol(&line_cov).unwrap();
        assert_eq!(symbol, GZeroSymbol::Line);
        assert_eq!(symbol.meaning(), "1D vector, path, desire, directedness");
    }

    #[test]
    fn test_symbol_compiler() {
        let tokenizer = GZeroTokenizer::new();

        // Compile CAT symbol back to covariance
        let cov = tokenizer.symbol_to_covariance(GZeroSymbol::Cat);
        let symbol = tokenizer.covariance_to_symbol(&cov).unwrap();
        assert_eq!(symbol, GZeroSymbol::Cat);
    }
}
