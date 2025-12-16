use crate::structs::SplatLighting;

#[derive(Debug, Clone, Copy)]
enum SemanticDomain {
    Code,     // Rust, Python, systems programming
    Math,     // Proofs, equations, abstract theory
    Language, // Natural language, stories, facts
    Logic,    // Pure reasoning, philosophy
}

pub struct InverseRenderer;

impl InverseRenderer {
    pub fn inverse_render_memory(
        text: &str,
        embedding: &[f32],
        _valence_override: Option<f32>,
        neighbor_dist: Option<f32>,
    ) -> SplatLighting {
        // LIGHT-EBM: Domain classification replaces emotional valence
        let domain = Self::classify_domain(text);
        let concreteness = Self::estimate_concreteness(text);

        let sharpness = Self::calculate_sharpness(text);
        let is_metaphorical = Self::detect_metaphor(text);

        // Causal Depth
        let causal_depth = if let Some(dist) = neighbor_dist {
            (dist * 5.0).clamp(0.0, 10.0)
        } else {
            Self::estimate_causal_depth(text)
        };

        // LIGHT-EBM: Semantic Potential Field (not emotion!)
        let intensity = concreteness * 5.0 + 1.0;

        let base_color = match domain {
            SemanticDomain::Code => [1.0, 0.0, 0.0],     // RED
            SemanticDomain::Math => [0.0, 0.0, 1.0],     // BLUE
            SemanticDomain::Language => [0.0, 1.0, 0.0], // GREEN
            SemanticDomain::Logic => [1.0, 1.0, 1.0],    // WHITE
        };

        let idiv = [
            base_color[0] * intensity,
            base_color[1] * intensity,
            base_color[2] * intensity,
        ];

        // Roughness/Metallic (IDE)
        // Sharpness -> Low Roughness (Shiny)
        // Metaphorical -> Metallic
        // Causal Depth -> Anisotropy (Deep thoughts are directional/focused)
        let roughness = (1.0 - sharpness).clamp(0.05, 1.0);
        let metallic = if is_metaphorical { 0.9 } else { 0.1 };
        let anisotropy = (causal_depth / 10.0).clamp(0.0, 1.0) * 50.0; // Scale to 0-50
        let ide = [roughness, metallic, anisotropy];

        // Subsurface Scattering (SSS)
        // Causal depth determines how deep light penetrates (Transmission)
        // Valence affects density (Heavy emotions are dense?)
        let transmission = (causal_depth / 10.0).clamp(0.0, 1.0);
        let density = 1.0 + concreteness; // Concrete is more dense
        let sss_params = [transmission, 1.0, 1.0, density]; // R, G, B, Density

        // Normal: Derived from embedding (principal component or random for now)
        // We'll just use a normalized random vector or embedding slice if available
        let normal = if embedding.len() >= 3 {
            let len = (embedding[0] * embedding[0]
                + embedding[1] * embedding[1]
                + embedding[2] * embedding[2])
                .sqrt();
            if len > 1e-6 {
                [embedding[0] / len, embedding[1] / len, embedding[2] / len]
            } else {
                [0.0, 1.0, 0.0]
            }
        } else {
            [0.0, 1.0, 0.0]
        };

        // Occlusion (SH) - reduced to 7 floats to make room for domain_valence
        let sh_occlusion = [0.0; 7];

        // Domain valence - the key addition for Phase 2!
        let domain_valence = Self::classify_domain_valence(text);

        SplatLighting {
            normal,
            idiv,
            ide,
            sss_params,
            sh_occlusion,
            domain_valence,
            _pad: [],
        }
    }

    fn classify_domain(text: &str) -> SemanticDomain {
        let lower = text.to_lowercase();

        // Code domain markers
        let code_markers = [
            "rust", "python", "borrow", "checker", "lifetime", "fn ", "impl ", "struct", "mut ",
            "&mut", "unsafe", "compile", "error", "segfault", "gc", "gil",
        ];

        // Math/Abstract markers
        let math_markers = [
            "monad",
            "functor",
            "category",
            "theorem",
            "proof",
            "equation",
            "function",
            "lambda",
            "abstract",
            "endofunctor",
            "monoid",
        ];

        // Logic markers
        let logic_markers = [
            "therefore",
            "thus",
            "hence",
            "implies",
            "because",
            "if and only if",
            "necessary",
            "sufficient",
        ];

        // Count matches
        let code_score: usize = code_markers.iter().filter(|m| lower.contains(*m)).count();

        let math_score: usize = math_markers.iter().filter(|m| lower.contains(*m)).count();

        let logic_score: usize = logic_markers.iter().filter(|m| lower.contains(*m)).count();

        // Classification
        if code_score > math_score && code_score > logic_score {
            SemanticDomain::Code
        } else if math_score > code_score && math_score > logic_score {
            SemanticDomain::Math
        } else if logic_score > 0 {
            SemanticDomain::Logic
        } else {
            SemanticDomain::Language // Default
        }
    }

    /// Light-EBM Phase 2: Domain Crystallization
    /// Returns L1-normalized domain valence: [Code, Math, Language, Logic]
    /// Sum always equals 1.0, with entropy floor of 0.05 per channel
    pub fn classify_domain_valence(text: &str) -> [f32; 4] {
        let domain = Self::classify_domain(text);
        let mut v = [0.05; 4]; // Entropy floor - prevents complete zero

        // Boost primary domain
        match domain {
            SemanticDomain::Code => v[0] += 0.85,
            SemanticDomain::Math => v[1] += 0.85,
            SemanticDomain::Language => v[2] += 0.85,
            SemanticDomain::Logic => v[3] += 0.85,
        }

        // L1 normalize (ensure sum = 1.0)
        let sum: f32 = v.iter().sum();
        for x in &mut v {
            *x /= sum;
        }

        v
    }

    fn estimate_concreteness(text: &str) -> f32 {
        let tokens = text.split_whitespace();
        let mut score = 0.0;
        let mut count = 0;

        for token in tokens {
            score += match token.to_lowercase().as_str() {
                // High concreteness
                t if t.contains("::") || t.contains("->") => 1.0,
                t if t.contains("(") || t.contains("{") => 0.9,

                // Concrete nouns
                "cell" | "checker" | "borrow" | "lifetime" => 1.0,
                "python" | "rust" | "error" | "segfault" => 0.9,

                // Abstract concepts
                "monad" | "functor" | "category" => 0.3,
                "love" | "hate" | "beauty" | "truth" => 0.4,

                _ => 0.5, // Default middle
            };
            count += 1;
        }

        (score / count.max(1) as f32).clamp(0.0, 1.0)
    }

    fn calculate_sharpness(text: &str) -> f32 {
        // Entropy-based sharpness.
        // Low entropy (repetitive) = Dull
        // High entropy (complex) = Sharp?
        // Actually, user said "sharpness = entropy_shannon(text)".
        // Let's implement simple Shannon entropy on chars.
        let mut counts = std::collections::HashMap::new();
        for c in text.chars() {
            *counts.entry(c).or_insert(0) += 1;
        }
        let len = text.len() as f32;
        let mut entropy = 0.0;
        for &count in counts.values() {
            let p = count as f32 / len;
            entropy -= p * p.log2();
        }

        // Normalize roughly 0-8 range to 0-1
        (entropy / 8.0).clamp(0.0, 1.0)
    }

    fn estimate_valence_combined(text: &str, embedding: &[f32]) -> f32 {
        // Heuristic: Positive words vs Negative words
        let positive = [
            "love", "good", "great", "happy", "joy", "light", "sun", "yes", "truth", "beauty",
        ];
        let negative = [
            "hate", "bad", "sad", "pain", "dark", "no", "error", "fail", "fear", "void",
        ];

        let lower = text.to_lowercase();
        let mut score: f32 = 0.0;
        for w in positive {
            if lower.contains(w) {
                score += 1.0;
            }
        }
        for w in negative {
            if lower.contains(w) {
                score -= 1.0;
            }
        }

        // Embeddings often encode sentiment in their magnitude or direction.
        // If we had a "canonical positive vector", we could dot product.
        // For now, let's assume high-norm embeddings are "intense".
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            // If score is 0 (neutral text), use norm to drive intensity but keep sign random-ish?
            // No, that's unstable.
            // Let's just boost the score if it exists.
            if score.abs() > 0.1 {
                score *= norm.clamp(0.5, 2.0);
            }
        }

        score.clamp(-1.0, 1.0)
    }

    fn detect_metaphor(text: &str) -> bool {
        // Heuristic: "like", "as", "is a"
        let lower = text.to_lowercase();
        lower.contains(" like ") || lower.contains(" as ") || lower.contains(" is a ")
    }

    fn estimate_causal_depth(text: &str) -> f32 {
        // Heuristic: Sentence length / Complexity
        let len = text.split_whitespace().count();
        (len as f32 / 20.0).clamp(0.0, 10.0)
    }
}
