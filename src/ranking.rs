use statrs::statistics::Statistics;

pub struct ReflexStats {
    pub weight: f32,
    pub std_dev: f32,
}

pub fn calculate_adaptive_weight(cosine_scores: &[f32]) -> ReflexStats {
    if cosine_scores.len() < 2 {
        return ReflexStats {
            weight: -0.05,
            std_dev: 0.0,
        }; // Default fallback
    }

    let top_n = 20.min(cosine_scores.len());
    let sample = &cosine_scores[0..top_n];

    // 1. Get Max Score (Confidence)
    // Assuming scores are sorted descending
    let max_score = sample[0];

    // 2. Calculate Standard Deviation
    // Manual implementation to be safe and fast:
    let mean = sample.iter().sum::<f32>() / top_n as f32;
    let variance = sample
        .iter()
        .map(|x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f32>()
        / (top_n as f32 - 1.0).max(1.0); // Sample variance
    let std_dev = variance.sqrt();

    // 3. 2D Signal Classifier Logic
    let weight = if max_score > 0.75 && std_dev < 0.015 {
        // Zone 1: The Consensus Zone (Scientific Fact)
        -0.01
    } else if std_dev > 0.05 {
        // Zone 2: The Clarity Zone (Clear Winner)
        -0.01 // or -0.02
    } else {
        // Zone 3: The Noise Zone (Generic Popularity or Low Confidence)
        // Map std_dev [0.015 ... 0.05] -> weight [-0.15 ... -0.02]
        if std_dev <= 0.015 {
            // If it's low variance but didn't pass Zone 1 (i.e. max_score <= 0.75),
            // it's "Consistently Bad/Mediocre". Filter hard.
            -0.15
        } else {
            // Interpolation
            // Range X: 0.05 - 0.015 = 0.035
            // Range Y: -0.02 - (-0.15) = 0.13
            let slope = 0.13 / 0.035;
            let offset = std_dev - 0.015;

            // Calculate and clamp
            (-0.15 + slope * offset).max(-0.15).min(-0.02)
        }
    };

    ReflexStats { weight, std_dev }
}
