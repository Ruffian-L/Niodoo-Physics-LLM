use crate::config::EvolutionKnobs;
use crate::physics::tissue::SemanticGaussian;

pub fn attempt_mitosis(
    parent: &SemanticGaussian,
    score: f32,
    params: &EvolutionKnobs,
) -> Option<(SemanticGaussian, SemanticGaussian)> {
    // 1. Check Threshold
    if score > params.mitosis_score_threshold {
        return None; // Signal is clear enough, no need to split
    }

    // 2. Use Native Split Logic
    let (child_a, child_b) = parent.split();

    // 3. Apply Evolution Knobs (Sharpening)
    // The native split() already does some scaling reduction.
    // But we might want to apply the explicit "mitosis_sharpen_factor".
    // Sharpening means reducing variance (scaling).
    // New Scaling = Old Scaling / Factor (if factor > 1)

    let sharpen = params.mitosis_sharpen_factor;
    if sharpen != 1.0 {
        // We can't modify child_a easily as it's immutable struct?
        // SemanticGaussian fields are public.
        let mut ca = child_a;
        let mut cb = child_b;

        // Apply extra sharpening
        for i in 0..ca.scaling.len() {
            ca.scaling[i] /= sharpen;
            cb.scaling[i] /= sharpen;
        }

        return Some((ca, cb));
    }

    Some((child_a, child_b))
}
