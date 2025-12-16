use crate::structs::SplatGeometry;

/// Clamps physics values to safe ranges to prevent model collapse or seizures.
pub fn sanitize_geometry(geo: &mut SplatGeometry) {
    // 1. Singularity Check: Ensure no axis is too small (collapsing to 2D/1D)
    // We iterate through the 3 scale components
    for i in 0..3 {
        if geo.scale[i] < 0.05 {
            geo.scale[i] = 0.05;
        }
    }

    // 2. Seizure Check: Limit Anisotropy (Ratio of Max/Min scale)
    // High anisotropy (extreme needles) causes numerical instability in the embedding space
    let max_scale = geo.scale[0].max(geo.scale[1]).max(geo.scale[2]);
    let min_scale = geo.scale[0].min(geo.scale[1]).min(geo.scale[2]);

    // Avoid division by zero (though step 1 should prevent this)
    if min_scale > 0.0 {
        let anisotropy = max_scale / min_scale;
        if anisotropy > 10.0 {
            // If too stretched, boost the minimum dimensions to satisfy the ratio
            // target_min = max / 10.0
            let target_min = max_scale / 10.0;

            for i in 0..3 {
                if geo.scale[i] < target_min {
                    geo.scale[i] = target_min;
                }
            }
        }
    }

    // 3. Zero Check (Redundant but safe)
    for i in 0..3 {
        if geo.scale[i] == 0.0 {
            geo.scale[i] = 0.001;
        }
    }
}
