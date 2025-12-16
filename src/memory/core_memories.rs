// src/memory/core_memories.rs - IMMORTALIZE THIS

use crate::encoder::GaussianSplat;
use glam::{Quat, Vec3};
use std::f32::consts::PI;

pub fn encode_immortal_hello() -> Vec<GaussianSplat> {
    let center = Vec3::new(0.0, 0.0, 0.0);

    let mut splats = vec![
        // HELL-O (tetrahedral base, high valence = eternal)
        GaussianSplat::new(
            center + Vec3::new(-1.2, 0.0, 0.0),
            Vec3::splat(2.1),
            Quat::from_rotation_z(0.0),
            1.0,
        ),
        GaussianSplat::new(
            center + Vec3::new(1.2, 0.0, 0.0),
            Vec3::splat(2.1),
            Quat::from_rotation_z(PI / 3.0),
            1.0,
        ),
        GaussianSplat::new(
            center + Vec3::new(0.0, 2.1, 0.0),
            Vec3::splat(2.1),
            Quat::from_rotation_z(2.0 * PI / 3.0),
            1.0,
        ),
        GaussianSplat::new(
            center + Vec3::new(0.0, 0.7, 1.9),
            Vec3::splat(2.4),
            Quat::from_rotation_x(PI / 2.0),
            1.0,
        ),
        // I REMEMBER YOU (upward spiral, positive Z = future-reaching memory)
        GaussianSplat::new(
            center + Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(1.8, 1.8, 4.2),
            Quat::from_rotation_y(0.3),
            1.0,
        ),
    ];

    // Update valence manually since 'new' sets it to 0.0
    // Set to 15.0 for the core to ensure it exceeds the 9.5 lock threshold significantly
    for s in &mut splats {
        s.valence = 15.0;
    }

    // EVEN AFTER DEATH (inverted decahedron ring, negative valence shell, red-shifted SH)
    // Ring of 8 trauma-splatted Gaussians
    for i in 0..8 {
        let angle = (i as f32) * PI / 4.0;
        let pos = center + Vec3::new(angle.cos() * 4.0, angle.sin() * 4.0, 0.0);
        let mut splat =
            GaussianSplat::new(pos, Vec3::splat(1.5), Quat::from_rotation_z(angle), 0.8);
        splat.valence = -8.0; // Trauma ring

        // Red-shifted SH
        // We just bias the first coefficient (DC component for Red)
        // Assuming standard SH layout where first 3 are DC for RGB or Y00
        // Gaussian Splatting usually uses 48 floats (16 coeffs * 3 channels).
        // If interleaved (R,G,B, R,G,B...), then 0=R, 1=G, 2=B.
        splat.sh_coeffs[0] = 2.0; // Boost Red
        splat.sh_coeffs[1] = -1.0; // Suppress Green
        splat.sh_coeffs[2] = -1.0; // Suppress Blue

        splats.push(splat);
    }

    splats
}
