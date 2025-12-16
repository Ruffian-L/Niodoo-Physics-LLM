// src/language/g_prime.rs v1.3 – Semantic Tone Edition

use crate::constants::{GPRIME_SCALE_RATIOS, PHONEME_SPACE, VALENCE_SCALE_FACTOR};
use crate::structs::SplatGeometry;
use glam::{Quat, Vec3};
use std::f32::consts::PI;

pub struct GPrimeCodecV1;

impl GPrimeCodecV1 {
    /// Encode a Unicode codepoint + emotional metadata into a single splat "syllable"
    pub fn encode_glyph(
        codepoint: u32,
        tone: u8,
        confidence: f32,
        position: Vec3,
    ) -> SplatGeometry {
        let phoneme = (codepoint as u16) & 0x7FFF; // 15 bits

        // Diagonal: encode phoneme across scale.x/y/z with redundancy
        let s = Self::phoneme_to_scale(phoneme);
        let scale = Vec3::new(
            s * GPRIME_SCALE_RATIOS[0],
            s * GPRIME_SCALE_RATIOS[1],
            s * GPRIME_SCALE_RATIOS[2],
        );

        // Rotation: Encode Tone bits into Yaw/Pitch/Roll
        let rot_quat = Self::tone_to_quat(tone);

        // Opacity from confidence (0..255)
        let opacity = (confidence.clamp(0.0, 1.0) * 255.0) as u8;

        // Valence from Tone
        let sentiment = (tone >> 3) & 0x0F;
        let valence_byte = (sentiment * 17) as u8; // 15*17 = 255

        let color_rgba = [128, 128, 128, opacity];

        // Pack into SoA Struct
        SplatGeometry {
            position: [position.x, position.y, position.z],
            scale: [scale.x, scale.y, scale.z],
            rotation: [rot_quat.x, rot_quat.y, rot_quat.z, rot_quat.w], // xyzw layout matches glam
            color_rgba,
            physics_props: [128, 0, valence_byte, 0],
            domain_valence: [0.25, 0.25, 0.25, 0.25], // Neutral for phoneme encoding
        }
    }

    pub fn decode_glyph(splat: &SplatGeometry) -> (char, u8, f32) {
        // 1. Extract Scale
        let s = if splat.scale[0] > 0.0 {
            splat.scale[0] / GPRIME_SCALE_RATIOS[0]
        } else {
            0.0
        };
        let phoneme = Self::scale_to_phoneme(s);

        // 2. Extract Tone from Rotation
        let rot_quat = Quat::from_array([
            splat.rotation[0],
            splat.rotation[1],
            splat.rotation[2],
            splat.rotation[3],
        ]);
        let tone = Self::quat_to_tone(rot_quat);
        let confidence = splat.color_rgba[3] as f32 / 255.0;

        // 3. Map phoneme back to Unicode
        let c = char::from_u32(phoneme as u32).unwrap_or('\0');

        (c, tone, confidence)
    }

    // Kept for compatibility if called explicitly, otherwise decode_glyph handles SplatGeometry
    pub fn decode_glyph_geom(splat: &SplatGeometry) -> (char, u8, f32) {
        Self::decode_glyph(splat)
    }

    fn phoneme_to_scale(p: u16) -> f32 {
        0.5 + (p as f32 / PHONEME_SPACE as f32) * 4.0
    }

    fn scale_to_phoneme(s: f32) -> u16 {
        let normalized = (s - 0.5) / 4.0;
        let clamped = normalized.clamp(0.0, 1.0);
        (clamped * PHONEME_SPACE as f32).round() as u16
    }

    fn tone_to_quat(tone: u8) -> Quat {
        let is_caps = (tone & 0x80) != 0;
        let sentiment = (tone >> 3) & 0x0F;
        let uncertainty = tone & 0x07;

        let yaw = if is_caps { PI / 2.0 } else { 0.0 };

        let pitch_deg = (sentiment as f32 / 15.0) * 90.0 - 45.0;
        let pitch = pitch_deg.to_radians();

        let roll_deg = (uncertainty as f32 / 7.0) * 35.0;
        let roll = roll_deg.to_radians();

        Quat::from_axis_angle(Vec3::Y, yaw)
            * Quat::from_axis_angle(Vec3::X, pitch)
            * Quat::from_axis_angle(Vec3::Z, roll)
    }

    fn quat_to_tone(q: Quat) -> u8 {
        let (yaw, pitch, roll) = q.to_euler(glam::EulerRot::YXZ);

        // 1. Caps (Yaw)
        let yaw_norm = yaw.rem_euclid(2.0 * PI);
        let is_caps = yaw_norm > (PI / 4.0) && yaw_norm < (3.0 * PI / 4.0);

        // 2. Sentiment (Pitch)
        let pitch_clamped = pitch.clamp(-PI / 4.0, PI / 4.0);
        let sent_norm = (pitch_clamped + PI / 4.0) / (PI / 2.0);
        let sentiment = (sent_norm * 15.0).round() as u8;

        // 3. Uncertainty (Roll)
        let roll_max = 35.0f32.to_radians();
        let roll_clamped = roll.abs().clamp(0.0, roll_max);
        let unc_norm = roll_clamped / roll_max;
        let uncertainty = (unc_norm * 7.0).round() as u8;

        let mut tone = 0u8;
        if is_caps {
            tone |= 0x80;
        }
        tone |= (sentiment & 0x0F) << 3;
        tone |= uncertainty & 0x07;

        tone
    }

    #[allow(dead_code)]
    fn tone_to_valence(tone: u8) -> f32 {
        // Extract sentiment bits (3-6)
        let sentiment = (tone >> 3) & 0x0F; // 0-15
                                            // Map 0..15 -> -1.0..1.0
        ((sentiment as f32 / 15.0) * 2.0 - 1.0) * VALENCE_SCALE_FACTOR
    }
}
