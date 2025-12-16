use super::GaussianSplat;

pub struct Disentangled4DGS {
    pub static_gaussians: Vec<GaussianSplat>,
    pub dynamic_gaussians: Vec<GaussianSplat>,
    pub time_range: (f32, f32),
}

impl Disentangled4DGS {
    pub fn new() -> Self {
        Self {
            static_gaussians: Vec::new(),
            dynamic_gaussians: Vec::new(),
            time_range: (0.0, 1.0),
        }
    }

    pub fn add_static(&mut self, splat: GaussianSplat) {
        self.static_gaussians.push(splat);
    }

    pub fn add_dynamic(&mut self, splat: GaussianSplat) {
        if !splat.is_4d() {
            tracing::warn!("Adding non-4D splat to dynamic set");
        }
        self.dynamic_gaussians.push(splat);
    }

    pub fn at_time(&self, t: f32) -> Vec<[f32; 3]> {
        let mut positions = Vec::new();

        for splat in &self.static_gaussians {
            positions.push(splat.position.to_array());
        }

        for splat in &self.dynamic_gaussians {
            if let Some(vel) = splat.velocity {
                let pos = splat.position + vel * t;
                positions.push(pos.to_array());
            }
        }

        positions
    }

    pub fn total_splats(&self) -> usize {
        self.static_gaussians.len() + self.dynamic_gaussians.len()
    }

    pub fn motion_energy(&self) -> f32 {
        crate::utils::fidelity::robust_sum(
            self.dynamic_gaussians
                .iter()
                .filter_map(|s| s.velocity)
                .map(|v| v.length()),
        )
    }
}

impl Default for Disentangled4DGS {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    #[test]
    fn test_4dgs_creation() {
        let gs = Disentangled4DGS::new();
        assert_eq!(gs.total_splats(), 0);
    }

    #[test]
    fn test_add_splats() {
        let mut gs = Disentangled4DGS::new();

        let static_splat = GaussianSplat::new(Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, 1.0);

        let dynamic_splat = GaussianSplat::new(Vec3::X, Vec3::ONE, Quat::IDENTITY, 1.0)
            .with_velocity(Vec3::new(0.1, 0.0, 0.0));

        gs.add_static(static_splat);
        gs.add_dynamic(dynamic_splat);

        assert_eq!(gs.total_splats(), 2);
        assert_eq!(gs.static_gaussians.len(), 1);
        assert_eq!(gs.dynamic_gaussians.len(), 1);
    }

    #[test]
    fn test_time_evolution() {
        let mut gs = Disentangled4DGS::new();

        let dynamic_splat =
            GaussianSplat::new(Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, 1.0).with_velocity(Vec3::X);

        gs.add_dynamic(dynamic_splat);

        let positions_t0 = gs.at_time(0.0);
        let positions_t1 = gs.at_time(1.0);

        assert_eq!(positions_t0[0][0], 0.0);
        assert_eq!(positions_t1[0][0], 1.0);
    }

    #[test]
    fn test_motion_energy() {
        let mut gs = Disentangled4DGS::new();

        let splat = GaussianSplat::new(Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, 1.0)
            .with_velocity(Vec3::new(3.0, 4.0, 0.0));

        gs.add_dynamic(splat);

        assert_eq!(gs.motion_energy(), 5.0);
    }
}
