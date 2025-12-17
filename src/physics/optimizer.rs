use friedrich::gaussian_process::GaussianProcess;
use friedrich::kernel::SquaredExp;
use friedrich::prior::ConstantPrior;
use rand::Rng;

/// Generalized Physics Parameters for Semantic Gravity
///
/// Mass Formula: M_S = alpha_info * m_info + alpha_sem * m_sem + alpha_coh * m_coh + alpha_emo * m_emo
/// Final Mass: M = M_S * (1.0 + delta_amp * kl_divergence)
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsParams {
    pub gravity: f64,
    pub dt: f64,
    pub repulsion: f64,

    // GENERALIZED SEMANTIC MASS WEIGHTS
    pub alpha_info: f64,      // Information mass (surprisal) weight
    pub alpha_sem: f64,       // Semantic density weight
    pub alpha_coh: f64,       // Coherence to context weight
    pub alpha_struct: f64,    // Syntactic/Structural complexity
    pub alpha_quantum: f64,   // Quantum coherence
    pub alpha_geometric: f64, // Geometric DL score
    pub alpha_emo: f64,       // Emotional mass weight (optional, default low)

    // KL DIVERGENCE (MEANING AS DELTA)
    pub delta_amp: f64, // Amplification factor for KL delta

    // NOVELTY/REDUNDANCY
    pub redundancy_threshold: f64, // If sim > this, particle becomes repulsor

    // DYNAMICS
    pub decay_lambda: f64, // Gravitational redshift over time
    pub mu: f64,           // Mobility (1/viscosity) for Overdamped Langevin
    pub sigma: f64,        // Temperature/Noise scale
    pub momentum: f64,     // Autonomic Inertia

    // FEATURE FLAGS
    pub use_emo: bool, // Enable emotional mass component
    pub pinn_enabled: bool,
    pub pinn_stiffness: f64,
    pub ghost_gravity: f64,
    pub gravity_well_strength: f64, // Multiplier for Centripetal Force
    pub orbit_speed: f64,           // Tangential Velocity
}

impl Default for PhysicsParams {
    fn default() -> Self {
        Self {
            gravity: 0.001,
            dt: 0.05,
            repulsion: -0.001,
            // Semantic mass defaults (sum to ~1.0 for normalization)
            alpha_info: 0.25,
            alpha_sem: 0.20,
            alpha_coh: 0.20,
            alpha_struct: 0.15,
            alpha_quantum: 0.10,
            alpha_geometric: 0.10,
            alpha_emo: 0.05,
            // Delta
            delta_amp: 1.0, // KL divergence multiplier
            // Novelty
            redundancy_threshold: 0.85, // High similarity = repulsor
            // Dynamics
            decay_lambda: 0.01,
            mu: 1.0,
            sigma: 0.1,
            momentum: 0.9,
            // Flags
            use_emo: false,
            pinn_enabled: true,
            pinn_stiffness: 0.1,
            ghost_gravity: 0.001,
            gravity_well_strength: 0.8,
            orbit_speed: 0.2,
        }
    }
}

impl PhysicsParams {
    pub fn new(
        gravity: f64,
        dt: f64,
        repulsion: f64,
        alpha_info: f64,
        alpha_sem: f64,
        alpha_coh: f64,
        alpha_struct: f64,
        alpha_quantum: f64,
        alpha_geometric: f64,
        alpha_emo: f64,
        use_emo: bool,
        decay_lambda: f64,
        mu: f64,
        sigma: f64,
        momentum: f64,
        pinn_enabled: bool,
        pinn_stiffness: f64,
        ghost_gravity: f64,
        gravity_well_strength: f64,
        orbit_speed: f64,
    ) -> Self {
        Self {
            gravity,
            dt,
            repulsion,
            alpha_info,
            alpha_sem,
            alpha_coh,
            alpha_struct,
            alpha_quantum,
            alpha_geometric,
            alpha_emo,
            delta_amp: 1.0,
            redundancy_threshold: 0.85,
            decay_lambda,
            mu,
            sigma,
            momentum,
            use_emo,
            pinn_enabled,
            pinn_stiffness,
            ghost_gravity,
            gravity_well_strength,
            orbit_speed,
        }
    }

    /// Create with full control over all parameters
    pub fn full(
        gravity: f64,
        dt: f64,
        repulsion: f64,
        alpha_info: f64,
        alpha_sem: f64,
        alpha_coh: f64,
        alpha_struct: f64,
        alpha_quantum: f64,
        alpha_geometric: f64,
        alpha_emo: f64,
        delta_amp: f64,
        redundancy_threshold: f64,
        decay_lambda: f64,
        mu: f64,
        sigma: f64,
        momentum: f64,
        use_emo: bool,
        pinn_enabled: bool,
        pinn_stiffness: f64,
        ghost_gravity: f64,
        gravity_well_strength: f64,
        orbit_speed: f64,
    ) -> Self {
        Self {
            gravity,
            dt,
            repulsion,
            alpha_info,
            alpha_sem,
            alpha_coh,
            alpha_struct,
            alpha_quantum,
            alpha_geometric,
            alpha_emo,
            delta_amp,
            redundancy_threshold,
            decay_lambda,
            mu,
            sigma,
            momentum,
            use_emo,
            pinn_enabled,
            pinn_stiffness,
            ghost_gravity,
            gravity_well_strength,
            orbit_speed,
        }
    }

    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.gravity,
            self.dt,
            self.repulsion,
            self.alpha_info,
            self.alpha_sem,
            self.decay_lambda,
            self.mu,
            self.sigma,
            self.momentum,
        ]
    }
}

pub struct PhysicsOptimizer {
    gp: Option<GaussianProcess<SquaredExp, ConstantPrior>>, // Initialized after few samples
    history_params: Vec<Vec<f64>>,
    history_scores: Vec<f64>,
    bounds_min: Vec<f64>,
    bounds_max: Vec<f64>,
}

impl PhysicsOptimizer {
    pub fn new() -> Self {
        Self {
            gp: None,
            history_params: Vec::new(),
            history_scores: Vec::new(),
            // Bounds: [Grav, DT, Repuls, A_Info, A_Sem, A_Coh, A_Str, A_Qnt, A_Geo, A_Emo, UseEmo, Lambda, Mu, Sigma, Mom]
            // 15 Dimensions
            bounds_min: vec![
                0.1, 0.05, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.01, 0.5,
            ],
            bounds_max: vec![
                20.0, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.99,
            ],
        }
    }

    pub fn suggest_next_params(&self) -> PhysicsParams {
        let mut rng = rand::thread_rng();

        // Helper to check if we are in exploration or random mode
        let random_sample = self.history_params.len() < 5 || rng.gen_bool(0.1);

        if random_sample {
            return self.random_params(&mut rng);
        }

        // Optimization on Surface (Random Shooting)
        let num_candidates = 100;
        let mut best_score = -f64::INFINITY;
        let mut best_candidate = vec![0.0; 15];

        if let Some(gp) = &self.gp {
            for _ in 0..num_candidates {
                let candidate: Vec<f64> = (0..15)
                    .map(|i| rng.gen_range(self.bounds_min[i]..=self.bounds_max[i]))
                    .collect();

                let prediction = gp.predict(&vec![candidate.clone()]);
                let mean = prediction[0];
                let score = mean;

                if score > best_score {
                    best_score = score;
                    best_candidate = candidate;
                }
            }
            self.vec_to_params(&best_candidate)
        } else {
            self.random_params(&mut rng)
        }
    }

    fn random_params(&self, rng: &mut impl Rng) -> PhysicsParams {
        let c: Vec<f64> = (0..15)
            .map(|i| rng.gen_range(self.bounds_min[i]..=self.bounds_max[i]))
            .collect();
        self.vec_to_params(&c)
    }

    fn vec_to_params(&self, p: &[f64]) -> PhysicsParams {
        PhysicsParams::new(
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            p[5],
            p[6],
            p[7],
            p[8],
            p[9],
            p[10] > 0.5, // use_emo
            p[11],
            p[12],
            p[13],
            p.get(14).copied().unwrap_or(0.9),
            true,                               // pinn_enabled default for optimizer
            0.1,                                // pinn_stiffness default for optimizer
            p.get(0).copied().unwrap_or(0.001), // ghost_gravity uses same as gravity for optimizer
            0.8,                                // gravity_well_strength default
            0.2,                                // orbit_speed default
        )
    }

    pub fn update(&mut self, params: PhysicsParams, score: f64) {
        self.history_params.push(params.to_vec());
        self.history_scores.push(score);

        if self.history_params.len() >= 5 {
            // Retrain GP
            // Inputs must be Vec<Vec<f64>>
            let gp =
                GaussianProcess::default(self.history_params.clone(), self.history_scores.clone());
            self.gp = Some(gp);
        }
    }

    pub fn best_params(&self) -> Option<PhysicsParams> {
        // Find index of max score
        let (idx, _) = self
            .history_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;
        let p = &self.history_params[idx];
        Some(self.vec_to_params(p))
    }
}
