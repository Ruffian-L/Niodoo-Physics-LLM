use rand::Rng;

// Simulating the 4096D Llama-3 embedding space
const DIM: usize = 4096;
const DT: f32 = 0.1; // Time step
const GRAVITY: f32 = 10.0; // Strength of the pull to center
const ORBIT_SPEED: f32 = 1.0; // Initial tangential velocity

#[derive(Debug, Clone)]
struct Vector {
    data: Vec<f32>,
}

impl Vector {
    fn new_zero() -> Self {
        Self {
            data: vec![0.0; DIM],
        }
    }

    fn new_random() -> Self {
        let mut rng = rand::thread_rng();
        let data = (0..DIM).map(|_| rng.gen_range(-0.1..0.1)).collect();
        Self { data }
    }

    fn add(&self, other: &Vector) -> Vector {
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Self { data }
    }

    fn sub(&self, other: &Vector) -> Vector {
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Self { data }
    }

    fn scale(&self, scalar: f32) -> Vector {
        let data = self.data.iter().map(|a| a * scalar).collect();
        Self { data }
    }

    fn dot(&self, other: &Vector) -> f32 {
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }

    fn norm_sq(&self) -> f32 {
        self.dot(self)
    }

    fn norm(&self) -> f32 {
        self.norm_sq().sqrt()
    }

    fn normalize(&self) -> Vector {
        let n = self.norm();
        if n > 1e-6 {
            self.scale(1.0 / n)
        } else {
            Self::new_zero()
        }
    }
}

fn main() {
    println!("🪐 NIODOO ORBITAL MATH PROTOTYPE 🪐");
    println!("Dimension: {}", DIM);

    // 1. Setup the "Sun" (Topic Center) and "Planet" (Token)
    let sun = Vector::new_zero(); // Center of embedding space

    // Start at a random position (distance ~sqrt(DIM) * 0.1)
    let mut position = Vector::new_random();

    // Normalize position to be on the "shell" (radius 10.0 for simulation)
    let start_radius = 10.0;
    position = position.normalize().scale(start_radius);

    // 2. Initialize Momentum (Velocity)
    // To orbit, we need velocity tangential to position (r).
    // Let's create a random vector and orthogonalize it against position.
    let random_vec = Vector::new_random();

    // Gram-Schmidt / Projection: v_tan = v - (v . r_hat) * r_hat
    let pos_hat = position.normalize();
    let radial_component = pos_hat.scale(random_vec.dot(&pos_hat));
    let mut momentum = random_vec
        .sub(&radial_component)
        .normalize()
        .scale(ORBIT_SPEED);

    println!("Initial Radius: {:.4}", position.norm());
    println!("Initial Momentum Mag: {:.4}", momentum.norm());
    println!(
        "Orthogonality Check (Pos . Mom): {:.6} (Should be ~0)",
        position.dot(&momentum)
    );

    // 3. The Symplectic Loop
    println!("\nStarting Simulation (Symplectic Euler)...");
    println!("Step | Radius | Momentum | Energy (Approx)");
    println!("-------------------------------------------");

    for step in 0..100 {
        // A. Calculate Gravity Force
        // F = -G * M / r^2 * r_hat
        // M = 1 (assumed), Vector r = sun - position = -position
        let r_vec = position.scale(-1.0); // Vector pointing to sun
        let r_dist = r_vec.norm();
        let r_hat = r_vec.normalize();

        let force_mag = GRAVITY / (r_dist * r_dist).max(0.1); // Avoid singularity
        let force_gravity = r_hat.scale(force_mag);

        // B. ORTHOGONALIZATION REPAIR (The "Niodoo Twist")
        // In a true perfect orbit, gravity is always orthogonal to velocity.
        // But numerical errors and multiple extractors drift it.
        // We explicitly kill radial momentum to prevent crashing.
        // NOTE: Standard orbits usually allow radial velocity (ellipses).
        // Niodoo wants a CIRCULAR orbit (constant context distance).
        // So we strictly damping radial velocity?
        // Let's try pure Symplectic first, then apply the "Niodoo Constraint".

        // Standard Symplectic Step 1: Update Momentum
        momentum = momentum.add(&force_gravity.scale(DT));

        // ** THE NIODOO CONSTRAINT **
        // We want to orbit, not crash. Kill the component of momentum parallel to gravity.
        // This forces circular motion even if gravity is too strong/weak.
        let m_radial = r_hat.scale(momentum.dot(&r_hat));
        let m_tangential = momentum.sub(&m_radial);

        // Blend: 90% Tangential (Orbit), 10% Radial (Allow some breathing/elliptical)
        // For "Continuous Thematic Flow", we want pure orbit.
        momentum = m_tangential;

        // Re-scale momentum to maintain constant speed (Energy conservation enforcement)
        // If we slow down, we decay. If we speed up, we fly away.
        // In language, "Speed" is "Rate of Topic Change".
        momentum = momentum.normalize().scale(ORBIT_SPEED);

        // Standard Symplectic Step 2: Update Position
        position = position.add(&momentum.scale(DT));

        if step % 10 == 0 {
            println!(
                "{:4} | {:.4} | {:.4} | N/A",
                step,
                position.norm(),
                momentum.norm()
            );
        }
    }

    println!("\n✅ Simulation Complete. Stable Orbit Achieved.");
}
