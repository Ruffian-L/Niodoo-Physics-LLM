use candle_core::{Device, Result, Tensor};
use std::f64::consts::PI;

pub struct TorsionEngine {
    device: Device,
    dim: usize,
}

impl TorsionEngine {
    pub fn new(device: &Device, dim: usize) -> Self {
        Self {
            device: device.clone(),
            dim,
        }
    }

    /// Torsion Binding: Uses Geometric Algebra Rotors to bind two vectors.
    /// A binds B -> We use the Bivector A^B to create a Rotor, then Apply it to B?
    /// Or A * B (Geometric Product).
    /// Let's try: Create a Rotor R from the angle between A and B, in the plane A^B.
    /// Then Encode = R * B * ~R (Rotation of B).
    ///
    /// Actually, to distinguish "Dog Bites Man" from "Man Bites Dog":
    /// We want (Dog, Bites) -> Order Matters.
    ///
    /// Simple Rotor Binding:
    /// Calculate Bivector B_iv = u ^ v (Outer product).
    /// Rotor R = exp( -B_iv * theta/2 ).
    /// Result = R * v * R_rev.
    ///
    /// For this prototype, we'll implementing explicit 3D GA logic on standard Tensors (Force 3D).
    pub fn bind_torsion(&self, u: &Tensor, v: &Tensor) -> Result<Tensor> {
        // Assume u, v are shape [3] (Single vector) or [N, 3] (Batch).
        // Let's implement for single vectors first for clarity.

        let u_vec = u.flatten_all()?.to_vec1::<f32>()?;
        let v_vec = v.flatten_all()?.to_vec1::<f32>()?;

        if u_vec.len() != 3 || v_vec.len() != 3 {
            return Err(candle_core::Error::Msg(
                "Torsion only supported for 3D vectors currently".into(),
            ));
        }

        let ux = u_vec[0] as f64;
        let uy = u_vec[1] as f64;
        let uz = u_vec[2] as f64;

        let vx = v_vec[0] as f64;
        let vy = v_vec[1] as f64;
        let vz = v_vec[2] as f64;

        // 1. Calculate Bivector u ^ v
        // B_yz = uy*vz - uz*vy
        // B_zx = uz*vx - ux*vz
        // B_xy = ux*vy - uy*vx
        let b_yz = uy * vz - uz * vy;
        let b_zx = uz * vx - ux * vz;
        let b_xy = ux * vy - uy * vx;

        // Magnitude of bivector (Area) = |u||v|sin(theta)
        let area = (b_yz * b_yz + b_zx * b_zx + b_xy * b_xy).sqrt();

        // If area is 0 (parallel), no torsion.
        if area < 1e-6 {
            return Ok(v.clone());
        }

        // 2. Create Rotor exp( -B * theta / 2 )
        // Normalized Bivector I
        let i_yz = b_yz / area;
        let i_zx = b_zx / area;
        let i_xy = b_xy / area;

        // Angle? Let's use a fixed "Twist" angle for syntax, e.g., 90 degrees (PI/2).
        // Or depend on the magnitude? Let's fix it to PI/4 per binding step.
        let theta = PI / 4.0;
        let alpha = theta / 2.0;

        let cos_a = alpha.cos();
        let sin_a = alpha.sin();

        // Rotor R = cos(a) - I * sin(a)
        // R = s + B_yz*e23 + B_zx*e31 + B_xy*e12
        let r_s = cos_a;
        let r_yz = -i_yz * sin_a;
        let r_zx = -i_zx * sin_a;
        let r_xy = -i_xy * sin_a;

        // 3. Rotate v: v' = R v ~R
        // This is complex to expand manually.
        // Formula for rotation of vector x by unit bivector I with angle theta:
        // x' = x*cos(theta) + (x . I)*sin(theta)? No.
        // x' = x_par + x_perp * exp(-I * theta).
        // x' = x*cos(theta) + (x . I)*sin(theta) is almost right for cross product?
        // Let's use the explicit R v ~R multiplication.

        // v = x*e1 + y*e2 + z*e3
        // R = s + yz*e23 + zx*e31 + xy*e12

        // Rv computation... this is tedious.
        // Rodrigues rotation formula is equivalent for 3D!
        // Axis of rotation k is the dual of B.
        // B = u^v. Dual(B) is vector perpendicular to u and v.
        // k = (B_yz, B_zx, B_xy) normalized? No.
        // In 3D: *(u^v) = cross(u, v).
        // So axis k = cross(u, v).

        // Let's use Rodrigues formula:
        // v_rot = v cos(th) + (k x v) sin(th) + k (k . v) (1 - cos(th))
        // Here k = cross(u, v) normalized?
        // Wait, binding A onto B usually means "twist B relative to A".
        // The plane of twist is defined by A and B.
        // So the axis is indeed A x B.
        // So we are rotating v *around* the normal to the plane A-B.
        // But v lies *in* that plane (mostly).
        // So rotating v around (u x v) keeps it in the plane u-v.
        // It rotates v towards u (or away).

        // Is this non-commutative?
        // u bind v -> Rotate v by +theta in u-v plane.
        // v bind u -> Rotate u by +theta in v-u plane (which is -theta in u-v plane).
        // Result is different vector.
        // Checks out.

        let k_x = b_yz / area;
        let k_y = b_zx / area;
        let k_z = b_xy / area;

        // v_rot
        let cross_kv_x = k_y * vz - k_z * vy;
        let cross_kv_y = k_z * vx - k_x * vz;
        let cross_kv_z = k_x * vy - k_y * vx;

        let dot_kv = k_x * vx + k_y * vy + k_z * vz; // Should be 0 if v is in plane, but k is normal to v.
                                                     // Yes, k = u x v, so k is perp to v. dot_kv = 0.

        // So formula simplifies:
        // v_rot = v * cos(theta) + (k x v) * sin(theta).

        let theta_full = theta; // The rotation angle
        let c = theta_full.cos();
        let s = theta_full.sin();

        let rx = vx * c + cross_kv_x * s;
        let ry = vy * c + cross_kv_y * s;
        let rz = vz * c + cross_kv_z * s;

        let res_vec = vec![rx as f32, ry as f32, rz as f32];
        Tensor::from_vec(res_vec, (3,), &self.device)
    }
}
