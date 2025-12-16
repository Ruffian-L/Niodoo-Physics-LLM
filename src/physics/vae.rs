use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

/// The Manifold VAE maps physical properties to a latent geometric space.
pub struct ManifoldVAE {
    // Encoder Layers
    enc_fc1: Linear,
    enc_mu: Linear,     // Head for Mean
    enc_logvar: Linear, // Head for Log-Variance

    // Decoder Layers
    dec_fc1: Linear,
    dec_out: Linear,
}

impl ManifoldVAE {
    /// Constructs the VAE with specified dimensions.
    pub fn new(
        vs: VarBuilder,
        input_dim: usize,  // e.g., 4 (1 Mass + 3 Charge)
        hidden_dim: usize, // e.g., 64
        latent_dim: usize, // e.g., 16
    ) -> Result<Self> {
        // Initialize Encoder
        let enc_fc1 = linear(input_dim, hidden_dim, vs.pp("enc_fc1"))?;
        let enc_mu = linear(hidden_dim, latent_dim, vs.pp("enc_mu"))?;
        let enc_logvar = linear(hidden_dim, latent_dim, vs.pp("enc_logvar"))?;

        // Initialize Decoder
        let dec_fc1 = linear(latent_dim, hidden_dim, vs.pp("dec_fc1"))?;
        let dec_out = linear(hidden_dim, input_dim, vs.pp("dec_out"))?;

        Ok(Self {
            enc_fc1,
            enc_mu,
            enc_logvar,
            dec_fc1,
            dec_out,
        })
    }

    /// Executes the reparameterization trick to sample from the latent distribution.
    ///
    /// # Arguments
    /// * `mu` - The mean tensor.
    /// * `logvar` - The log-variance tensor.
    fn reparameterize(&self, mu: &Tensor, logvar: &Tensor) -> Result<Tensor> {
        // Step 1: Derive Sigma
        // sigma = exp(0.5 * logvar)
        // We multiply logvar by 0.5 first to reduce magnitude before exponentiation,
        // which helps numerical stability.
        let half_logvar = (logvar * 0.5)?;
        let sigma = half_logvar.exp()?;

        // Step 2: Sample Epsilon (Noise)
        // We generate a tensor of the same shape as sigma, filled with values
        // drawn from a standard normal distribution N(0, 1).
        // Note: randomness is handled by the device's RNG.
        let epsilon = Tensor::randn_like(&sigma, 0.0, 1.0)?;

        // Step 3: Scale and Shift
        // z = mu + (sigma * epsilon)
        // The operations are element-wise.
        let scaled_noise = (sigma * epsilon)?;
        let z = (mu + scaled_noise)?;

        Ok(z)
    }

    /// Forward pass through the entire VAE.
    /// Returns: (Reconstructed Input, Mean, LogVariance)
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // Encode
        let h = self.enc_fc1.forward(x)?;
        let h_act = h.relu()?;

        // Predict Latent Parameters
        let mu = self.enc_mu.forward(&h_act)?;
        let logvar = self.enc_logvar.forward(&h_act)?;

        // Sample Latent Vector
        let z = self.reparameterize(&mu, &logvar)?;

        // Decode
        let dec_h = self.dec_fc1.forward(&z)?;
        let dec_h_act = dec_h.relu()?;
        let reconstruction = self.dec_out.forward(&dec_h_act)?;

        // Note: If input data is normalized to , a Sigmoid might be appended here.
        // Since our data spans [-0.5, 0.5], a Tanh or linear output is appropriate.
        // Here we leave it linear, assuming the loss function handles the range.

        Ok((reconstruction, mu, logvar))
    }
}
