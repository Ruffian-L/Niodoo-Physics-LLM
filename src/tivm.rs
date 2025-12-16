use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VpbWeightFn {
    Uniform,
    Gaussian,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VpbParams {
    pub grid_res: (usize, usize),
    pub birth_range: (f64, Option<f64>),
    pub death_range: (f64, Option<f64>),
    pub weight_fn: VpbWeightFn,
}

impl Default for VpbParams {
    fn default() -> Self {
        Self {
            grid_res: (32, 32),
            birth_range: (0.0, None),
            death_range: (0.0, None),
            weight_fn: VpbWeightFn::Uniform,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplatRagConfig {
    pub hom_dims: Vec<usize>,
    pub vpb_params: VpbParams,
    pub proto_mode: bool,
    pub flood_mode: bool,
    pub ef_search: usize,
    pub api_key: Option<String>,
}

impl Default for SplatRagConfig {
    fn default() -> Self {
        Self {
            hom_dims: vec![0, 1],
            vpb_params: VpbParams::default(),
            proto_mode: false,
            flood_mode: false,
            ef_search: 64,
            api_key: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SplatRagBuilder {
    config: SplatRagConfig,
}

impl SplatRagBuilder {
    pub fn new() -> Self {
        Self {
            config: SplatRagConfig::default(),
        }
    }

    pub fn with_hom_dims(mut self, hom_dims: Vec<usize>) -> Self {
        self.config.hom_dims = hom_dims;
        self
    }

    pub fn with_vpb(mut self, vpb_params: VpbParams) -> Self {
        self.config.vpb_params = vpb_params;
        self
    }

    pub fn with_proto_mode(mut self, proto_mode: bool) -> Self {
        self.config.proto_mode = proto_mode;
        self
    }

    pub fn with_flood_mode(mut self, flood_mode: bool) -> Self {
        self.config.flood_mode = flood_mode;
        self
    }

    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.config.ef_search = ef_search;
        self
    }

    pub fn build(self) -> SplatRagConfig {
        self.config
    }
}
