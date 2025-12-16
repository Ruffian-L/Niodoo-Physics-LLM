use crate::structs::PackedSemantics;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PSCComponent {
    // 768 floats packed into 192 vec4s
    pub features: [[f32; 4]; 192],
}

unsafe impl bytemuck::Zeroable for PSCComponent {}
unsafe impl bytemuck::Pod for PSCComponent {}

impl PSCComponent {
    pub fn zero() -> Self {
        Self {
            features: [[0.0; 4]; 192],
        }
    }

    pub fn from_slice(slice: &[f32]) -> Self {
        let mut features = [[0.0; 4]; 192];
        for i in 0..192 {
            for j in 0..4 {
                if let Some(&val) = slice.get(i * 4 + j) {
                    features[i][j] = val;
                }
            }
        }
        Self { features }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    pub padding: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RenderedFeature {
    pub final_feature: [[f32; 4]; 192],
    pub projected_pos: [f32; 2],
    pub final_alpha: f32,
    pub _pad: f32,
}

unsafe impl bytemuck::Zeroable for RenderedFeature {}
unsafe impl bytemuck::Pod for RenderedFeature {}

pub struct M3Compute {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub psc_buffer: wgpu::Buffer,
    pub pipeline: wgpu::ComputePipeline,
    pub camera_buffer: wgpu::Buffer,
    pub memory_buffer: wgpu::Buffer,
    pub output_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl M3Compute {
    pub fn new(
        device: &wgpu::Device,
        psc_data: &[PSCComponent],
        initial_memories: &[PackedSemantics],
    ) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);
        let psc_buffer = Self::create_psc_buffer(device, psc_data);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("M3 Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/m3_compute.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("M3 Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("M3 Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        // Initialize buffers
        let camera_uniforms = CameraUniforms {
            view: [[0.0; 4]; 4], // Identity-ish
            proj: [[0.0; 4]; 4],
            camera_position: [0.0; 3],
            padding: 0.0,
        };
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Uniforms"),
            contents: bytemuck::cast_slice(&[camera_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Memory Buffer
        // If initial_memories is empty, create a dummy one to avoid validation errors?
        // Or just handle empty case.
        let memory_contents = if initial_memories.is_empty() {
            // Create a dummy memory to satisfy buffer creation if needed, or just empty.
            // WGPU buffers can be empty? No, usually need non-zero size for binding if used.
            // Let's create at least one dummy if empty.
            vec![PackedSemantics::default()]
        } else {
            initial_memories.to_vec()
        };

        let memory_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussian Memory Buffer"),
            contents: bytemuck::cast_slice(&memory_contents),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Output Buffer
        let output_count = memory_contents.len();
        let output_size = (output_count * std::mem::size_of::<RenderedFeature>()) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Feature Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("M3 Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: memory_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: psc_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            bind_group_layout,
            psc_buffer,
            pipeline,
            camera_buffer,
            memory_buffer,
            output_buffer,
            bind_group,
        }
    }

    pub fn update_memory_bank(&mut self, device: &wgpu::Device, memories: &[PackedSemantics]) {
        // Recreate memory buffer and output buffer and bind group
        let memory_contents = if memories.is_empty() {
            vec![PackedSemantics::default()]
        } else {
            memories.to_vec()
        };

        self.memory_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gaussian Memory Buffer"),
            contents: bytemuck::cast_slice(&memory_contents),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let output_count = memory_contents.len();
        let output_size = (output_count * std::mem::size_of::<RenderedFeature>()) as u64;
        self.output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Feature Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("M3 Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.memory_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.psc_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.output_buffer.as_entire_binding(),
                },
            ],
        });
    }

    pub fn upload_psc(&mut self, device: &wgpu::Device, psc_data: &[PSCComponent]) {
        self.psc_buffer = Self::create_psc_buffer(device, psc_data);
        // Need to recreate bind group because psc_buffer changed
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("M3 Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.memory_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.psc_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.output_buffer.as_entire_binding(),
                },
            ],
        });
    }

    pub async fn execute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        num_gaussians: u32,
    ) -> Vec<RenderedFeature> {
        // 1. Setup Command Encoder
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("M3 Compute Encoder"),
        });

        // 2. The Compute Pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("M3 Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);

            // Dispatch: Ceiling division to ensure we cover all Gaussians
            // Workgroup size is usually 256 in the shader (@workgroup_size(256, 1, 1))
            let workgroup_count = (num_gaussians as f32 / 256.0).ceil() as u32;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // 3. Copy to Staging Buffer for Readback
        // We need to calculate size: num_gaussians * size_of::<RenderedFeature>()
        // Note: num_gaussians passed here should match what's in the buffer, or at least not exceed it.
        // We use the actual buffer size to be safe or the passed num.
        // Let's use the passed num but ensure it fits.
        let output_size = (num_gaussians as u64) * std::mem::size_of::<RenderedFeature>() as u64;

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("M3 Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0, // Source (GPU Storage)
            &staging_buffer,
            0, // Dest (CPU Staging)
            output_size,
        );

        // 4. Submit and Wait
        queue.submit(Some(encoder.finish()));

        // Create the future for mapping
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = tokio::sync::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });

        // POLLING: Critical for non-async runtimes or blocking ops
        device.poll(wgpu::Maintain::Wait);

        // 5. Read and Deserialize
        if let Ok(Ok(())) = receiver.await {
            let data = buffer_slice.get_mapped_range();
            // Unsafe cast from bytes to RenderedFeature structs
            let result: Vec<RenderedFeature> = unsafe {
                let (prefix, body, suffix) = data.align_to::<RenderedFeature>();
                if !prefix.is_empty() || !suffix.is_empty() {
                    panic!("Alignment error in shader readback");
                }
                body.to_vec()
            };

            drop(data); // Release lock
            staging_buffer.unmap();

            return result;
        } else {
            panic!("Failed to map GPU memory for readback!");
        }
    }

    pub fn update_camera(
        &self,
        queue: &wgpu::Queue,
        prompt_embedding: &[f32],          // The 768-dim query
        projection_matrix: &[[f32; 4]; 4], // Your TDA/CSMP matrix
    ) {
        // 1. Project Prompt to 3D Semantic Space (Simple MatMul for now)
        // This places the "mind's eye" inside the memory cluster relevant to the topic.
        // For now, just take first 3 dims if available, or 0.
        let x = prompt_embedding.get(0).copied().unwrap_or(0.0);
        let y = prompt_embedding.get(1).copied().unwrap_or(0.0);
        let z = prompt_embedding.get(2).copied().unwrap_or(0.0);

        let uniform_data = CameraUniforms {
            view: *projection_matrix, // Or look_at(target)
            proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ], // Identity/Orthographic
            camera_position: [x, y, z],
            padding: 0.0,
        };

        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[uniform_data]),
        );
    }

    pub fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("M3 Compute Bind Group Layout"),
            entries: &[
                // Binding 0: Camera Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Gaussian Storage (ReadOnly)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: PSC Bank (ReadOnly)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Output Buffer (ReadWrite)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn create_psc_buffer(device: &wgpu::Device, data: &[PSCComponent]) -> wgpu::Buffer {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PSC Bank Buffer"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }
}
