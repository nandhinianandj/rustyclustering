//! GPU-accelerated DBSCAN implementation.

use ndarray::{ArrayView2, Array1};
use crate::gpu_context::{GpuContext, GpuError};
use crate::utils::{workgroup_count, array_to_flat_f32};
use ghdbscan_core::DistanceMetric;
use std::sync::Arc;

/// Parameters for GPU-accelerated DBSCAN.
#[derive(Debug, Clone)]
pub struct DBSCANParams {
    pub eps: f64,
    pub min_samples: usize,
    pub metric: DistanceMetric,
}

impl DBSCANParams {
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            metric: DistanceMetric::Euclidean,
        }
    }

    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }
}

/// Result of GPU DBSCAN clustering.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    pub labels: Array1<i32>,
    pub core_points: Array1<bool>,
}

impl ClusterResult {
    pub fn n_clusters(&self) -> usize {
        let max_label = self.labels.iter().max().copied().unwrap_or(-1);
        if max_label < 0 {
            0
        } else {
            (max_label + 1) as usize
        }
    }

    pub fn n_noise(&self) -> usize {
        self.labels.iter().filter(|&&label| label == -1).count()
    }
}

/// GPU-accelerated DBSCAN clustering algorithm.
pub struct DBSCAN_GPU {
    params: DBSCANParams,
    context: Arc<GpuContext>,
}

impl DBSCAN_GPU {
    /// Create a new GPU DBSCAN instance.
    pub async fn new(params: DBSCANParams) -> Result<Self, GpuError> {
        let context = Arc::new(GpuContext::new().await?);
        Ok(Self { params, context })
    }

    /// Fit the DBSCAN algorithm to the data using GPU acceleration.
    pub async fn fit(&self, data: &ArrayView2<'_, f64>) -> Result<ClusterResult, GpuError> {
        let n_points = data.nrows();
        let n_features = data.ncols();

        // For very small datasets, fall back to CPU
        if n_points < 100 {
            return self.fit_cpu(data);
        }

        // Convert data to f32 for GPU
        let data_flat = array_to_flat_f32(data);

        // Step 1: Compute adjacency matrix on GPU (fused distance + threshold)
        // Returns a bit-packed adjacency matrix (N x ceil(N/32) u32s)
        let adjacency = self.compute_adjacency_matrix_gpu(&data_flat, n_points, n_features).await?;

        // Step 2: Perform clustering on CPU using the adjacency matrix
        let result = self.cluster_on_cpu(&adjacency, n_points);

        Ok(result)
    }

    /// Compute adjacency matrix on GPU.
    /// Returns a flat vector of u32s representing the bit-packed adjacency matrix.
    async fn compute_adjacency_matrix_gpu(
        &self,
        data: &[f32],
        n_points: usize,
        n_features: usize,
    ) -> Result<Vec<u32>, GpuError> {
        // Load shader
        let shader_source = include_str!("kernels/adjacency.wgsl");
        let shader = self.context.create_shader_module("Adjacency Shader", shader_source)?;

        // Calculate dimensions
        let n_blocks = (n_points + 31) / 32;
        let adjacency_size = n_points * n_blocks; // Total u32s needed

        // Create buffers
        let data_buffer = self.context.create_buffer_from_data(
            "Data Buffer",
            data,
            wgpu::BufferUsages::STORAGE,
        );

        let adjacency_buffer = self.context.create_buffer(
            "Adjacency Buffer",
            (adjacency_size * std::mem::size_of::<u32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_points: u32,
            n_features: u32,
            eps: f32,
        }

        let params = Params {
            n_points: n_points as u32,
            n_features: n_features as u32,
            eps: self.params.eps as f32,
        };

        let params_buffer = self.context.create_buffer_from_data(
            "Params Buffer",
            &[params],
            wgpu::BufferUsages::UNIFORM,
        );

        // Create bind group layout
        let bind_group_layout = self.context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Adjacency Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group
        let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Adjacency Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: adjacency_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout = self.context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Adjacency Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Adjacency Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        // Execute compute pass
        let mut encoder = self.context.create_compute_encoder();
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Adjacency Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch: total_threads = n_points * n_blocks
            // Workgroup size is 64
            let total_threads = adjacency_size as u32;
            let workgroups = workgroup_count(total_threads, 64);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.context.submit_encoder(encoder);

        // Read back results
        let staging_buffer = self.context.create_buffer(
            "Staging Buffer",
            (adjacency_size * std::mem::size_of::<u32>()) as u64,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &adjacency_buffer,
            0,
            &staging_buffer,
            0,
            (adjacency_size * std::mem::size_of::<u32>()) as u64,
        );

        self.context.queue.submit(Some(encoder.finish()));

        let adjacency = self.context.read_buffer::<u32>(&staging_buffer, adjacency_size).await?;

        Ok(adjacency)
    }

    /// Perform clustering on CPU using adjacency matrix from GPU.
    fn cluster_on_cpu(
        &self,
        adjacency: &[u32],
        n_points: usize,
    ) -> ClusterResult {
        let n_blocks = (n_points + 31) / 32;
        let mut labels = Array1::from_elem(n_points, -2i32);
        let mut core_points = Array1::from_elem(n_points, false);

        // Identify core points
        for i in 0..n_points {
            let mut neighbor_count = 0;
            let row_start = i * n_blocks;
            
            for b in 0..n_blocks {
                neighbor_count += adjacency[row_start + b].count_ones() as usize;
            }
            
            // Self is included in adjacency (distance 0 <= eps), so we might want to subtract 1
            // But usually min_samples includes the point itself in DBSCAN definitions.
            // Let's assume standard DBSCAN where point itself counts.
            
            if neighbor_count >= self.params.min_samples {
                core_points[i] = true;
            }
        }

        // Cluster expansion
        let mut cluster_id = 0i32;
        for i in 0..n_points {
            if labels[i] != -2 {
                continue;
            }

            if !core_points[i] {
                labels[i] = -1; // Noise
                continue;
            }

            // Start new cluster
            self.expand_cluster_cpu(
                i,
                cluster_id,
                adjacency,
                &mut labels,
                &core_points,
                n_points,
                n_blocks,
            );
            cluster_id += 1;
        }

        ClusterResult { labels, core_points }
    }

    /// Expand cluster using BFS on CPU.
    fn expand_cluster_cpu(
        &self,
        start: usize,
        cluster_id: i32,
        adjacency: &[u32],
        labels: &mut Array1<i32>,
        core_points: &Array1<bool>,
        n_points: usize,
        n_blocks: usize,
    ) {
        let mut queue = vec![start];
        labels[start] = cluster_id;

        while let Some(point) = queue.pop() {
            if !core_points[point] {
                continue;
            }

            let row_start = point * n_blocks;

            // Iterate over neighbors using bitmask
            for b in 0..n_blocks {
                let mut word = adjacency[row_start + b];
                if word == 0 {
                    continue;
                }

                let base_idx = b * 32;
                
                // Iterate over set bits
                while word != 0 {
                    let trailing_zeros = word.trailing_zeros();
                    let bit_idx = trailing_zeros;
                    let neighbor = base_idx + bit_idx as usize;
                    
                    // Clear the bit to continue
                    word &= !(1 << bit_idx);

                    if neighbor >= n_points {
                        continue;
                    }

                    if labels[neighbor] == -2 {
                        labels[neighbor] = cluster_id;
                        queue.push(neighbor);
                    } else if labels[neighbor] == -1 {
                        labels[neighbor] = cluster_id;
                    }
                }
            }
        }
    }

    /// Fallback to CPU implementation for small datasets.
    fn fit_cpu(&self, data: &ArrayView2<'_, f64>) -> Result<ClusterResult, GpuError> {
        // Use the CPU implementation from ghdbscan-core
        let params = ghdbscan_core::DBSCANParams::new(self.params.eps, self.params.min_samples)
            .with_metric(self.params.metric.clone());
        let dbscan = ghdbscan_core::DBSCAN::new(params);
        let result = dbscan.fit(data);

        Ok(ClusterResult {
            labels: result.labels,
            core_points: result.core_points,
        })
    }
}
