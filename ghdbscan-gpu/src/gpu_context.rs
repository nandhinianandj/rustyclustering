//! GPU context management for wgpu-based compute operations.

use std::sync::Arc;
use thiserror::Error;
use wgpu;

/// Errors that can occur during GPU operations.
#[derive(Error, Debug)]
pub enum GpuError {
    #[error("Failed to request GPU adapter")]
    AdapterRequestFailed,
    
    #[error("Failed to request GPU device: {0}")]
    DeviceRequestFailed(String),
    
    #[error("Failed to create shader module: {0}")]
    ShaderCreationFailed(String),
    
    #[error("Failed to create buffer")]
    BufferCreationFailed,
    
    #[error("GPU operation failed: {0}")]
    OperationFailed(String),
}

/// GPU context for managing device, queue, and compute operations.
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Create a new GPU context.
    ///
    /// This will request a GPU adapter and device. If no GPU is available,
    /// this will return an error.
    pub async fn new() -> Result<Self, GpuError> {
        // Create instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::AdapterRequestFailed)?;

        let adapter_info = adapter.get_info();
        
        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GHDBSCAN GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuError::DeviceRequestFailed(e.to_string()))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }

    /// Get information about the GPU adapter.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Create a buffer from data.
    pub fn create_buffer_from_data<T: bytemuck::Pod>(
        &self,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        })
    }

    /// Create an empty buffer.
    pub fn create_buffer(
        &self,
        label: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Read data from a buffer.
    pub async fn read_buffer<T: bytemuck::Pod>(
        &self,
        buffer: &wgpu::Buffer,
        size: usize,
    ) -> Result<Vec<T>, GpuError> {
        let buffer_slice = buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);
        
        receiver
            .receive()
            .await
            .ok_or(GpuError::OperationFailed("Failed to receive buffer mapping result".to_string()))?
            .map_err(|e| GpuError::OperationFailed(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        buffer.unmap();
        
        Ok(result)
    }

    /// Create a compute shader module from WGSL source.
    pub fn create_shader_module(&self, label: &str, source: &str) -> Result<wgpu::ShaderModule, GpuError> {
        Ok(self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        }))
    }

    /// Create a command encoder for compute operations.
    pub fn create_compute_encoder(&self) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        })
    }

    /// Submit a command encoder.
    pub fn submit_encoder(&self, encoder: wgpu::CommandEncoder) {
        self.queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_context_creation() {
        // This test requires a GPU to be available
        pollster::block_on(async {
            let result = GpuContext::new().await;
            if result.is_ok() {
                let ctx = result.unwrap();
                println!("GPU Adapter: {:?}", ctx.adapter_info());
            } else {
                println!("No GPU available, skipping test");
            }
        });
    }
}
