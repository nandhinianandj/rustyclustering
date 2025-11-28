//! Utility functions for GPU operations.

use ndarray::ArrayView2;

/// Calculate the number of workgroups needed for a given size.
pub fn workgroup_count(size: u32, workgroup_size: u32) -> u32 {
    (size + workgroup_size - 1) / workgroup_size
}

/// Convert 2D array to flat f32 vector for GPU transfer.
pub fn array_to_flat_f32(data: &ArrayView2<f64>) -> Vec<f32> {
    data.iter().map(|&x| x as f32).collect()
}

/// Get the size in bytes for a buffer.
pub fn buffer_size<T>(count: usize) -> u64 {
    (count * std::mem::size_of::<T>()) as u64
}

/// Pad size to alignment requirements.
pub fn align_to(size: u64, alignment: u64) -> u64 {
    ((size + alignment - 1) / alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workgroup_count() {
        assert_eq!(workgroup_count(100, 64), 2);
        assert_eq!(workgroup_count(64, 64), 1);
        assert_eq!(workgroup_count(65, 64), 2);
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(100, 256), 256);
        assert_eq!(align_to(256, 256), 256);
        assert_eq!(align_to(257, 256), 512);
    }
}
