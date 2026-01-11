// Genome utilities for reading base pairs

/// Exception thrown when trying to read from an exhausted RNA sequence
#[derive(Debug)]
pub struct RnaExhaustedException;

impl std::fmt::Display for RnaExhaustedException {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "RNA sequence exhausted")
    }
}

impl std::error::Error for RnaExhaustedException {}

use std::collections::VecDeque;

/// Reads the first base from the RNA sequence and removes it
/// Returns the base as an integer (0-3)
pub fn read_base(rna: &mut VecDeque<u8>) -> Result<u8, RnaExhaustedException> {
    rna.pop_front().ok_or(RnaExhaustedException)
}

/// Reads a range of bases linearly, adding up each base's contribution
/// Returns a normalized value [0, 1]
pub fn read_base_range(rna: &mut VecDeque<u8>, length: usize) -> Result<f32, RnaExhaustedException> {
    let mut result = 0.0f32;
    for _ in 0..length {
        let base = read_base(rna)? as f32;
        result += base;
    }
    Ok(result / (3.0 * length as f32))
}

/// Reads a range of bases to create a unique number
/// Uses formula: 0.25 * first + 0.25^2 * second + 0.25^3 * third + ...
/// Returns a uniformly distributed value [0, 1]
pub fn read_unique_base_range(rna: &mut VecDeque<u8>, length: usize) -> Result<f32, RnaExhaustedException> {
    let mut result = 0.0f32;
    for i in 0..length {
        let base = read_base(rna)? as f32;
        result += base * 0.25f32.powi((i + 1) as i32);
    }
    Ok(result)
}

/// Reads a range of bases exponentially
/// Creates exponential tail ends of the distribution
pub fn read_exp_base_range(rna: &mut VecDeque<u8>, length: usize) -> Result<f32, RnaExhaustedException> {
    let result = read_base_range(rna, length)?;
    Ok((1.45 * result - 0.6).powi(3) + result / 5.0 + 0.25)
}

