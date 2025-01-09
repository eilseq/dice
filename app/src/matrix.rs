use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// Representation Standards

pub fn flat_horizontal_flip<T: Clone>(
    matrix_flat: Vec<T>,
    rows: usize,
    cols: usize,
) -> Result<Vec<T>, String> {
    if matrix_flat.len() != rows * cols {
        return Err("invalid matrix values according to size".to_string());
    }

    let mut flipped = Vec::with_capacity(matrix_flat.len());

    for r in 0..rows {
        let row_start = r * cols;
        let row_end = row_start + cols;
        let row: Vec<T> = matrix_flat[row_start..row_end]
            .iter()
            .cloned()
            .rev()
            .collect();
        flipped.extend(row);
    }

    Ok(flipped)
}

pub fn flat_to_coo(
    matrix_flat: Vec<usize>,
    rows: usize,
    cols: usize,
) -> Result<Vec<usize>, String> {
    if (rows * cols) != matrix_flat.len() {
        return Err("invalid matrix values according to size".to_string());
    }

    let mut coo_flat = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            if matrix_flat[row * cols + col] == 1 {
                // 1 to inf std
                coo_flat.push(row + 1);
                coo_flat.push(col + 1);
            }
        }
    }

    Ok(coo_flat)
}

pub fn coo_to_flat(coo_flat: Vec<usize>, rows: usize, cols: usize) -> Result<Vec<usize>, String> {
    if (rows + cols) % 2 != 0 {
        return Err("Invalid COO coordinates: rows + cols must be even".to_string());
    }

    if coo_flat.len() % 2 != 0 {
        return Err("COO array must have an even number of elements".to_string());
    }

    let mut matrix_flat = vec![0; rows * cols];
    for i in (0..coo_flat.len()).step_by(2) {
        // 1 to inf std
        let row = coo_flat[i] - 1;
        let col = coo_flat[i + 1] - 1;
        matrix_flat[row * cols + col] = 1;
    }

    Ok(matrix_flat)
}

// Model Inference Tools
pub fn apply_noise(matrix_flat: Vec<usize>, noise_level: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    matrix_flat
        .iter()
        .map(|&x| x as f32)
        .map(|x| x + noise_level * (rng.gen::<f32>() * 2.0 - 1.0))
        .collect()
}

pub fn apply_threshold(matrix_flat: Vec<f32>, threshold: f32) -> Vec<usize> {
    matrix_flat
        .iter()
        .map(|&x| if x > threshold { 1 } else { 0 })
        .collect()
}
