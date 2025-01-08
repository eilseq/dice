pub mod binary_matrix {
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

    /// Converts a flat COO format (list of coordinates) back to a flattened matrix.
    pub fn coo_to_flat(
        coo_flat: Vec<usize>,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<usize>, String> {
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
}
