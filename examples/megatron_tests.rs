


fn exscan_from_cumsum(arr: &mut [u64]) {
    if arr.len() > 1 {
        arr.copy_within(0..arr.len() - 1, 1);
    }
    if !arr.is_empty() {
        arr[0] = 0;
    }
}

fn get_pointers_with_total(sizes: &Vec<u32>, dtype_bytes: u8) -> (Vec<u64>, u64) {
    let mut pointers: Vec<u64> = Vec::with_capacity(sizes.len());
    let mut cumulative_sum: u64 = 0;

    // Calculate cumulative sizes in bytes
    for &size in sizes {
        cumulative_sum += size as u64 * dtype_bytes as u64;
        pointers.push(cumulative_sum);
    }

    let total_bytes = cumulative_sum;

    // Convert inclusive cumulative sums to exclusive
    exscan_from_cumsum(&mut pointers);

    (pointers, total_bytes as u64)
}

fn main() {
    let sizes = vec![2, 3, 5];
    let (pointers, total_bytes) = get_pointers_with_total(&sizes, 4);
    println!("Pointers: {:?}", pointers);
    println!("Total bytes: {}", total_bytes);
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;
    use std::io::{Read, Seek, SeekFrom};
    use tempfile::tempdir;

    #[test]
    fn test_get_pointers_with_total() {
        let sizes = vec![2, 3, 5];
        let (pointers, total_bytes) = get_pointers_with_total(&sizes, 4);
        assert_eq!(pointers, vec![0, 8, 20]);
        assert_eq!(total_bytes, 40);

        let sizes = vec![2048, 2048, 2048, 2048];
        let (pointers, total_bytes) = get_pointers_with_total(&sizes, 4);
        assert_eq!(pointers, vec![0, 8192, 16384, 24576]);
        assert_eq!(total_bytes, 32768);
    }
}