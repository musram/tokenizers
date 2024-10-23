use env_logger::{self, Builder};
use lazy_static::lazy_static;
use log::{debug, error, info, LevelFilter};
use std::collections::HashMap;
use std::convert::TryInto;
use std::fs::{self, File};
use std::io::Seek;
use std::io::{self, Read, Write};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::Arc;
use std::sync::Once;
use tempfile::NamedTempFile;

const HDR_MAGIC: &[u8] = b"MMIDIDX\x00\x00";
const VERSION: u64 = 1;
const DTYPE_CODE: u8 = 4; // Assuming 4 represents the dtype code for np.int32
const HDR_SIZE: u64 = 9 + 8 + 1 + 8 + 8; // Header size

lazy_static! {
    static ref DTYPE_MAP: HashMap<&'static str, u8> = {
        let mut m = HashMap::new();
        m.insert("np.uint8", 1);
        m.insert("np.int8", 2);
        m.insert("np.int16", 3);
        m.insert("np.int32", 4);
        m.insert("np.int64", 5);
        m.insert("np.float32", 6);
        m.insert("np.float64", 7);
        m.insert("np.uint16", 8);
        m
    };
}

fn get_dtype_code(dtype_str: &str) -> Option<u8> {
    DTYPE_MAP.get(dtype_str).cloned()
}

static INIT: Once = Once::new();

fn init_logger() {
    INIT.call_once(|| {
        let log_file = File::create("merges_log.txt").unwrap();
        Builder::new()
            .target(env_logger::Target::Pipe(Box::new(log_file)))
            .filter_level(LevelFilter::Info)
            .init();
    });
}

fn read_contents(path: &str, dtype_str: &str) -> io::Result<(Vec<u32>, Vec<u64>, Vec<u64>)> {
    let mut file = File::open(&path)?;

    let mut index_buffer = Vec::new();
    file.read_to_end(&mut index_buffer)?;

    let hdr_magic = &index_buffer[0..HDR_MAGIC.len()];
    assert_eq!(hdr_magic, HDR_MAGIC, "Invalid header magic");

    let version = u64::from_le_bytes(
        index_buffer[HDR_MAGIC.len()..HDR_MAGIC.len() + 8]
            .try_into()
            .unwrap(),
    );
    assert_eq!(version, VERSION, "Unsupported version");

    let dtype_code = index_buffer[HDR_MAGIC.len() + 8];
    assert_eq!(
        dtype_code,
        get_dtype_code(dtype_str).unwrap(),
        "Unsupported dtype code"
    );

    let sizes_count = u64::from_le_bytes(
        index_buffer[HDR_MAGIC.len() + 9..HDR_MAGIC.len() + 17]
            .try_into()
            .unwrap(),
    );

    info!("Sizes count: {}", sizes_count);
    let doc_indices_count = u64::from_le_bytes(
        index_buffer[HDR_MAGIC.len() + 17..HDR_MAGIC.len() + 25]
            .try_into()
            .unwrap(),
    );

    info!("Doc indices count: {}", doc_indices_count);

    let sizes_start: u64 = HDR_SIZE;

    let sizes = index_buffer[(sizes_start as usize)..(sizes_start + sizes_count * 4) as usize]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<u32>>();

    //info!("Read sizes {:?}", sizes);
    assert_eq!(sizes.len(), sizes_count as usize, "Invalid sizes count");

    let pointers_start: u64 = sizes_start + sizes_count * 4;

    let pointers = index_buffer
        [(pointers_start as usize)..(pointers_start + sizes_count * 8) as usize]
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<u64>>();

    //info!("Read pointers {:?}", pointers);
    assert_eq!(
        pointers.len(),
        sizes_count as usize,
        "Invalid pointers count"
    );

    let doc_idx_start: u64 = pointers_start + (pointers.len() as u64 * 8);

    let doc_idx = index_buffer
        [(doc_idx_start as usize)..(doc_idx_start + doc_indices_count * 8) as usize]
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<u64>>();

    //info!("Read doc_idx {:?}", doc_idx);

    assert_eq!(
        doc_idx.len(),
        doc_indices_count as usize,
        "Invalid doc_idx count"
    );

    Ok((sizes, pointers, doc_idx))
}
struct MMapIndexedDatasetBuilder {
    sizes: Vec<u32>,
    pointers: Vec<u64>,
    doc_idx: Vec<u64>,
    data_file: NamedTempFile,
    dtype_code: u8,
}

impl MMapIndexedDatasetBuilder {
    fn new(dtype: &str) -> io::Result<Self> {
        let dtype_code = get_dtype_code(dtype).unwrap();
        let data_file = NamedTempFile::new()?;
        Ok(MMapIndexedDatasetBuilder {
            sizes: Vec::new(),
            pointers: Vec::new(),
            doc_idx: Vec::new(),
            data_file,
            dtype_code,
        })
    }

    fn merge_file(&mut self, another_file_path: &str) -> io::Result<()> {
        info!("Merging file: {}", another_file_path);

        let (sizes, pointers, doc_idx) = self.read_index(another_file_path)?;

        println!(
            "Read index: {} sizes, {} pointers, {} doc_idx for {}",
            sizes.len(),
            pointers.len(),
            doc_idx.len(),
            another_file_path
        );
        info!(
            "Read index: {} sizes, {} pointers, {} doc_idx",
            sizes.len(),
            pointers.len(),
            doc_idx.len()
        );

        let last_pointer = self.pointers.last().cloned().unwrap_or(0);
        let last_size = self.sizes.last().cloned().unwrap_or(0) as u64;

        self.sizes.extend(sizes);

        let total_size = last_pointer + last_size;
        let adjusted_pointers: Vec<u64> = pointers.iter().map(|&ptr| ptr + total_size).collect();
        self.pointers.extend(adjusted_pointers);

        let doc_count = self.doc_idx.len() as u64;

        let adjusted_doc_idx: Vec<u64> = doc_idx.iter().map(|&idx| idx + doc_count).collect();
        self.doc_idx.extend(adjusted_doc_idx.into_iter());

        // Read the data file
        let data = self.read_data(another_file_path)?;

        // Append the data to self.data_file
        self.data_file.write_all(&data)?;

        // // Write the combined data to the output path
        // let output_data_path = format!("{}.bin", output_path);
        // fs::rename(self.data_file.path(), &output_data_path)?;

        info!("Finished merging file: {}", another_file_path);
        Ok(())
    }

    fn read_index(&self, path: &str) -> io::Result<(Vec<u32>, Vec<u64>, Vec<u64>)> {
        let index_path = format!("{}.idx", path);
        let mut file = File::open(&index_path)?;

        let mut index_buffer = Vec::new();
        file.read_to_end(&mut index_buffer)?;

        let hdr_magic = &index_buffer[0..HDR_MAGIC.len()];
        assert_eq!(hdr_magic, HDR_MAGIC, "Invalid header magic");

        let version = u64::from_le_bytes(
            index_buffer[HDR_MAGIC.len()..HDR_MAGIC.len() + 8]
                .try_into()
                .unwrap(),
        );
        assert_eq!(version, VERSION, "Unsupported version");

        let dtype_code = index_buffer[HDR_MAGIC.len() + 8];
        assert_eq!(self.dtype_code, dtype_code, "Unsupported dtype code");

        let sizes_count = u64::from_le_bytes(
            index_buffer[HDR_MAGIC.len() + 9..HDR_MAGIC.len() + 17]
                .try_into()
                .unwrap(),
        );

        info!("Sizes count: {}", sizes_count);
        let doc_indices_count = u64::from_le_bytes(
            index_buffer[HDR_MAGIC.len() + 17..HDR_MAGIC.len() + 25]
                .try_into()
                .unwrap(),
        );

        info!("Doc indices count: {}", doc_indices_count);

        let sizes_start: u64 = HDR_SIZE;

        let sizes = index_buffer[(sizes_start as usize)..(sizes_start + sizes_count * 4) as usize]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<u32>>();

        //info!("Read sizes {:?}", sizes);
        assert_eq!(sizes.len(), sizes_count as usize, "Invalid sizes count");

        let pointers_start: u64 = sizes_start + sizes_count * 4;

        let pointers = index_buffer
            [(pointers_start as usize)..(pointers_start + sizes_count * 8) as usize]
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<u64>>();

        //info!("Read pointers {:?}", pointers);
        assert_eq!(
            pointers.len(),
            sizes_count as usize,
            "Invalid pointers count"
        );

        let doc_idx_start: u64 = pointers_start + (pointers.len() as u64 * 8);

        let doc_idx = index_buffer
            [(doc_idx_start as usize)..(doc_idx_start + doc_indices_count * 8) as usize]
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<u64>>();

        //info!("Read doc_idx {:?}", doc_idx);

        assert_eq!(
            doc_idx.len(),
            doc_indices_count as usize,
            "Invalid doc_idx count"
        );

        Ok((sizes, pointers, doc_idx))
    }

    fn read_data(&self, path: &str) -> io::Result<Vec<u8>> {
        let data_path = format!("{}.bin", path);
        let mut data_file = File::open(&data_path)?;
        let mut data_buffer = Vec::new();
        data_file.read_to_end(&mut data_buffer)?;
        Ok(data_buffer)
    }

    fn finalize(&self, file_name: &str) -> io::Result<()> {
        info!("Finalizing and writing output index file");

        // Create and write the index file
        let mut index_file = File::create(format!("{}.idx", file_name))?;
        index_file.write_all(HDR_MAGIC)?;
        index_file.write_all(&VERSION.to_le_bytes())?;
        index_file.write_all(&[self.dtype_code])?;
        index_file.write_all(&(self.sizes.len() as u64).to_le_bytes())?;
        index_file.write_all(&(self.doc_idx.len() as u64).to_le_bytes())?;

        info!("Writing sizes");
        for &size in &self.sizes {
            index_file.write_all(&(size).to_le_bytes())?;
        }

        info!("Writing pointers");
        for &pointer in &self.pointers {
            index_file.write_all(&pointer.to_le_bytes())?;
        }

        info!("Writing doc_idx");
        for &doc in &self.doc_idx {
            index_file.write_all(&doc.to_le_bytes())?;
        }

        info!("Closing the index file");
        index_file.sync_data()?;

        info!("Writing data file");
        let data_file_path = format!("{}.bin", file_name);
        let mut output_file = File::create(&data_file_path)?;
        let mut input_file = File::open(self.data_file.path())?;
        io::copy(&mut input_file, &mut output_file)?;

        // make sure the index and data files are created
        assert!(Path::new(&format!("{}.idx", file_name)).exists());
        assert!(Path::new(&format!("{}.bin", file_name)).exists());

        info!("Merge process completed successfully.");
        Ok(())
    }
}

fn main() -> io::Result<()> {
    init_logger();

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        error!(
            "Usage: {} <output_dir> <dtype> <input_dir> <output_file_name>",
            args[0]
        );
        error!("Received {} arguments: {:?}", args.len() - 1, &args[1..]);
        std::process::exit(1);
    }

    let output_dir = &args[1];
    let dtype = &args[2];
    let input_dir = &args[3];
    let output_file_name = &args[4];

    info!("Starting merge process");
    info!("Output directory: {}", output_dir);
    info!("Data type: {}", dtype);
    info!("Input directory: {}", input_dir);
    info!("Output file name: {}", output_file_name);

    // Validate dtype
    if !DTYPE_MAP.contains_key(dtype.as_str()) {
        error!(
            "Invalid dtype: {}. Supported dtypes are: {:?}",
            dtype,
            DTYPE_MAP.keys()
        );
        std::process::exit(1);
    }

    // check if input directory exists
    if !Path::new(input_dir).is_dir() {
        error!("Input directory does not exist: {}", input_dir);
        std::process::exit(1);
    }

    // Check if output directory exists
    if !Path::new(output_dir).is_dir() {
        error!("Output directory does not exist: {}", output_dir);
        std::process::exit(1);
    }

    let mut builder = MMapIndexedDatasetBuilder::new(dtype).map_err(|e| {
        error!("Failed to create MMapIndexedDatasetBuilder: {}", e);
        e
    })?;

    info!("Created MMapIndexedDatasetBuilder");

    let input_path = Path::new(input_dir);
    if !input_path.is_dir() {
        error!(
            "Input directory does not exist or is not a directory: {}",
            input_dir
        );
        std::process::exit(1);
    }

    // Iterate over files in the input directory
    let mut files_processed = 0;
    for entry in fs::read_dir(input_path).map_err(|e| {
        error!("Failed to read input directory: {}", e);
        e
    })? {
        let entry = entry.map_err(|e| {
            error!("Failed to read directory entry: {}", e);
            e
        })?;
        let path = entry.path();
        if path.is_file() && path.extension().map_or(false, |ext| ext == "idx") {
            let file_stem = path
                .file_stem()
                .ok_or_else(|| {
                    let e = io::Error::new(io::ErrorKind::Other, "Invalid file name");
                    error!("Invalid file name: {:?}", path);
                    e
                })?
                .to_str()
                .ok_or_else(|| {
                    let e = io::Error::new(io::ErrorKind::Other, "Non-UTF8 file name");
                    error!("Non-UTF8 file name: {:?}", path);
                    e
                })?;
            info!("Processing file: {}", file_stem);

            // Merge each file
            match builder.merge_file(&format!("{}/{}", input_dir, file_stem)) {
                Ok(_) => {
                    info!("Successfully merged file: {}", file_stem);
                    files_processed += 1;
                }
                Err(e) => {
                    error!("Error merging file {}: {}", file_stem, e);
                }
            }
        }
    }

    if files_processed == 0 {
        error!(
            "No files were processed in the input directory: {}",
            input_dir
        );
        std::process::exit(1);
    }

    let output_path = Path::new(output_dir).join(output_file_name);
    // Finalize and write the output index file
    info!("Finalizing and writing output index file and data file");
    builder.finalize(output_path.to_str().unwrap()).map_err(|e| {
        error!("Failed to finalize and write output index file: {}", e);
        e
    })?;

    info!(
        "Merge process completed successfully. Processed {} files.",
        files_processed
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::{self, Write};
    use tempfile::tempdir;

    fn create_test_dataset(
        path: &str,
        sizes: &[u32],
        pointers: &[u64],
        doc_idx: &[u64],
        data: &[u8],
    ) -> io::Result<()> {
        let idx_path = format!("{}.idx", path);
        let data_path = format!("{}.bin", path);

        let mut index_file = File::create(&idx_path)?;
        index_file.write_all(HDR_MAGIC)?;
        index_file.write_all(&VERSION.to_le_bytes())?;
        index_file.write_all(&[DTYPE_CODE])?;
        index_file.write_all(&(sizes.len() as u64).to_le_bytes())?;
        index_file.write_all(&(doc_idx.len() as u64).to_le_bytes())?;

        for &size in sizes {
            index_file.write_all(&size.to_le_bytes())?;
        }
        for &pointer in pointers {
            index_file.write_all(&pointer.to_le_bytes())?;
        }
        for &doc in doc_idx {
            index_file.write_all(&doc.to_le_bytes())?;
        }

        let mut data_file = File::create(&data_path)?;
        data_file.write_all(data)?;

        // check the files are created
        assert!(Path::new(&idx_path).exists());
        assert!(Path::new(&data_path).exists());

        // read the index file
        let dtype_str = "np.int32";
        let (sizes, pointers, doc_idx) = read_contents(&idx_path, dtype_str)?;
        assert_eq!(sizes, sizes);
        assert_eq!(pointers, pointers);
        assert_eq!(doc_idx, doc_idx);

        Ok(())
    }

    #[test]
    fn test_merge_single_file() -> io::Result<()> {
        let dir = tempdir()?;
        let dataset_a_path = dir.path().join("dataset_a");
        let dataset_index_path = dataset_a_path.join("dataset_a.idx");
        let dataset_data_path = dataset_a_path.join("dataset_a.bin");
        create_test_dataset(
            dataset_a_path.to_str().unwrap(),
            &[3, 4],
            &[0, 3],
            &[0, 1],
            &[1, 2, 3, 4, 5, 6, 7],
        )?;

        let output_path = dir.path().join("output");
        let mut builder = MMapIndexedDatasetBuilder::new("np.int32")?;

        // Merge single file
        builder.merge_file(dataset_a_path.to_str().unwrap())?;

        // Finalize output and then create a new builder for reading
        builder.finalize(output_path.to_str().unwrap())?;

        // Read index from the finalized dataset
        // Create a new MMapIndexedDatasetBuilder instance for reading
        let reader = MMapIndexedDatasetBuilder::new("np.int32")?;

        // Read index from the finalized dataset
        let (sizes, pointers, doc_idx) = reader.read_index(output_path.to_str().unwrap())?;

        assert_eq!(sizes, vec![3, 4]); // Sizes
        assert_eq!(pointers, vec![0, 3]); // Pointers
        assert_eq!(doc_idx, vec![0, 1]); // Document indices

        let merged_data_path = format!("{}.bin", output_path.to_str().unwrap());
        println!("merged_data_path: {}", merged_data_path);

        let merged_data = std::fs::read(merged_data_path)?;
        println!("merged_data: {:?}", merged_data);
        assert_eq!(merged_data, vec![1, 2, 3, 4, 5, 6, 7]); // Data

        Ok(())
    }

    #[test]
    fn test_merge_datasets() -> io::Result<()> {
        let dir = tempdir()?;
        let dataset_a_path = dir.path().join("dataset_a");
        let dataset_b_path = dir.path().join("dataset_b");
        let dataset_c_path = dir.path().join("dataset_c");

        // Create Dataset A
        create_test_dataset(
            dataset_a_path.to_str().unwrap(),
            &[3, 4],
            &[0, 3],
            &[0, 2],
            &[1, 2, 3, 4, 5, 6, 7],
        )?;

        // Create Dataset B
        create_test_dataset(
            dataset_b_path.to_str().unwrap(),
            &[2, 5],
            &[0, 2],
            &[0, 2],
            &[8, 9, 10, 11, 12, 13, 14],
        )?;

        // Create Dataset C
        create_test_dataset(
            dataset_c_path.to_str().unwrap(),
            &[1, 3],
            &[0, 1],
            &[0, 1],
            &[15, 16, 17, 18],
        )?;

        let output_path = dir.path().join("output");
        let mut builder = MMapIndexedDatasetBuilder::new("np.int32")?;

        // Merge Dataset A, B, and C
        builder.merge_file(dataset_a_path.to_str().unwrap())?;
        builder.merge_file(dataset_b_path.to_str().unwrap())?;
        builder.merge_file(dataset_c_path.to_str().unwrap())?;

        // Finalize output and then create a new builder for reading
        builder.finalize(output_path.to_str().unwrap())?;

        // Read index from the finalized dataset
        // Create a new MMapIndexedDatasetBuilder instance for reading
        let reader = MMapIndexedDatasetBuilder::new("np.int32")?;

        // Read index from the finalized dataset
        let (sizes, pointers, doc_idx) = reader.read_index(output_path.to_str().unwrap())?;

        assert_eq!(sizes, vec![3, 4, 2, 5, 1, 3]); // Sizes
        assert_eq!(pointers, vec![0, 3, 7, 9, 14, 15]); // Pointers
        assert_eq!(doc_idx, vec![0, 2, 2, 4, 4, 5]); // Document indices

        let merged_data = std::fs::read(format!("{}.bin", output_path.to_str().unwrap()))?;
        assert_eq!(
            merged_data,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        ); // Data

        Ok(())
    }

    #[test]
    fn test_merge_datasets_error_handling() {
        let dir = tempdir().unwrap();
        let output_path = dir.path().join("output");

        let mut builder = MMapIndexedDatasetBuilder::new("np.int32").unwrap();

        // Attempt to merge a non-existent dataset
        let result = builder.merge_file("non_existent.idx");
        assert!(result.is_err());
    }
    #[test]
    fn test_merge_empty_dataset() -> io::Result<()> {
        let dir = tempdir()?;
        let dataset_a_path = dir.path().join("dataset_a");
        let dataset_b_path = dir.path().join("dataset_b");

        // Create an empty Dataset A
        create_test_dataset(dataset_a_path.to_str().unwrap(), &[], &[], &[], &[])?;

        // Create Dataset B
        create_test_dataset(
            dataset_b_path.to_str().unwrap(),
            &[2, 5],
            &[0, 2],
            &[0, 2],
            &[8, 9, 10, 11, 12, 13, 14],
        )?;

        let output_path = dir.path().join("output");
        let mut builder = MMapIndexedDatasetBuilder::new("np.int32")?;

        // Merge the empty dataset and the populated dataset
        builder.merge_file(dataset_a_path.to_str().unwrap())?;
        builder.merge_file(dataset_b_path.to_str().unwrap())?;

        // Finalize output
        builder.finalize(output_path.to_str().unwrap())?;

        // Read index from the finalized dataset
        let reader = MMapIndexedDatasetBuilder::new("np.int32")?;
        let (sizes, pointers, doc_idx) = reader.read_index(output_path.to_str().unwrap())?;

        assert_eq!(sizes, vec![2, 5]); // Sizes from Dataset B
        assert_eq!(pointers, vec![0, 2]); // Pointers from Dataset B
        assert_eq!(doc_idx, vec![0, 2]); // Document indices from Dataset B

        Ok(())
    }

    #[test]
    fn test_merge_datasets_with_duplicate_doc_ids() -> io::Result<()> {
        let dir = tempdir()?;
        let dataset_a_path = dir.path().join("dataset_a");
        let dataset_b_path = dir.path().join("dataset_b");

        // Create Dataset A
        create_test_dataset(
            dataset_a_path.to_str().unwrap(),
            &[3, 4],
            &[0, 3],
            &[0, 1],
            &[1, 2, 3, 4, 5, 6, 7],
        )?;

        // Create Dataset B with duplicate document indices
        create_test_dataset(
            dataset_b_path.to_str().unwrap(),
            &[2, 5],
            &[0, 2],
            &[1, 2],
            &[8, 9, 10, 11, 12, 13, 14],
        )?;

        let output_path = dir.path().join("output");
        let mut builder = MMapIndexedDatasetBuilder::new("np.int32")?;

        // Merge Dataset A and B
        builder.merge_file(dataset_a_path.to_str().unwrap())?;
        builder.merge_file(dataset_b_path.to_str().unwrap())?;

        // Finalize output
        builder.finalize(output_path.to_str().unwrap())?;

        // Read index from the finalized dataset
        let reader = MMapIndexedDatasetBuilder::new("np.int32")?;
        let (sizes, pointers, doc_idx) = reader.read_index(output_path.to_str().unwrap())?;

        assert_eq!(sizes, vec![3, 4, 2, 5]); // Sizes
        assert_eq!(pointers, vec![0, 3, 7, 9]); // Pointers
        assert_eq!(doc_idx, vec![0, 1, 3, 4]); // Document indices should be adjusted

        Ok(())
    }

    #[test]
    fn test_merge_varied_size_datasets() -> io::Result<()> {
        let dir = tempdir()?;
        let dataset_a_path = dir.path().join("dataset_a");
        let dataset_b_path = dir.path().join("dataset_b");

        // Create Dataset A with a small size
        create_test_dataset(dataset_a_path.to_str().unwrap(), &[1], &[0], &[0], &[1])?;

        // Create Dataset B with a larger size
        create_test_dataset(
            dataset_b_path.to_str().unwrap(),
            &[5, 10],
            &[0, 5],
            &[0, 5],
            &[2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        )?;

        let output_path = dir.path().join("output");
        let mut builder = MMapIndexedDatasetBuilder::new("np.int32")?;

        // Merge both datasets
        builder.merge_file(dataset_a_path.to_str().unwrap())?;
        builder.merge_file(dataset_b_path.to_str().unwrap())?;

        // Finalize output
        builder.finalize(output_path.to_str().unwrap())?;

        // Read index from the finalized dataset
        let reader = MMapIndexedDatasetBuilder::new("np.int32")?;
        let (sizes, pointers, doc_idx) = reader.read_index(output_path.to_str().unwrap())?;

        assert_eq!(sizes, vec![1, 5, 10]); // Sizes
        assert_eq!(pointers, vec![0, 1, 6]); // Pointers
        assert_eq!(doc_idx, vec![0, 1, 6]); // Document indices should be adjusted

        Ok(())
    }
    #[test]
    fn test_merge_large_datasets() -> io::Result<()> {
        let dir = tempdir()?;
        let dataset_a_path = dir.path().join("dataset_a");
        let dataset_b_path = dir.path().join("dataset_b");

        // Create a large Dataset A
        create_test_dataset(
            dataset_a_path.to_str().unwrap(),
            &[1000],
            &[0],
            &[0],
            &vec![1; 1000], // 1000 elements of value 1
        )?;

        // Create a large Dataset B
        create_test_dataset(
            dataset_b_path.to_str().unwrap(),
            &[2000],
            &[0],
            &[0],
            &vec![2; 2000], // 2000 elements of value 2
        )?;

        let output_path = dir.path().join("output");
        let mut builder = MMapIndexedDatasetBuilder::new("np.int32")?;

        // Merge both large datasets
        builder.merge_file(dataset_a_path.to_str().unwrap())?;
        builder.merge_file(dataset_b_path.to_str().unwrap())?;

        // Finalize output
        builder.finalize(output_path.to_str().unwrap())?;

        // Read index from the finalized dataset
        let reader = MMapIndexedDatasetBuilder::new("np.int32")?;
        let (sizes, pointers, doc_idx) = reader.read_index(output_path.to_str().unwrap())?;

        assert_eq!(sizes, vec![1000, 2000]); // Sizes
        assert_eq!(pointers, vec![0, 1000]); // Pointers
        assert_eq!(doc_idx, vec![0, 1]); // Document indices

        Ok(())
    }
}
