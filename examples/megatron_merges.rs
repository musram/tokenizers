use env_logger::{self, Builder};
use lazy_static::lazy_static;
use log::{error, info, LevelFilter};
use std::collections::HashMap;
use std::convert::TryInto;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::sync::Arc;
use std::sync::Once;
use std::io::Seek;

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

struct MMapIndexedDatasetBuilder {
    sizes: Vec<u64>,
    pointers: Vec<u64>,
    doc_idx: Vec<u64>,
    data_file: File,
    dtype_code: u8, // Add this field
}

impl MMapIndexedDatasetBuilder {
    fn new(output_path: &str, dtype: &str) -> io::Result<Self> {
        let dtype_code = get_dtype_code(dtype).unwrap();
        let data_file = File::create(output_path)?;
        Ok(MMapIndexedDatasetBuilder {
            sizes: Vec::new(),
            pointers: Vec::new(),
            doc_idx: vec![0], // Start with an initial document index
            data_file,
            dtype_code, // Initialize the new field
        })
    }

    fn merge_file(&mut self, another_file_path: &str) -> io::Result<()> {
        info!("Merging file: {}", another_file_path);

        let (sizes, pointers, doc_idx) = self.read_index(another_file_path)?;

        info!(
            "Read index: {} sizes, {} pointers, {} doc_idx",
            sizes.len(),
            pointers.len(),
            doc_idx.len()
        );

        let offset = self.sizes.len() as u64;

        self.sizes.extend(sizes.into_iter().map(|s| s as u64));

        let adjusted_pointers: Vec<u64> = pointers.iter().map(|&ptr| ptr + offset).collect();
        self.pointers.extend(adjusted_pointers);

        let adjusted_doc_idx: Vec<u64> = doc_idx.iter().map(|&idx| idx + offset).collect();
        self.doc_idx.extend(adjusted_doc_idx.into_iter().skip(1)); // Skip the first element, which is always 0

        let another_data_path = format!("{}.bin", another_file_path);
        let mut another_data_file = File::open(&another_data_path)?;
        io::copy(&mut another_data_file, &mut self.data_file)?;

        let another_data_buffer = self.read_data(another_file_path)?;
        self.data_file.write_all(&another_data_buffer)?;

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

    fn finalize(mut self, file_name: &str) -> io::Result<()> {
        info!("Finalizing and writing output index file");

        let mut index_file = File::create(format!("{}.idx", file_name))?;

        index_file.write_all(HDR_MAGIC)?;
        index_file.write_all(&VERSION.to_le_bytes())?;
        index_file.write_all(&[DTYPE_CODE])?;
        index_file.write_all(&(self.sizes.len() as u64).to_le_bytes())?;
        index_file.write_all(&(self.doc_idx.len() as u64).to_le_bytes())?;

        info!("Writing sizes");
        for &size in &self.sizes {
            index_file.write_all(&size.to_le_bytes())?;
        }

        info!("Writing pointers");
        for &pointer in &self.pointers {
            index_file.write_all(&pointer.to_le_bytes())?;
        }

        info!("Writing doc_idx");

        for &doc in &self.doc_idx {
            index_file.write_all(&doc.to_le_bytes())?;
        }

        info!("closing the index file");
        index_file.sync_data()?;
        drop(index_file);


        info!("Writing data");

        let mut data_file = File::create(format!("{}.bin", file_name))?;
        self.data_file.seek(std::io::SeekFrom::Start(0))?;
        let mut buffer = Vec::new();
        match self.data_file.read_to_end(&mut buffer) {
            Ok(_) => {
                match data_file.write_all(&buffer) {
                    Ok(_) => {
                        info!("Data file copied successfully");
                    },
                    Err(e) => {
                        error!("Failed to write data file: {}", e);
                        return Err(e);
                    }
                }
            },
            Err(e) => {
                error!("Failed to read data file: {}", e);
                return Err(e);
            }
        }

        info!("closing the data file");
        data_file.sync_all()?;
        data_file.flush()?;
        drop(data_file);
        self.data_file.flush()?;
        self.data_file.sync_all()?;

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

    // Check if output directory exists
    if !Path::new(output_dir).is_dir() {
        error!("Output directory does not exist: {}", output_dir);
        std::process::exit(1);
    }

    let output_path = format!("{}/{}", output_dir, output_file_name);
    let mut builder = MMapIndexedDatasetBuilder::new(&output_path, dtype).map_err(|e| {
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

    // Finalize and write the output index file
    info!("Finalizing and writing output index file");
    builder.finalize(&output_path).map_err(|e| {
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

    fn create_test_dataset(path: &str, sizes: &[u32], pointers: &[u64], doc_idx: &[u64], data: &[u8]) -> io::Result<()> {
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

        Ok(())
    }

    #[test]
    fn test_merge_datasets() -> io::Result<()> {
        let dir = tempdir()?;
        let dataset_a_path = dir.path().join("dataset_a");
        let dataset_b_path = dir.path().join("dataset_b");
        
        // Create Dataset A
        create_test_dataset(
            dataset_a_path.to_str().unwrap(),
            &[3, 4],
            &[0, 3, 7], // Pointers
            &[0, 2], // Document indices
            &[1, 2, 3, 4, 5, 6, 7, 8],
        )?;
        
        // Create Dataset B
        create_test_dataset(
            dataset_b_path.to_str().unwrap(),
            &[2, 5],
            &[0, 2, 7], // Pointers
            &[0, 2], // Document indices
            &[9, 10, 11, 12, 13, 14, 15],
        )?;

        let output_path = dir.path().join("output");
        let mut builder = MMapIndexedDatasetBuilder::new(output_path.to_str().unwrap(), "np.int32")?;

        // Merge Dataset B into Dataset A
        builder.merge_file(dataset_a_path.to_str().unwrap())?;
        builder.merge_file(dataset_b_path.to_str().unwrap())?;
        
        // Finalize output
        builder.finalize(output_path.to_str().unwrap())?;

        // Verify output
        let merged_index = builder.read_index(output_path.to_str().unwrap())?;
        
        assert_eq!(merged_index.0, vec![3, 4, 2, 5]); // Sizes
        assert_eq!(merged_index.1, vec![0, 3, 7, 8, 10]); // Pointers
        assert_eq!(merged_index.2, vec![0, 2, 2, 4]); // Document indices

        let merged_data = builder.read_data(output_path.to_str().unwrap())?;
        assert_eq!(merged_data, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]); // Data

        Ok(())
    }
}

