use bytemuck;
use byteorder::{LittleEndian, WriteBytesExt};
use env_logger::Builder;
use indicatif::{ProgressBar, ProgressStyle};
use lazy_static::lazy_static;
use log::{error, info, warn, LevelFilter};
use std::collections::{HashMap};
use std::io::{self, Write};


//const HDR_MAGIC: &[u8] = b"PACKED"; // 6 bytes
const HDR_MAGIC: &[u8] = b"MMIDIDX\x00\x00";
const VERSION: u64 = 1;
const HDR_SIZE: usize = 24; // 6 (magic) + 8 (version) + 8 (chunk size) + 1 (dtype code) + 1 (padding threshold)
const DTYPE_CODE: u8 = 4; // Assuming 4 represents the dtype code for np.int32

#[derive(Debug)]
pub enum DataType {
    Int32,
    Int64,
    Float32,
    Float64,
    UInt8,
    Int8,
    Int16,
    UInt16,
}

impl DataType {
    fn code(&self) -> u8 {
        match self {
            DataType::Int32 => 4,
            DataType::Int64 => 5,
            DataType::Float32 => 6,
            DataType::Float64 => 7,
            DataType::UInt8 => 1,
            DataType::Int8 => 2,
            DataType::Int16 => 3,
            DataType::UInt16 => 8,
        }
    }

    fn size(&self) -> usize {
        match self {
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::UInt8 => 1,
            DataType::Int8 => 1,
            DataType::Int16 => 2,
            DataType::UInt16 => 2,
        }
    }
}

pub struct MMapIndexedDatasetBuilder<W: Write> {
    file: W,
    dtype: DataType,
    sizes: Vec<usize>,
    doc_indices: Vec<u64>,
}

impl<W: Write> MMapIndexedDatasetBuilder<W> {
    pub fn new(mut file: W, dtype: DataType) -> io::Result<Self> {
        // Write the header
        file.write_all(HDR_MAGIC)?;
        file.write_u64::<LittleEndian>(VERSION)?;
        file.write_u8(dtype.code())?; // Data type code
        file.write_u8(0)?; // Placeholder for padding threshold

        Ok(MMapIndexedDatasetBuilder {
            file,
            dtype,
            sizes: Vec::new(),
            doc_indices: Vec::new(),
        })
    }

    pub fn add_item(&mut self, tensor: &[u8]) -> io::Result<()> {
        let size = tensor.len();
        self.sizes.push(size);

        // Write the tensor data to the file
        self.file.write_all(tensor)?;

        Ok(())
    }

    pub fn end_document(&mut self) -> io::Result<()> {
        self.doc_indices.push(self.sizes.len() as u64);
        Ok(())
    }

    pub fn finalize(mut self) -> io::Result<()> {
        // Write sizes and document indices
        self.file
            .write_u64::<LittleEndian>(self.sizes.len() as u64)?; // Number of sizes
        self.file
            .write_u64::<LittleEndian>(self.doc_indices.len() as u64)?; // Number of document indexes

        // Write sizes as 32-bit integers
        let sizes32: Vec<u32> = self.sizes.iter().map(|&s| s as u32).collect();
        self.file.write_all(bytemuck::cast_slice(&sizes32))?;

        // Get pointers and write them
        let pointers = self.get_pointers();
        self.file.write_all(bytemuck::cast_slice(&pointers))?;

        // Write document indices
        self.file
            .write_all(bytemuck::cast_slice(&self.doc_indices))?;

        Ok(())
    }

    fn get_pointers(&self) -> Vec<u64> {
        let elem_size = self.dtype.size() as u64;
        let mut pointers: Vec<u64> = self.sizes.iter().map(|&s| s as u64 * elem_size).collect();

        // Perform exclusive scan
        for i in 1..pointers.len() {
            pointers[i] += pointers[i - 1];
        }

        // Shift pointers to create exclusive scan
        if !pointers.is_empty() {
            pointers.insert(0, 0);
        }

        pointers
    }
}
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

fn main() -> io::Result<()> {
    // Your main function implementation here
    Ok(())
}

// Test the code
#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, ReadBytesExt};
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::path::PathBuf;

    #[test]
    fn test_mmap_indexed_dataset_builder() -> io::Result<()> {
        // Prepare a temporary file for testing
        let path = PathBuf::from("test_dataset.bin");
        let file = File::create(&path)?;

        // Initialize the dataset builder with Int32 data type
        let mut builder = MMapIndexedDatasetBuilder::new(file, DataType::Int32)?;

        // Add some items (example tensors)
        builder.add_item(
            &[1u32, 2, 3, 4]
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        builder.end_document()?;
        builder.add_item(
            &[5u32, 6, 7, 8]
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        builder.end_document()?;
        builder.finalize()?;

        // Read back the file to verify the contents
        let mut file = File::open(&path)?;
        let mut header = [0u8; HDR_SIZE];
        file.read_exact(&mut header)?;

        // Verify the header
        assert_eq!(&header[..10], HDR_MAGIC);
        assert_eq!(&header[10..18], VERSION.to_le_bytes());
        assert_eq!(header[18], DataType::Int32.code());

        // Read number of sizes and document indices
        let num_sizes = file.read_u64::<LittleEndian>()?;
        let num_doc_indices = file.read_u64::<LittleEndian>()?;
        assert_eq!(num_sizes, 2); // We added 2 items
        assert_eq!(num_doc_indices, 2); // Corresponding to 2 documents

        // Read sizes
        let mut sizes = vec![0u32; num_sizes as usize];
        file.read_exact(bytemuck::cast_slice_mut(&mut sizes))?;
        assert_eq!(sizes, vec![16, 16]); // Each tensor had 4 integers (4 bytes each)

        // Read pointers
        let mut pointers = vec![0u64; (num_sizes + 1) as usize];
        file.read_exact(bytemuck::cast_slice_mut(&mut pointers))?;
        assert_eq!(pointers, vec![0, 16, 32]); // Pointers to the start of each tensor

        // Read document indices
        let mut doc_indices = vec![0u64; num_doc_indices as usize];
        file.read_exact(bytemuck::cast_slice_mut(&mut doc_indices))?;
        assert_eq!(doc_indices, vec![1, 2]); // Indices of documents

        // Clean up
        std::fs::remove_file(path)?;

        Ok(())
    }
}

// Add this at the end of the file, outside of the `mod tests` block

#[cfg(test)]
fn main() {
    use std::process;
    let args: Vec<String> = std::env::args().collect();
    let test_name = args.get(1).map(|s| s.as_str());
    
    let result = tests::test_mmap_indexed_dataset_builder();
    
    if let Err(e) = result {
        eprintln!("Test failed: {:?}", e);
        process::exit(1);
    }
}
