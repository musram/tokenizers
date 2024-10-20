use serde::de::value::CowStrDeserializer;
use std::convert::TryInto;
use std::fs::{File, OpenOptions};
use std::io::Seek;
use std::io::{self, Read, Write};
use std::mem;
use std::slice;
use tempfile::tempdir;

const HDR_MAGIC: &[u8] = b"MMIDIDX\x00\x00";
const VERSION: u64 = 1;

#[derive(Debug, PartialEq, Clone)]
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
    fn from_str(dtype: &str) -> DataType {
        match dtype {
            "np.32" => DataType::Int32,
            "np.64" => DataType::Int64,
            "fp.32" => DataType::Float32,
            "fp.64" => DataType::Float64,
            "u8" => DataType::UInt8,
            "i8" => DataType::Int8,
            "i16" => DataType::Int16,
            "u16" => DataType::UInt16,
            _ => panic!("Invalid dtype: {}", dtype),
        }
    }

    fn size(&self) -> u8 {
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

#[derive(Debug)]
pub struct MMapIndexedDatasetBuilder {
    data_file: File,
    sizes: Vec<usize>,
    doc_idx: Vec<usize>,
    dtype: DataType,
}

impl MMapIndexedDatasetBuilder {
    pub fn new(out_file: &str, dtype: DataType) -> io::Result<Self> {
        let data_file = OpenOptions::new().write(true).create(true).open(out_file)?;

        Ok(Self {
            data_file,
            sizes: Vec::new(),
            doc_idx: vec![0],
            dtype,
        })
    }

    pub fn add_item(&mut self, tensor: &[i64]) -> io::Result<()> {
        let converted_tensor: Vec<u8> = match self.dtype {
            DataType::Int32 => tensor
                .iter()
                .flat_map(|&x| (x as i32).to_le_bytes().to_vec())
                .collect(),
            DataType::Int64 => tensor
                .iter()
                .flat_map(|&x| x.to_le_bytes().to_vec())
                .collect(),
            DataType::Float32 => tensor
                .iter()
                .flat_map(|&x| (x as f32).to_le_bytes().to_vec())
                .collect(),
            DataType::Float64 => tensor
                .iter()
                .flat_map(|&x| (x as f64).to_le_bytes().to_vec())
                .collect(),
            DataType::UInt8 => tensor.iter().map(|&x| x as u8).collect(),
            DataType::Int8 => tensor.iter().map(|&x| x as u8).collect(),
            DataType::Int16 => tensor
                .iter()
                .flat_map(|&x| (x as i16).to_le_bytes().to_vec())
                .collect(),
            DataType::UInt16 => tensor
                .iter()
                .flat_map(|&x| (x as u16).to_le_bytes().to_vec())
                .collect(),
        };
        self.data_file.write_all(&converted_tensor)?;
        self.sizes.push(tensor.len());
        Ok(())
    }

    pub fn add_doc(&mut self, tensor: &[i64], sizes: &[usize]) -> io::Result<()> {
        self.data_file.write_all(bytemuck::cast_slice(tensor))?;
        self.sizes.extend_from_slice(sizes);
        self.doc_idx.push(self.sizes.len());
        Ok(())
    }

    pub fn end_document(&mut self) {
        self.doc_idx.push(self.sizes.len());
    }

    pub fn finalize(&mut self, index_file: &str) -> io::Result<()> {
        self.data_file.sync_all()?;
        let mut index = Index::new(index_file, self.dtype.clone())?;
        index.write(&self.sizes, &self.doc_idx)?;
        Ok(())
    }

    pub fn merge_file(&mut self, another_file: &str) -> io::Result<()> {
        let index = Index::new(&index_file_path(another_file), self.dtype.clone())?;
        let offset = self.sizes.len();
        self.sizes.extend_from_slice(&index.sizes()?);
        self.doc_idx
            .extend(index.doc_idx()?[1..].iter().map(|&idx| offset + idx));

        let mut another_data_file = File::open(data_file_path(another_file))?;
        io::copy(&mut another_data_file, &mut self.data_file)?;
        Ok(())
    }
}
pub struct Index {
    file: File,
    dtype_bytes: u8,
}

impl Index {
    pub fn new(path: &str, dtype: DataType) -> io::Result<Self> {
        let file = OpenOptions::new().write(true).create(true).open(path)?;
        let dtype_bytes = dtype.size() as u8;
        Ok(Self { file, dtype_bytes })
    }

    pub fn write(&mut self, sizes: &[usize], doc_idx: &[usize]) -> io::Result<()> {
        self.file.write_all(HDR_MAGIC)?;
        self.file.write_all(&VERSION.to_le_bytes())?;
        self.file.write_all(&[self.dtype_bytes])?;
        self.file.write_all(&(sizes.len() as u64).to_le_bytes())?;
        self.file.write_all(&(doc_idx.len() as u64).to_le_bytes())?;

        for &size in sizes {
            self.file.write_all(&(size as u32).to_le_bytes())?;
        }

        for &doc in doc_idx {
            self.file.write_all(&(doc as u64).to_le_bytes())?;
        }

        Ok(())
    }

    pub fn sizes(&self) -> io::Result<Vec<usize>> {
        // Implementation to read sizes from the file
        unimplemented!()
    }

    pub fn doc_idx(&self) -> io::Result<Vec<usize>> {
        // Implementation to read doc_idx from the file
        unimplemented!()
    }
}

// Helper function for exclusive scan
fn exscan_from_cumsum(arr: &mut [usize]) {
    if arr.len() > 1 {
        arr.copy_within(0..arr.len() - 1, 1);
    }
    if !arr.is_empty() {
        arr[0] = 0;
    }
}

// Helper function to get pointers with total
fn get_pointers_with_total(sizes: &[usize], dtype_bytes: u8) -> (Vec<usize>, usize) {
    let mut pointers: Vec<usize> = Vec::with_capacity(sizes.len());
    let mut cumulative_sum = 0;
    for &size in sizes {
        pointers.push(cumulative_sum * dtype_bytes as usize);
        cumulative_sum += size;
    }
    let total_bytes = cumulative_sum * dtype_bytes as usize;
    (pointers, total_bytes)
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryInto;
    use std::io::Seek;
    use tempfile::tempdir;

    #[test]
    fn test_data_type() {
        assert_eq!(DataType::from_str("np.32"), DataType::Int32);
        assert_eq!(DataType::from_str("np.64"), DataType::Int64);
        assert_eq!(DataType::from_str("fp.32"), DataType::Float32);
        assert_eq!(DataType::from_str("fp.64"), DataType::Float64);
        assert_eq!(DataType::from_str("u8"), DataType::UInt8);

        assert_eq!(DataType::Int32.size(), 4);
        assert_eq!(DataType::Int64.size(), 8);
        assert_eq!(DataType::Float32.size(), 4);
        assert_eq!(DataType::Float64.size(), 8);
        assert_eq!(DataType::UInt8.size(), 1);
    }

    #[test]
    fn test_mmap_indexed_dataset_builder() -> io::Result<()> {
        let dir = tempdir()?;
        let data_path = dir.path().join("test_data.bin");
        let index_path = dir.path().join("test_index.idx");

        let dtype = "np.32";
        let dtype_code = DataType::from_str(dtype);
        let mut builder =
            MMapIndexedDatasetBuilder::new(data_path.to_str().unwrap(), dtype_code.clone())?;

        builder.add_item(&[1, 2, 3])?;
        builder.add_item(&[4, 5])?;
        builder.end_document();

        builder.add_item(&[6, 7, 8, 9])?;
        builder.end_document();

        builder.finalize(index_path.to_str().unwrap())?;

        // Verify data file
        let mut data_file = File::open(data_path)?;
        let mut buffer = Vec::new();
        data_file.read_to_end(&mut buffer)?;
        assert_eq!(buffer.len(), 36); // 9 * 4 bytes (i32)

        // Verify index file
        let mut index_file = File::open(index_path)?;
        let mut index_buffer = Vec::new();
        index_file.read_to_end(&mut index_buffer)?;

        // Check header
        assert_eq!(&index_buffer[0..9], HDR_MAGIC);
        assert_eq!(index_buffer[9..17], VERSION.to_le_bytes());
        assert_eq!(index_buffer[17], dtype_code.size());

        // Check sizes and doc_idx counts
        let sizes_count = u64::from_le_bytes(index_buffer[18..26].try_into().unwrap());
        let doc_idx_count = u64::from_le_bytes(index_buffer[26..34].try_into().unwrap());
        assert_eq!(sizes_count, 3);
        assert_eq!(doc_idx_count, 3);

        Ok(())
    }

    #[test]
    fn test_index() -> io::Result<()> {
        let dir = tempdir()?;
        let index_path = dir.path().join("test_index.idx");

        let dtype = "np.32";
        let dtype_code = DataType::from_str(dtype);
        let mut index = Index::new(index_path.to_str().unwrap(), dtype_code)?;
        index.write(&[2, 3, 5], &[0, 2, 5])?;

        // Verify written data
        let mut file = File::open(index_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        assert_eq!(&buffer[0..9], HDR_MAGIC);
        assert_eq!(u64::from_le_bytes(buffer[18..26].try_into().unwrap()), 3); // sizes length
        assert_eq!(u64::from_le_bytes(buffer[26..34].try_into().unwrap()), 3); // doc_idx length

        // Check sizes
        assert_eq!(u32::from_le_bytes(buffer[34..38].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(buffer[38..42].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(buffer[42..46].try_into().unwrap()), 5);

        // Check doc_idx
        assert_eq!(u64::from_le_bytes(buffer[46..54].try_into().unwrap()), 0);
        assert_eq!(u64::from_le_bytes(buffer[54..62].try_into().unwrap()), 2);
        assert_eq!(u64::from_le_bytes(buffer[62..70].try_into().unwrap()), 5);

        Ok(())
    }

    #[test]
    fn test_add_item() -> io::Result<()> {
        let dir = tempdir()?;
        let data_path = dir.path().join("test_data.bin");
        let index_path = dir.path().join("test_index.idx");

        let dtype = "np.32";
        let mut builder =
            MMapIndexedDatasetBuilder::new(data_path.to_str().unwrap(), DataType::from_str(dtype))?;
        builder.add_item(&[10, 20, 30])?;
        builder.end_document();
        builder.finalize(index_path.to_str().unwrap())?;

        let mut file = File::open(data_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        assert_eq!(buffer.len(), 12); // 3 * 4 bytes (i32)

        Ok(())
    }

    #[test]
    fn test_exscan_from_cumsum() {
        let mut arr = vec![10, 30, 35, 50];
        exscan_from_cumsum(&mut arr);
        assert_eq!(arr, vec![0, 10, 30, 35]);

        let mut empty: Vec<usize> = vec![];
        exscan_from_cumsum(&mut empty);
        // assert_eq!(empty, vec![]);

        let mut single = vec![5];
        exscan_from_cumsum(&mut single);
        assert_eq!(single, vec![0]);
    }

    #[test]
    fn test_get_pointers_with_total() {
        let dtype = "np.32";
        let dtype_bytes = DataType::from_str(dtype).size();
        println!("dtype_bytes: {}", dtype_bytes);
        let sizes = vec![2, 3, 5];
        let (pointers, total) = get_pointers_with_total(&sizes, dtype_bytes);
        assert_eq!(pointers, vec![0, 8, 20]); // 0, 2*8, (2+3)*8
        assert_eq!(total, 40); // (2+3+5) * 8 bytes

        let empty: Vec<usize> = vec![];
        let (empty_pointers, empty_total) = get_pointers_with_total(&empty, dtype_bytes);
        //assert_eq!(empty_pointers, vec![]);
        assert_eq!(empty_total, 0);
    }

    #[test]
    fn test_file_path_helpers() {
        assert_eq!(index_file_path("data.bin"), "data.bin.idx");
        assert_eq!(data_file_path("data.bin"), "data.bin");
    }

    #[test]
    fn test_header() -> io::Result<()> {
        let dir = tempdir()?;
        let index_path = dir.path().join("test_header.idx");
        let dtype = "np.32";
        let dtype_code = DataType::from_str(dtype);
        let mut index = Index::new(index_path.to_str().unwrap(), dtype_code.clone())?;
        index.write(&[1, 2, 3], &[0, 1, 3])?;

        // Reopen the file to read its contents
        let mut index_file = File::open(index_path)?;
        let mut index_buffer = Vec::new();
        index_file.read_to_end(&mut index_buffer)?;

        // Check header
        assert_eq!(&index_buffer[0..9], HDR_MAGIC);
        assert_eq!(index_buffer[9..17], VERSION.to_le_bytes());
        assert_eq!(index_buffer[17], dtype_code.size());

        // Check sizes count
        println!("index_buffer[18..26]: ");
        let sizes_count = match u64::from_le_bytes(index_buffer[18..26].try_into().unwrap()) {
            count => {
                println!("sizes_count: {}", count);
                count
            }
        };
        assert_eq!(sizes_count, 3, "Sizes count doesn't match");

        Ok(())
    }
}

fn index_file_path(file_path: &str) -> String {
    format!("{}.idx", file_path)
}

fn data_file_path(file_path: &str) -> String {
    file_path.to_string()
}

fn main() -> io::Result<()> {
    let dtype = "np.32";
    let mut builder = MMapIndexedDatasetBuilder::new("output.bin", DataType::from_str(dtype))?;
    builder.add_item(&[1, 2, 3]).unwrap();
    builder.finalize("index.idx")?;
    Ok(())
}
