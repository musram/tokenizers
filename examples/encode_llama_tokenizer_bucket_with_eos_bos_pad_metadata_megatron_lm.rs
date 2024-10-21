use env_logger;
use env_logger::Builder;
use indicatif::{ProgressBar, ProgressStyle};
use lazy_static::lazy_static;
use log::LevelFilter;
use log::{error, info, warn};
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;

const HDR_MAGIC: &[u8] = b"MMIDIDX\x00\x00";
const VERSION: u64 = 1;
const DTYPE_CODE: u8 = 4; // Assuming 4 represents the dtype code for np.int32

// [ HDR_MAGIC (9 bytes) ][ VERSION (8 bytes) ][ TYPE CODE (1 byte) ]
// [ SIZES COUNT (8 bytes) ][ DOC INDICES COUNT (8 bytes) ]
// [ SIZES ( 4 bytes) ] [ POINTERS (8 bytes) ] [ DOC INDICES (8 bytes) ]
const HDR_SIZE: u64 = 9 + 8 + 1 + 8 + 8;

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

fn init_logger() {
    let log_file = File::create("program_log.txt").unwrap();
    Builder::new()
        .target(env_logger::Target::Pipe(Box::new(log_file)))
        .filter_level(LevelFilter::Info)
        .init();
}

#[derive(Debug)]
struct Document {
    length: usize,
    content: Vec<u32>,
}

fn fill_buckets(
    documents: &mut VecDeque<Document>,
    bucket_size: usize,
    padding_threshold: f64,
    seperator_id: u32,
) -> (Vec<Vec<u32>>, usize, usize) {
    info!("Starting fill_buckets function");

    let mut result_bucket: Vec<Vec<u32>> = Vec::new();
    let original_docs_count = documents.len();
    let mut split_docs_count = 0;

    // Sort documents by length in descending order
    documents
        .make_contiguous()
        .sort_by(|a, b| b.length.cmp(&a.length));

    while !documents.is_empty() {
        let mut current_bucket: Vec<u32> = Vec::new();
        let mut remaining_bucket_size = bucket_size;

        let i = 0;
        while i < documents.len() && remaining_bucket_size > 0 {
            let document = &documents[i];
            if document.length + 1 <= remaining_bucket_size {
                // +1 for separator
                // If document fits in the bucket
                current_bucket.extend_from_slice(&document.content);
                current_bucket.push(seperator_id);

                remaining_bucket_size -= document.length + 1; // Subtract document length and separator

                documents.remove(i);
            } else {
                // If document is too large, split the document
                if current_bucket.is_empty() {
                    let space_for_content = remaining_bucket_size.saturating_sub(1); // Reserve space for separator
                    current_bucket.extend_from_slice(&document.content[0..space_for_content]);
                    current_bucket.push(seperator_id);

                    let new_document = Document {
                        length: document.length - space_for_content,
                        content: document.content[space_for_content..].to_vec(),
                    };
                    documents[i] = new_document;
                    split_docs_count += 1;
                }
                break; // Stop processing this bucket
            }
        }

        // Handle remaining space in the bucket
        let remaining_space = bucket_size - current_bucket.len();
        if (remaining_space as f64) / (bucket_size as f64) > padding_threshold {
            // If remaining space is significant, fill with part of the shortest document
            if let Some(shortest) = documents.back_mut() {
                let words_to_add = remaining_space.saturating_sub(1).min(shortest.length); // Reserve space for separator
                current_bucket.extend_from_slice(&shortest.content[0..words_to_add]);
                current_bucket.push(seperator_id);

                shortest.length -= words_to_add;
                shortest.content = shortest.content[words_to_add..].to_vec();

                if shortest.length == 0 {
                    documents.pop_back();
                }
            }
        } else {
            info!("No padding added");
        }

        // Remove trailing separator if present
        if current_bucket.last() == Some(&seperator_id) {
            current_bucket.pop();
        }

        // Ensure the bucket size is not exceeded
        debug_assert!(
            current_bucket.len() <= bucket_size,
            "Bucket size exceeded: {} > {}",
            current_bucket.len(),
            bucket_size
        );

        // Add the current bucket to the result bucket
        result_bucket.push(current_bucket);
    }

    info!("Fill_buckets completed");
    info!("Number of training buckets: {}", result_bucket.len());
    info!("Remaining documents: {}", documents.len());

    (result_bucket, original_docs_count, split_docs_count)
}

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

fn calculate_index_file_size(
    header_size: u64,
    sizes: &[u32],
    pointers: &[u64],
    doc_index: &[u64],
) -> u64 {
    // Calculate size of the meta part
    let sizes_size: u64 = sizes.len() as u64 * std::mem::size_of::<u32>() as u64; // Size of sizes array
    let pointers_size: u64 = pointers.len() as u64 * std::mem::size_of::<u64>() as u64; // Size of pointers array
    let doc_index_size: u64 = doc_index.len() as u64 * std::mem::size_of::<u64>() as u64; // Size of document index

    let sizes_count_bytes: u64 = std::mem::size_of::<u64>() as u64;
    let doc_count_bytes: u64 = std::mem::size_of::<u64>() as u64;

    // Total meta part size
    let meta_part_size: u64 = header_size
        + sizes_size as u64
        + pointers_size as u64
        + doc_index_size as u64
        + sizes_count_bytes as u64
        + doc_count_bytes as u64;

    meta_part_size
}

fn calculate_data_file_size(sizes: &[u32], dtype_size: u8) -> u64 {
    sizes
        .iter()
        .map(|&size| size as u64 * dtype_size as u64)
        .sum()
}

fn calculate_total_bytes(
    header_size: u64,
    sizes: &[u32],
    pointers: &[u64],
    doc_index: &[u64],
    dtype_size: u8,
) -> u64 {
    // Calculate size of the meta part
    let sizes_size: u64 = sizes.len() as u64 * std::mem::size_of::<u32>() as u64; // Size of sizes array
    let pointers_size: u64 = pointers.len() as u64 * std::mem::size_of::<u64>() as u64; // Size of pointers array
    let doc_index_size: u64 = doc_index.len() as u64 * std::mem::size_of::<u64>() as u64; // Size of document index

    let sizes_count_bytes: u64 = std::mem::size_of::<u64>() as u64;
    let doc_count_bytes: u64 = std::mem::size_of::<u64>() as u64;

    // Total meta part size
    let meta_part_size: u64 = header_size
        + sizes_size as u64
        + pointers_size as u64
        + doc_index_size as u64
        + sizes_count_bytes as u64
        + doc_count_bytes as u64;

    // Calculate size of the data part
    let data_part_size: u64 = sizes
        .iter()
        .map(|&size| size as u64 * dtype_size as u64)
        .sum();

    // Total size
    meta_part_size + data_part_size
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    init_logger();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 7 {
        error!(
            "Usage: {} <input_files...> <output_dir> <context_length> <model_name> <num_workers> <dtype>",
            args[0]
        );
        std::process::exit(1);
    }

    let input_dir = &args[args.len() - 6];
    let output_dir = &args[args.len() - 5];
    let context_length: usize = args[args.len() - 4].parse()?;
    let model_name = &args[args.len() - 3];
    let num_workers: usize = args[args.len() - 2].parse()?;
    let dtype = &args[args.len() - 1];

    std::fs::create_dir_all(output_dir)?;

    let tokenizer = Arc::new(Tokenizer::from_pretrained(model_name, None)?);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build_global()
        .unwrap();

    let input_files: Vec<_> = std::fs::read_dir(input_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().map_or(false, |ext| ext == "jsonl"))
        .map(|entry| entry.path())
        .collect();

    for chunk in input_files.chunks(num_workers) {
        chunk.par_iter().try_for_each(|input_file| {
            process_jsonl_file(
                input_file.to_str().unwrap(),
                output_dir,
                context_length,
                dtype,
                Arc::clone(&tokenizer),
            )
        })?;
    }

    Ok(())
}

fn process_jsonl_file(
    input_file: &str,
    output_dir: &str,
    context_length: usize,
    dtype: &str,
    tokenizer: Arc<Tokenizer>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("Processing file: {}", input_file);

    let input_path = Path::new(input_file);
    let file_name = input_path.file_name().unwrap().to_str().unwrap();

    let file = File::open(input_file)?;
    let reader = BufReader::new(file);

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {wide_msg}")
            .unwrap(),
    );

    let start_time = Instant::now();

    let mut documents: VecDeque<Document> = reader
        .lines()
        .enumerate()
        .par_bridge()
        .filter_map(|(i, line)| {
            if i % 1000 == 0 {
                pb.set_message(format!("Processed {} lines", i));
            }
            line.ok()
                .and_then(|l| serde_json::from_str(&l).ok())
                .and_then(|entry: Value| process_entry(&entry, &tokenizer))
        })
        .collect();

    let total_time = start_time.elapsed();
    pb.finish_with_message(format!("Done processing in {:?}", total_time));

    // Define special tokens for sentence seperator, padding, begin of text and end of text
    //let pad_id = tokenizer.token_to_id("[PAD]").unwrap_or(0);
    let seperator_id = tokenizer
        .token_to_id("<|reserved_special_token_2|>")
        .unwrap_or(0);
    // Define special tokens
    let bos_id = tokenizer.token_to_id("<|begin_of_text|>").unwrap_or(0);
    let eos_id = tokenizer.token_to_id("<|end_of_text|>").unwrap_or(0);
    let pad_id = tokenizer
        .token_to_id("<|finetune_right_pad_id|>")
        .unwrap_or(0);

    let padding_threshold = 0.1; // 10%

    let total_original_length: usize = documents.iter().map(|doc| doc.length).sum();
    let (training_buckets, original_docs_count, split_docs_count) = fill_buckets(
        &mut documents,
        context_length,
        padding_threshold,
        seperator_id,
    );

    info!("training_buckets filled");

    // Create new buckets with bos_id, eos_id and pad_id
    let mut training_buckets_with_bos_eos = Vec::new();
    // Add the bos_id to the beginning of each bucket, eos_id to the end of each bucket and pad_id to the rest if the bucket is less than context_length  bucket is equal to context_length then remove the last token and add the eos_id
    for mut bucket in training_buckets {
        bucket.insert(0, bos_id);
        if bucket.len() == context_length {
            bucket.pop();
        }
        bucket.push(eos_id);
        while bucket.len() < context_length {
            bucket.push(pad_id);
        }
        training_buckets_with_bos_eos.push(bucket);
    }

    // Calculate ratios

    let total_padding: usize = training_buckets_with_bos_eos
        .iter()
        .map(|bucket| bucket.iter().filter(|&&token| token == pad_id).count())
        .sum();

    info!("Padding distribution:");
    for (i, bucket) in training_buckets_with_bos_eos.iter().enumerate() {
        let padding = context_length - bucket.len();
        info!("Bucket length {}: {}", i, bucket.len());
        info!("Bucket padding {}: {}", i, padding);
    }

    let total_training_samples = training_buckets_with_bos_eos.len();
    let truncation_ratio = split_docs_count as f64 / original_docs_count as f64;
    let padding_ratio = total_padding as f64 / total_original_length as f64;
    let concatenation_ratio = original_docs_count as f64 / total_training_samples as f64;

    info!(
        "Total number of buckets: {}",
        training_buckets_with_bos_eos.len()
    );
    info!("Writing training_buckets to output file");

    // Create directories if they don't exist
    let jsonl_dir = Path::new(output_dir).join("jsonl");
    let bin_dir = Path::new(output_dir).join("bin");
    std::fs::create_dir_all(&jsonl_dir)?;
    std::fs::create_dir_all(&bin_dir)?;

    // Write training_buckets to JSONL file
    info!("Writing training_buckets to JSONL file");

    let file_name = format!(
        "processed_{}",
        file_name.split('.').next().unwrap_or(file_name)
    );
    let jsonl_file = jsonl_dir.join(format!("processed_{}.jsonl", file_name));
    let jsonl_output = File::create(jsonl_file)?;
    let mut jsonl_writer = BufWriter::new(jsonl_output);

    for bucket in &training_buckets_with_bos_eos {
        let json = serde_json::json!({"tokens": bucket});
        writeln!(jsonl_writer, "{}", json.to_string())?;
    }
    jsonl_writer.flush()?;

    info!("Writing training_buckets to numpy.memmap file in megatron format");

    let dtype_bytes = get_dtype_code(dtype).unwrap_or(DTYPE_CODE);

    info!("The dtype in bytes: {}", dtype_bytes);

    // Get the sizes of each bucket
    let sizes: Vec<u32> = training_buckets_with_bos_eos
        .iter()
        .map(|bucket| bucket.len() as u32)
        .collect();

    info!("The sizes: {:?}", sizes);

    let sizes_count: u64 = sizes.len() as u64;

    info!("The sizes count: {}", sizes_count);

    let mut doc_idx: Vec<u64> = Vec::with_capacity(sizes.len());
    let mut cumulative_length: u64 = 0;
    for size in sizes.iter() {
        doc_idx.push(cumulative_length);
        cumulative_length += *size as u64;
    }

    let doc_count: u64 = doc_idx.len() as u64;

    info!("The doc_idx: {:?}", doc_idx);
    info!("The doc_count: {}", doc_count);

    let (pointers, total_bytes) = get_pointers_with_total(&sizes, DTYPE_CODE);

    info!("The pointers: {:?}", pointers);

    // HDR_SIZE + sizes * dtype_bytes + doc_count * 8 + pointers.len() * 8
    let sizes_bytes: u64 = sizes
        .iter()
        .map(|&size| size as u64 * dtype_bytes as u64)
        .sum();

    info!("The sizes_bytes: {}", sizes_bytes);

    let index_file_size: u64 = calculate_index_file_size(HDR_SIZE, &sizes, &pointers, &doc_idx);
    info!("The index_file_size: {}", index_file_size);

    let data_file_size: u64 = calculate_data_file_size(&sizes, dtype_bytes);
    info!("The data_file_size: {}", data_file_size);

    let mut index_memmap = memmap::MmapMut::map_anon(index_file_size as usize)?;

    info!("Created memory-mapped file with {} bytes", index_file_size);

    let mut data_memmap = memmap::MmapMut::map_anon(data_file_size as usize)?;

    info!("Created memory-mapped file with {} bytes", data_file_size);

    // Write header
    (&mut index_memmap[..HDR_MAGIC.len()]).copy_from_slice(HDR_MAGIC);
    (&mut index_memmap[HDR_MAGIC.len()..HDR_MAGIC.len() + 8])
        .copy_from_slice(&VERSION.to_le_bytes());
    index_memmap[HDR_MAGIC.len() + 8] = dtype_bytes;

    info!("Wrote header information");

    // Write sizes count
    (&mut index_memmap[HDR_MAGIC.len() + 9..HDR_MAGIC.len() + 17])
        .copy_from_slice(&(sizes_count).to_le_bytes());

    // Write doc indices count
    (&mut index_memmap[HDR_MAGIC.len() + 17..HDR_MAGIC.len() + 25])
        .copy_from_slice(&(doc_count).to_le_bytes());

    info!(
        "Wrote sizes count: {} and doc indices count: {}",
        sizes_count, doc_count
    );

    // Write sizes
    let sizes_start = HDR_SIZE;
    info!("sizes_start at : {}", sizes_start);

    for (i, size) in sizes.iter().enumerate() {
        let start = sizes_start + (i as u64) * 4; // Use 4 bytes for each size (u32)
        let end = start + 4;
        (&mut index_memmap[start as usize..end as usize]).copy_from_slice(&size.to_le_bytes());
    }

    info!("Wrote {} sizes", sizes.len());

    // write pointers
    let pointers_start: u64 = sizes_start + (sizes_count * 4);
    info!("pointers_start at : {}", pointers_start);

    for (i, pointer) in pointers.iter().enumerate() {
        let start = pointers_start + (i as u64) * 8;
        let end = start + 8;
        (&mut index_memmap[start as usize..end as usize]).copy_from_slice(&pointer.to_le_bytes());
    }

    info!("Wrote {} pointers", pointers.len());
    info!("pointers {:?}", pointers);

    // write doc indices
    let doc_idx_start = pointers_start + (pointers.len() as u64 * 8);

    info!("doc_idx_start starts at: {}", doc_idx_start);

    for (i, idx) in doc_idx.iter().enumerate() {
        let start = doc_idx_start + (i as u64) * 8;
        let end = start + 8;
        (&mut index_memmap[start as usize..end as usize]).copy_from_slice(&idx.to_le_bytes());
    }

    info!("Wrote {} doc indices", doc_idx.len());

    //let data_start = doc_idx_start + (doc_count * 8);
    let data_start = 0;

    info!("Data starts at: {}", data_start);

    // write data
    for (i, bucket) in training_buckets_with_bos_eos.iter().enumerate() {
        let start = data_start + (i as u64 * bucket.len() as u64 * dtype_bytes as u64);

        // Ensure the number of tokens to write is within the bounds
        let num_tokens = bucket.len().min(context_length);
        let end = start + (num_tokens as u64 * dtype_bytes as u64);

        if end > data_memmap.len() as u64 {
            warn!("Bucket {} exceeds allocated memory. Truncating. W", i);
            continue;
        }

        let slice: &mut [u8] = &mut data_memmap[start as usize..end as usize];
        for (j, &token) in bucket.iter().take(num_tokens).enumerate() {
            let token_start = j * std::mem::size_of::<u32>();
            let token_end = (j + 1) * std::mem::size_of::<u32>();
            if token_end <= slice.len() {
                slice[token_start..token_end].copy_from_slice(&token.to_le_bytes());
            } else {
                warn!(
                    "Token {} in bucket {} exceeds allocated memory. Skipping.",
                    j, i
                );
                break;
            }
        }
    }

    info!(
        "Wrote data for {} buckets",
        training_buckets_with_bos_eos.len()
    );

    // Persist the memmap to disk
    index_memmap.flush()?;
    data_memmap.flush()?;

    let index_memmap_file = bin_dir.join(format!("processed_{}.idx", file_name));
    let data_memmap_file = bin_dir.join(format!("processed_{}.bin", file_name));

    std::fs::write(&index_memmap_file, &index_memmap)?;
    std::fs::write(&data_memmap_file, &data_memmap)?;

    info!(
        "Flushed and wrote memory-mapped file to disk: {:?}",
        index_memmap
    );

    info!(
        "Flushed and wrote memory-mapped file to disk: {:?}",
        data_memmap
    );

    info!(
        "Wrote training_buckets to memmap file: {:?}",
        index_memmap_file
    );
    info!(
        "Wrote training_buckets to memmap file: {:?}",
        data_memmap_file
    );
    // Output bookkeeping metrics
    info!("Padding Ratio (rpad): {}", padding_ratio);
    info!("Truncation Ratio (rtru): {}", truncation_ratio);
    info!("Concatenation Ratio (rcat): {}", concatenation_ratio);
    info!("Processed {} and saved results", input_file);
    Ok(())
}

fn process_entry(entry: &Value, tokenizer: &Tokenizer) -> Option<Document> {
    entry["text"].as_str().and_then(|text| {
        tokenizer.encode(text, false).ok().map(|encoding| {
            let ids = encoding.get_ids().to_vec();
            Document {
                length: ids.len(),
                content: ids,
            }
        })
    })
}
