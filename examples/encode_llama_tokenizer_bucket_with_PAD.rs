use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde_json::Value;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use log::{info, warn, error};
use env_logger;

fn init_logger() {
    env_logger::init();
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
    pad_id: u32,
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
        while i < documents.len() {
            let document = &documents[i];
            if document.length <= remaining_bucket_size {
                // If document fits in the bucket
                current_bucket.extend_from_slice(&document.content);
                remaining_bucket_size -= document.length;
                documents.remove(i);
            } else {
                // If document is too large, split the document
                if current_bucket.is_empty() {
                    // If the bucket is empty, truncate the document to fit
                    current_bucket.extend_from_slice(&document.content[0..remaining_bucket_size]);
                    let new_document = Document {
                        length: document.length - remaining_bucket_size,
                        content: document.content[remaining_bucket_size..].to_vec(),
                    };
                    documents[i] = new_document;
                    split_docs_count += 1;
                }
                break; // Stop processing this bucket if a document doesn't fit
            }
        }

        // Handle remaining space in the bucket
        let remaining_space = bucket_size - current_bucket.len();
        if (remaining_space as f64) / (bucket_size as f64) > padding_threshold {
            // If remaining space is significant, fill with part of the shortest document
            if let Some(shortest) = documents.back_mut() {
                let words_to_add = remaining_space.min(shortest.length);
                current_bucket.extend_from_slice(&shortest.content[0..words_to_add]);
                shortest.length -= words_to_add;
                shortest.content = shortest.content[words_to_add..].to_vec();

                if shortest.length == 0 {
                    documents.pop_back();
                }
            }
        } else {
            // Add padding if remaining space is not significant
            current_bucket.extend(vec![pad_id; remaining_space]);
        }

        // Add the current bucket to the result bucket
        result_bucket.push(current_bucket);
    }

    info!("Fill_buckets completed");
    info!("Number of training buckets: {}", result_bucket.len());
    info!("Remaining documents: {}", documents.len());

    (result_bucket, original_docs_count, split_docs_count)
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        error!(
            "Usage: {} <input_files...> <output_dir> <context_length> <model_name> <num_workers>",
            args[0]
        );
        std::process::exit(1);
    }

    let input_files: Vec<&str> = args[1..args.len() - 4].iter().map(AsRef::as_ref).collect();
    let output_dir = &args[args.len() - 4];
    let context_length: usize = args[args.len() - 3].parse()?;
    let model_name = &args[args.len() - 2];
    let num_workers: usize = args[args.len() - 1].parse()?;

    std::fs::create_dir_all(output_dir)?;

    let tokenizer = Arc::new(Tokenizer::from_pretrained(model_name, None)?);

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build_global()
        .unwrap();

    input_files.par_iter().try_for_each(|&input_file| {
        process_jsonl_file(
            input_file,
            output_dir,
            context_length,
            Arc::clone(&tokenizer),
        )
    })?;

    Ok(())
}

fn process_jsonl_file(
    input_file: &str,
    output_dir: &str,
    context_length: usize,
    tokenizer: Arc<Tokenizer>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("Processing file: {}", input_file);

    let input_path = Path::new(input_file);
    let file_name = input_path.file_name().unwrap().to_str().unwrap();
    let output_file = Path::new(output_dir).join(format!("processed_{}", file_name));

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

    let pad_id = tokenizer.token_to_id("[PAD]").unwrap_or(0);
    let padding_threshold = 0.1; // 10%

    let total_original_length: usize = documents.iter().map(|doc| doc.length).sum();
    let (training_buckets, original_docs_count, split_docs_count) =
        fill_buckets(&mut documents, context_length, padding_threshold, pad_id);

    info!("training_buckets filled");

    // Calculate ratios
    let total_padding: usize = training_buckets
        .iter()
        .map(|bucket| bucket.iter().filter(|&&token| token == pad_id).count())
        .sum();

    info!("Padding distribution:");
    for (i, bucket) in training_buckets.iter().enumerate() {
        let padding = bucket.iter().filter(|&&token| token == pad_id).count();
        info!("Bucket length {}: {}", i, bucket.len());
        info!("Bucket padding {}: {}", i, padding);
    }

    let total_training_samples = training_buckets.len();
    let truncation_ratio = split_docs_count as f64 / original_docs_count as f64;
    let padding_ratio = total_padding as f64 / total_original_length as f64;
    let concatenation_ratio = original_docs_count as f64 / total_training_samples as f64;

    info!("Total number of buckets: {}", training_buckets.len());
    info!("Writing training_buckets to output file");

    let output_file = File::create(output_file)?;
    let mut writer = BufWriter::new(output_file);

    for bucket in training_buckets {
        let json = serde_json::json!({"tokens": bucket});
        writeln!(writer, "{}", json.to_string())?;
    }

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
