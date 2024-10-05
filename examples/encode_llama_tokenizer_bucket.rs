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

#[derive(Debug)]
struct Bucket {
    capacity: usize,
    remaining: usize,
}

#[derive(Debug)]
struct Document {
    length: usize,
    content: Vec<u32>,
}

fn fill_buckets(
    documents: &mut VecDeque<Document>,
    buckets: &mut Vec<Bucket>,
    padding_threshold: f64,
    pad_id: u32,
) -> Vec<Vec<u32>> {
    println!("Starting fill_buckets function");
    println!("Initial number of documents: {}", documents.len());
    println!("Initial number of buckets: {}", buckets.len());
    println!("Padding threshold: {}", padding_threshold);

    let mut training_buckets: Vec<Vec<u32>> = Vec::new();

    // Step 1: Sort documents by length in descending order
    documents
        .make_contiguous()
        .sort_by(|a, b| b.length.cmp(&a.length));
    println!("Documents sorted by length in descending order");
    println!(
        "Longest document length: {}",
        documents.front().map_or(0, |d| d.length)
    );

    while !documents.is_empty() {
        let mut bucket = buckets.pop().unwrap();
        println!("Processing new bucket with capacity: {}", bucket.capacity);
        let mut current_bucket: Vec<u32> = Vec::new();

        // Step 4: For each document in D
        while !documents.is_empty() {
            let document = documents.front_mut().unwrap();
            println!("Considering document of length: {}", document.length);

            // Step 5: Check if the document fits into the current bucket
            if document.length <= bucket.remaining {
                println!("Adding entire document of length {}", document.length);
                current_bucket.extend_from_slice(&document.content);
                bucket.remaining -= document.length;
                documents.pop_front(); // Remove document from D
                println!("Remaining space in bucket: {}", bucket.remaining);
            } else if current_bucket.is_empty() {
                // Step 7: If bucket is empty, add part of the document
                println!("Adding part of document to empty bucket");
                current_bucket.extend_from_slice(&document.content[0..bucket.remaining]);
                document.content = document.content[bucket.remaining..].to_vec();
                document.length -= bucket.remaining;
                bucket.remaining = 0; // Bucket is now full
                println!(
                    "Bucket filled. Remaining document length: {}",
                    document.length
                );
            } else {
                // Break if the current document does not fit
                println!("Document doesn't fit. Moving to next step.");
                break;
            }
        }

        // Step 12: Calculate remaining space in the bucket
        let remaining_space = bucket.capacity - current_bucket.len();
        println!("Remaining space: {}", remaining_space);

        // Step 13: Check padding condition
        if (remaining_space as f64) / (bucket.capacity as f64) > padding_threshold {
            println!("Remaining space exceeds padding threshold");
            if let Some(shortest) = documents.iter_mut().min_by_key(|d| d.length) {
                println!(
                    "Taking from shortest document with length {}",
                    shortest.length
                );
                let chunk_length = remaining_space.min(shortest.length);
                current_bucket.extend_from_slice(&shortest.content[0..chunk_length]);
                shortest.length -= chunk_length;
                shortest.content = shortest.content[chunk_length..].to_vec();

                if shortest.length == 0 {
                    println!("Shortest document completely used, removing empty documents");
                    let before = documents.len();
                    documents.retain(|d| d.length > 0); // Remove empty documents
                    let after = documents.len();
                    println!("Removed {} empty documents", before - after);
                }
            } else {
                println!("No suitable document found to fill the remaining space");
            }
        } else {
            // Step 17: Fill remaining space with padding
            println!("Filling remaining space with padding");
            current_bucket.extend(vec![pad_id; remaining_space]);
        }

        println!("Bucket filled, length: {}", current_bucket.len());
        assert_eq!(
            current_bucket.len(),
            bucket.capacity,
            "Bucket length mismatch"
        );
        training_buckets.push(current_bucket);
    }

    // Fill remaining buckets if any
    while !buckets.is_empty() {
        let bucket = buckets.pop().unwrap();
        println!(
            "Filling remaining bucket with capacity: {}",
            bucket.capacity
        );
        let current_bucket = vec![pad_id; bucket.capacity];
        training_buckets.push(current_bucket);
    }

    println!("Fill_buckets completed");
    println!("Number of training buckets: {}", training_buckets.len());
    println!("Remaining documents: {}", documents.len());
    training_buckets
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!(
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
    println!("Processing file: {}", input_file);

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
    let mut buckets = vec![Bucket {
        capacity: context_length,
        remaining: context_length,
    }];

    let padding_threshold = 0.1; // 10%

    let training_buckets = fill_buckets(&mut documents, &mut buckets, padding_threshold, pad_id);

    println!("training_buckets filled");

    // Write training_buckets to output file
    let output_file = File::create(output_file)?;
    let mut writer = BufWriter::new(output_file);

    println!("Writing training_buckets to output file");

    // Print the number of tokens in each bucket
    for (index, bucket) in training_buckets.iter().enumerate() {
        println!("Bucket {}: {} tokens", index + 1, bucket.len());
    }

    println!("Total number of buckets: {}", training_buckets.len());
    println!("Writing training_buckets to output file");

    for bucket in training_buckets {
        let json = serde_json::json!({"tokens": bucket});
        writeln!(writer, "{}", json.to_string())?;
    }

    println!("Processed {} and saved results", input_file);
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
