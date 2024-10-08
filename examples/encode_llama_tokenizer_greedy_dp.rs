use env_logger::Builder;
use indicatif::{ProgressBar, ProgressStyle};
use log::LevelFilter;
use rayon::prelude::*;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;

// Add these new imports
use lru::LruCache;
use std::cell::RefCell;
use std::num::NonZeroUsize;
fn init_logger() {
    let log_file = File::create("program_log.txt").unwrap();
    Builder::new()
        .target(env_logger::Target::Pipe(Box::new(log_file)))
        .filter_level(LevelFilter::Info)
        .init();
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

    // Build a global thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build_global()
        .unwrap();

    // Preload the tokenizer once
    let tokenizer = Arc::new(Tokenizer::from_pretrained(model_name, None).unwrap());

    // Process each file in parallel
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
    let reader = BufReader::with_capacity(1024 * 1024, file); // Increase buffer size

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {wide_msg}")
            .unwrap(),
    );

    let start_time = Instant::now();

    let tokenized_entries: Vec<Vec<u32>> = reader
        .lines()
        .par_bridge() // Use par_bridge() for parallel iteration
        .map(|line| {
            line.ok()
                .and_then(|l| serde_json::from_str(&l).ok())
                .and_then(|entry: Value| process_entry(&entry, &tokenizer))
        })
        .filter_map(|entry| entry)
        .collect();

    let total_time = start_time.elapsed();
    pb.finish_with_message(format!("Done processing in {:?}", total_time));

    let bos_id = tokenizer.token_to_id("<|begin_of_text|>").unwrap_or(1);
    let eos_id = tokenizer.token_to_id("<|end_of_text|>").unwrap_or(2);
    let separator_id = tokenizer
        .token_to_id("<|reserved_special_token_2|> ")
        .unwrap_or(3);

    // Combine tokenized entries using the greedy approach or dynamic programming
    // Consider using a parallel iterator here as well
    let combined_tokenized = combine_tokenized_messages_parallel(
        &tokenized_entries,
        context_length,
        bos_id,
        eos_id,
        separator_id,
    );

    // Use a larger buffer for writing
    let output_file = File::create(output_file)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, output_file);

    for combined in combined_tokenized {
        let json = serde_json::json!({"combined_text": combined});
        writeln!(writer, "{}", json.to_string())?;
    }

    println!("Processed {} and saved results", input_file);
    Ok(())
}

// Modify the process_entry function to use a thread-local cache
thread_local! {
    static ENCODING_CACHE: RefCell<LruCache<String, Vec<u32>>> = RefCell::new(LruCache::new(NonZeroUsize::new(1000).unwrap()));
}

fn process_entry(entry: &Value, tokenizer: &Tokenizer) -> Option<Vec<u32>> {
    entry["text"].as_str().and_then(|text| {
        ENCODING_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(cached) = cache.get(text) {
                Some(cached.clone())
            } else {
                let encoding = tokenizer.encode(text, false).ok()?;
                let ids = encoding.get_ids().to_vec();
                cache.put(text.to_string(), ids.clone());
                Some(ids)
            }
        })
    })
}

// Function to combine tokenized messages using greedy approach
fn combine_tokenized_messages_greedy(
    tokenized_messages: &[Vec<u32>],
    max_tokens: usize,
    bos_id: u32,
    eos_id: u32,
    separator_id: u32,
) -> Vec<Vec<u32>> {
    let mut combined = Vec::new();
    let mut current = Vec::with_capacity(max_tokens);
    current.push(bos_id);

    for message in tokenized_messages {
        if current.len() + message.len() + 2 <= max_tokens {
            // +2 for separator and EOS
            if current.len() > 1 {
                current.push(separator_id);
            }
            current.extend_from_slice(message);
        } else {
            current.push(eos_id);
            combined.push(std::mem::replace(
                &mut current,
                Vec::with_capacity(max_tokens),
            ));
            current.push(bos_id);
            current.extend_from_slice(message);
        }
    }

    if current.len() > 1 {
        current.push(eos_id);
        combined.push(current);
    }

    combined
}

// Function to combine tokenized messages dynamically
fn combine_tokenized_messages_dynamicprogramming(
    tokenized_messages: &[Vec<u32>],
    max_tokens: usize,
    bos_id: u32,
    eos_id: u32,
    separator_id: u32,
) -> Vec<Vec<u32>> {
    let n = tokenized_messages.len();
    let total_lengths: Vec<usize> = tokenized_messages.iter().map(|msg| msg.len()).collect();

    let cum_len: Vec<usize> = std::iter::once(0)
        .chain(total_lengths.iter().scan(0, |acc, &x| {
            *acc += x;
            Some(*acc)
        }))
        .collect();

    let mut dp = vec![0; n + 1];
    let mut prev = vec![0; n + 1];

    for i in 1..=n {
        dp[i] = dp[i - 1] + 1;
        prev[i] = i - 1;

        for j in (1..=i).rev() {
            let num_separators = i - j;
            let length = cum_len[i] - cum_len[j - 1] + num_separators + 2;

            if length > max_tokens {
                break;
            }

            if dp[j - 1] + 1 <= dp[i] {
                dp[i] = dp[j - 1] + 1;
                prev[i] = j - 1;
            }
        }
    }

    let mut combined = Vec::new();
    let mut i = n;
    while i > 0 {
        let j = prev[i];
        let mut current_message = vec![bos_id];
        current_message.reserve(cum_len[i] - cum_len[j] + i - j + 1);

        for k in j + 1..=i {
            if k > j + 1 {
                current_message.push(separator_id);
            }
            current_message.extend_from_slice(&tokenized_messages[k - 1]);
        }
        current_message.push(eos_id);
        combined.push(current_message);
        i = j;
    }

    combined.reverse();
    combined
}

// Add this new function for parallel combination
fn combine_tokenized_messages_parallel(
    tokenized_messages: &[Vec<u32>],
    max_tokens: usize,
    bos_id: u32,
    eos_id: u32,
    separator_id: u32,
) -> Vec<Vec<u32>> {
    let chunk_size = tokenized_messages.len() / rayon::current_num_threads().max(1);
    tokenized_messages
        .par_chunks(chunk_size)
        .flat_map(|chunk| {
            combine_tokenized_messages_dynamicprogramming(
                chunk,
                max_tokens,
                bos_id,
                eos_id,
                separator_id,
            )
        })
        .collect()
}
