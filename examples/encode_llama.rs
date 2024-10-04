use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use std::time::Instant;

// Main function and argument parsing
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
        process_jsonl_file(input_file, output_dir, context_length, Arc::clone(&tokenizer))
    })?;

    Ok(())
}

// Function to process a single JSONL file
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

    let tokenized_entries: Vec<Vec<u32>> = reader
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

    let bos_id = tokenizer.token_to_id("<|begin_of_text|>").unwrap_or(1);
    let eos_id = tokenizer.token_to_id("<|end_of_text|>").unwrap_or(2);
    let separator_id = tokenizer.token_to_id("<|reserved_special_token_2|>").unwrap_or(3);

    let combined_tokenized = combine_tokenized_messages_dynamicprogramming(
        &tokenized_entries,
        context_length,
        bos_id,
        eos_id,
        separator_id,
    );

    let output_file = File::create(output_file)?;
    let mut writer = BufWriter::new(output_file);

    for combined in combined_tokenized {
        let json = serde_json::json!({"combined_text": combined});
        writeln!(writer, "{}", json.to_string())?;
    }

    println!("Processed {} and saved results", input_file);
    Ok(())
}

// Function to process a single entry
fn process_entry(entry: &Value, tokenizer: &Tokenizer) -> Option<Vec<u32>> {
    entry["text"].as_str().and_then(|text| {
        tokenizer
            .encode(text, false)
            .ok()
            .map(|encoding| encoding.get_ids().to_vec())
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
    let mut current = Vec::with_capacity(max_tokens); // Pre-allocate maximum size

    current.push(bos_id);

    for message in tokenized_messages {
        // Check if adding this message would exceed the max token limit
        if current.len() + message.len() + 1 <= max_tokens { // +1 for separator
            if current.len() > 1 { // If not the first message, add separator
                current.push(separator_id);
            }
            current.extend_from_slice(message);
        } else {
            if current.len() > 1 { // Ensure we have more than just the BOS token
                current.push(eos_id);
                combined.push(current);
            }
            current = vec![bos_id];
            current.extend_from_slice(message);
        }
    }

    // Final message check
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

    // Precompute cumulative lengths
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
