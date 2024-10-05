use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rayon::prelude::*;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

fn benchmark_tokenization(c: &mut Criterion) {
    let input_file = "./data/sample_sc_50k.json";
    let model_name = "teknium/Llama-3.1-AlternateTokenizer";
    let context_length = 2048;

    let tokenizer = Arc::new(Tokenizer::from_pretrained(model_name, None).unwrap());
    let file = File::open(input_file).unwrap();
    let reader = BufReader::new(file);

    let lines: Vec<String> = reader.lines().filter_map(Result::ok).collect();

    c.bench_function("tokenize_and_combine", |b| {
        b.iter(|| {
            let tokenized_entries: Vec<Vec<u32>> = lines
                .par_iter()
                .filter_map(|line| {
                    serde_json::from_str::<Value>(line)
                        .ok()
                        .and_then(|entry| process_entry(&entry, &tokenizer))
                })
                .collect();

            let bos_id = tokenizer.token_to_id("<s>").unwrap_or(1);
            let eos_id = tokenizer.token_to_id("</s>").unwrap_or(2);
            let separator_id = tokenizer.token_to_id("<reserved_special_token_2>").unwrap_or(3);

            black_box(combine_tokenized_messages_greedy(
                &tokenized_entries,
                context_length,
                bos_id,
                eos_id,
                separator_id,
            ));
        })
    });
}

fn process_entry(entry: &Value, tokenizer: &Tokenizer) -> Option<Vec<u32>> {
    entry["text"].as_str().and_then(|text| {
        tokenizer
            .encode(text, false)
            .ok()
            .map(|encoding| encoding.get_ids().to_vec())
    })
}

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
        if current.len() + message.len() + 1 <= max_tokens {
            if current.len() > 1 {
                current.push(separator_id);
            }
            current.extend_from_slice(message);
        } else {
            if current.len() > 1 {
                current.push(eos_id);
                combined.push(current);
            }
            current = vec![bos_id];
            current.extend_from_slice(message);
        }
    }

    if current.len() > 1 {
        current.push(eos_id);
        combined.push(current);
    }

    combined
}

criterion_group!(benches, benchmark_tokenization);
criterion_main!(benches);
