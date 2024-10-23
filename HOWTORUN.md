- compile the program with optimizations

```bash
cd tokenizers
```

This should be /home/xxxxx/softwares/programs/rust_projects/tokenizers/tokenizers

```bash
pwd
```

- compile the program

```bash
cargo build --release --example encode_llama_tokenizer_bucket_with_eos_bos_pad_metadata_megatron_lm --features="http"
```

- check the binary

```bash
ls -l target/release/examples/encode_llama_tokenizer_bucket_with_eos_bos_pad_metadata_megatron_lm
```

- run the binary

```bash
target/release/examples/encode_llama_tokenizer_bucket_with_eos_bos_pad_metadata_megatron_lm   ../data ../data/output_folder 2048    teknium/Llama-3.1-AlternateTokenizer 4 np.int32
```

- run the program

```bash
cargo run --release --example encode_llama_tokenizer_bucket_with_eos_bos_pad_metadata_megatron_lm --features="http" ../data/sample_10.jsonl ../data/sample_sc_50k.jsonl ../data/output_folder 2048 teknium/Llama-3.1-AlternateTokenizer 4 np.int32
```

- use perf

```bash
sudo perf record --call-graph dwarf    target/release/examples/encode_llama_tokenizer_bucket_with_eos_bos_pad_metadata_megatron_lm    ../data/sample_sc_50k.jsonl ../data/output_folder 2048    teknium/Llama-3.1-AlternateTokenizer 4 np.int32
```

- validate the output

```bash
cd tokenizers
```

```bash
pwd
```

should be /home/xxxxx/softwares/programs/rust_projects/tokenizers/

```bash
python3 examples/validate_megatron_data.py
```

-- Compile merges

```bash
cargo build --example megatron_merges
```

-- Run the merges

```bash
target/release/examples/megatron_merges ../data/output_folder/merges np.int32 ../data/output_folder/bin  merge 
```

- test the merges

```bash
cargo test --example megatron_merges
```
