- compile the program with optimizations

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
