- compile the program with optimizations

```bash
cargo build --release --example encode_llama_tokenizer_greedy_dp --features="http"
```

- check the binary

```bash
ls -l target/release/examples/encode_llama_tokenizer_greedy_dp
```

- run the binary

```bash
 target/release/examples/encode_llama_tokenizer_greedy_dp    ../data/sample_sc_50k.jsonl ../data/output_folder 2048    teknium/Llama-3.1-AlternateTokenizer 4
```

- run the program

```bash
cargo run --release --example encode_llama --features="http" ../data/sample_10.jsonl ../data/sample_sc_50k.jsonl ../data/output_folder 2048 teknium/Llama-3.1-AlternateTokenizer 4
```

- use perf

```bash
sudo perf record --call-graph dwarf    target/release/examples/encode_llama_tokenizer_greedy_dp    ../data/sample_sc_50k.jsonl ../data/output_folder 2048    teknium/Llama-3.1-AlternateTokenizer 4
```
