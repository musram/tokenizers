- compile the program with optimizations

```bash
cargo build --release --example encode_llama --features="http"
```

- run the program

```bash
cargo run --release --example encode_llama --features="http" ../data/sample_10.jsonl ../data/sample_sc_50k.jsonl ../data/output_folder 2048 teknium/Llama-3.1-AlternateTokenizer 4
```

- use flamegraph to profile the program

```bash
   cargo flamegraph --bin encode_llama -- ../data/sample_10.jsonl ../data/sample_sc_50k.jsonl ../data/output_folder 2048 teknium/Llama-3.1-AlternateTokenizer 4
```
