use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = Tokenizer::from_pretrained("teknium/Llama-3.1-AlternateTokenizer", None)?;

    let encoding = tokenizer.encode("Hey there dear friend!", false)?;
    assert_eq!(
        encoding.get_tokens(),
        &["Hey", "Ġthere", "Ġdear", "Ġfriend", "!"]
    );

    let encoding_eos = tokenizer.encode("<|end_of_text|>", false)?;
    let decoding_eos = tokenizer.decode(encoding_eos.get_ids(), true)?;

    let encoding_bos = tokenizer.encode("<|begin_of_text|>", false)?;
    let decoding_bos = tokenizer.decode(encoding_bos.get_ids(), true)?;

    let special_tokens = tokenizer.encode("<|reserved_special_token_2|>", false)?;
    let decoding_special_tokens = tokenizer.decode(special_tokens.get_ids(), true)?;

    println!("encoding_eos: {:?}", encoding_eos.get_ids());
    println!("encoding_bos: {:?}", encoding_bos.get_ids());
    println!("decoding_eos: {}", decoding_eos.as_str());
    println!("decoding_bos: {}", decoding_bos.as_str());
    println!("special_tokens: {:?}", special_tokens.get_ids());
    println!(
        "decoding_special_tokens: {}",
        decoding_special_tokens.as_str()
    );
    // let data = std::fs::read_to_string("data/big.txt")?;
    // let data: Vec<_> = data.lines().collect();
    // let add_special_tokens = false;
    // tokenizer.encode_batch_char_offsets(data, add_special_tokens)?;
    Ok(())
}
