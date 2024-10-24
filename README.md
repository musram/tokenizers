<p align="center">
    <br>
    <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
    <br>
<p>
<p align="center">
    <img alt="Build" src="https://github.com/huggingface/tokenizers/workflows/Rust/badge.svg">
<<<<<<< HEAD
    <a href="https://github.com/huggingface/tokenizers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue&cachedrop">
    </a>
    <a href="https://pepy.tech/project/tokenizers">
        <img src="https://pepy.tech/badge/tokenizers/week" />
    </a>
</p>

Provides an implementation of today's most used tokenizers, with a focus on performance and
versatility.

## Main features:

 - Train new vocabularies and tokenize, using today's most used tokenizers.
 - Extremely fast (both training and tokenization), thanks to the Rust implementation. Takes
   less than 20 seconds to tokenize a GB of text on a server's CPU.
 - Easy to use, but also extremely versatile.
 - Designed for research and production.
 - Normalization comes with alignments tracking. It's always possible to get the part of the
   original sentence that corresponds to a given token.
 - Does all the pre-processing: Truncate, Pad, add the special tokens your model needs.

## Performances
Performances can vary depending on hardware, but running the [~/bindings/python/benches/test_tiktoken.py](bindings/python/benches/test_tiktoken.py) should give the following on a g6 aws instance:
![image](https://github.com/user-attachments/assets/2b913d4b-e488-4cbc-b542-f90a6c40643d)


## Bindings

We provide bindings to the following languages (more to come!):
  - [Rust](https://github.com/huggingface/tokenizers/tree/main/tokenizers) (Original implementation)
  - [Python](https://github.com/huggingface/tokenizers/tree/main/bindings/python)
  - [Node.js](https://github.com/huggingface/tokenizers/tree/main/bindings/node)
  - [Ruby](https://github.com/ankane/tokenizers-ruby) (Contributed by @ankane, external repo)
 
## Quick example using Python:

Choose your model between Byte-Pair Encoding, WordPiece or Unigram and instantiate a tokenizer:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())
```

You can customize how pre-tokenization (e.g., splitting into words) is done:

```python
from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()
```

Then training your tokenizer on a set of files just takes two lines of codes:

```python
from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"], trainer=trainer)
```

Once your tokenizer is trained, encode any text with just one line:
```python
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]
```

Check the [documentation](https://huggingface.co/docs/tokenizers/index)
or the [quicktour](https://huggingface.co/docs/tokenizers/quicktour) to learn more!
=======
    <a href="https://github.com/huggingface/tokenizers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue">
    </a>
    <a href="https://docs.rs/tokenizers/">
        <img alt="Doc" src="https://docs.rs/tokenizers/badge.svg">
    </a>
</p>
<br>


The core of `tokenizers`, written in Rust.
Provides an implementation of today's most used tokenizers, with a focus on performance and
versatility.

## What is a Tokenizer

A Tokenizer works as a pipeline, it processes some raw text as input and outputs an `Encoding`.
The various steps of the pipeline are:

1. The `Normalizer`: in charge of normalizing the text. Common examples of normalization are
   the [unicode normalization standards](https://unicode.org/reports/tr15/#Norm_Forms), such as `NFD` or `NFKC`.
   More details about how to use the `Normalizers` are available on the
   [Hugging Face blog](https://huggingface.co/docs/tokenizers/components#normalizers)
2. The `PreTokenizer`: in charge of creating initial words splits in the text. The most common way of
   splitting text is simply on whitespace.
3. The `Model`: in charge of doing the actual tokenization. An example of a `Model` would be
   `BPE` or `WordPiece`.
4. The `PostProcessor`: in charge of post-processing the `Encoding` to add anything relevant
   that, for example, a language model would need, such as special tokens.

### Loading a pretrained tokenizer from the Hub
```rust
use tokenizers::tokenizer::{Result, Tokenizer};

fn main() -> Result<()> {
    # #[cfg(feature = "http")]
    # {
        let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;

        let encoding = tokenizer.encode("Hey there!", false)?;
        println!("{:?}", encoding.get_tokens());
    # }
    Ok(())
}
```

### Deserialization and tokenization example

```rust
use tokenizers::tokenizer::{Result, Tokenizer, EncodeInput};
use tokenizers::models::bpe::BPE;

fn main() -> Result<()> {
    let bpe_builder = BPE::from_file("./path/to/vocab.json", "./path/to/merges.txt");
    let bpe = bpe_builder
        .dropout(0.1)
        .unk_token("[UNK]".into())
        .build()?;

    let mut tokenizer = Tokenizer::new(bpe);

    let encoding = tokenizer.encode("Hey there!", false)?;
    println!("{:?}", encoding.get_tokens());

    Ok(())
}
```

### Training and serialization example

```rust
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence, NormalizerWrapper};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{AddedToken, Model, Result, TokenizerBuilder};

use std::path::Path;

fn main() -> Result<()> {
    let vocab_size: usize = 100;

    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(vocab_size)
        .min_frequency(0)
        .special_tokens(vec![
            AddedToken::from(String::from("<s>"), true),
            AddedToken::from(String::from("<pad>"), true),
            AddedToken::from(String::from("</s>"), true),
            AddedToken::from(String::from("<unk>"), true),
            AddedToken::from(String::from("<mask>"), true),
        ])
        .build();

    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()?;

    let pretty = false;
    tokenizer
        .train_from_files(
            &mut trainer,
            vec!["path/to/vocab.txt".to_string()],
        )?
        .save("tokenizer.json", pretty)?;

    Ok(())
}
```

## Additional information

- tokenizers is designed to leverage CPU parallelism when possible. The level of parallelism is determined
  by the total number of core/threads your CPU provides but this can be tuned by setting the `RAYON_RS_NUM_THREADS`
  environment variable. As an example setting `RAYON_RS_NUM_THREADS=4` will allocate a maximum of 4 threads.
  **_Please note this behavior may evolve in the future_**

## Features
**progressbar**: The progress bar visualization is enabled by default. It might be disabled if
  compilation for certain targets is not supported by the [termios](https://crates.io/crates/termios)
  dependency of the [indicatif](https://crates.io/crates/indicatif) progress bar.
>>>>>>> 081235bd (Initial commit)
