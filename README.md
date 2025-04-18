## 💻 Installation

### From Releases

1. Visit the [Releases page](https://github.com/o-kollar/ShoeString/releases).  
2. Download the binary matching your OS/architecture.  
    | OS      | Architecture | Target                | Binary Suffix |
    | :------ | :----------- | :-------------------- | :---------------------- |
    | Linux   | amd64        | 64-bit Linux          | linux-amd64   |
    | Windows | amd64        | 64-bit Windows        | windows-amd64.exe |
    | macOS   | amd64        | macOS (Intel)         | darwin-amd64  |
    | macOS   | arm64        | macOS (Apple Silicon) | darwin-arm64  |
    | Linux   | arm64        | Raspberry Pi / ARM    | linux-arm64   |
3. Rename to `shoestring`, e.g.:
   ```bash
   mv shoestring-linux-amd64 shoestring && chmod +x shoestring
   ```

## 🧰 Usage

### Training

Train BPE tokenizer + model from scratch:

```bash
./shoestring \
  -train \
  -bpe-data path/to/bpe_corpus.txt \
  -model-data path/to/model_corpus.txt \
  -bpe-vocab-size 1000 \
  -embedding-dim 128 \
  -gru-hidden-size 128 \
  -gru-layers 2 \
  -num-experts 4 \
  -seq-length 64 \
  -batch-size 32 \
  -epochs 10 \
  -lr 0.001
```

What happens:
1. BPE tokenizer is fit on your corpus.  
2. Text is tokenized & batched.  
3. Model trains, with periodic checkpoints in `checkpoints/`.  
4. When done, an interactive chat REPL launches.

### Resume Training

Continue from an existing checkpoint:

```bash
./shoestring \
  -train \
  -checkpoint checkpoints/epoch_5.json \
  -model-data path/to/model_corpus.txt \
  -epochs 20
```

- Loads model + optimizer state from checkpoint.  
- Continues training until total epochs reach `-epochs`.

### Chat Mode

Start an interactive session with a trained model:

```bash
./shoestring -checkpoint checkpoints/epoch_10.json
```

Type your messages and receive model responses. Enter `exit` or `quit` to leave.

---

## ⚙️ Configuration Flags

Run `./shoestring -help` for the full list. Key flags include:

| Flag                 | Type     | Default | Description                                      |
| -------------------- | -------- | ------- | ------------------------------------------------ |
| `-train`             | bool     | `false` | Enable training mode                             |
| `-checkpoint PATH`   | string   | `""`    | Path to load or resume from a checkpoint         |
| `-bpe-data PATH`     | string   | `""`    | Corpus for BPE tokenizer training                |
| `-model-data PATH`   | string   | `""`    | Corpus for language model training               |
| `-bpe-vocab-size N`  | int      | `850`   | Number of BPE tokens                             |
| `-embedding-dim N`   | int      | `96`    | Dimensionality of token embeddings               |
| `-gru-hidden-size N` | int      | `96`    | Hidden size per GRU layer                        |
| `-gru-layers N`      | int      | `2`     | Number of GRU layers                             |
| `-num-experts N`     | int      | `6`     | Experts per MoE layer                            |
| `-seq-length N`      | int      | `80`    | Sequence length for training                     |
| `-batch-size N`      | int      | `16`    | Training batch size                              |
| `-epochs N`          | int      | `5`     | Total training epochs                            |
| `-lr F`              | float64 | `0.001` | Learning rate for AdamW optimizer                |
| `-max-response N`    | int      | `260`   | Max tokens to generate in chat mode              |

---