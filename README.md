# 💻 Installation

#### From Releases

1. Visit the [Releases page](https://github.com/o-kollar/ShoeString/releases).  
2. Download the binary matching your OS/architecture.  
    | OS      | Architecture | Target                | Binary Suffix |
    | :------ | :----------- | :-------------------- | :------------ |
    | Linux   | amd64        | 64-bit Linux          | linux-amd64   |
    | Windows | amd64        | 64-bit Windows        | windows-amd64.exe |
    | macOS   | amd64        | macOS (Intel)         | darwin-amd64  |
    | macOS   | arm64        | macOS (Apple Silicon) | darwin-arm64  |
    | Linux   | arm64        | Raspberry Pi / ARM    | linux-arm64   |
3. Rename to `shoestring`, e.g.:
   ```bash
   mv shoestring-linux-amd64 shoestring && chmod +x shoestring
   ```

---


### 🧪 How to Use

#### 1. Train the Tokenizer
Train a Byte-Pair Encoding tokenizer on your corpus:

```bash
go run your_program.go \
  --mode bpe-train \
  --bpe-data path/to/your/corpus.txt \
  --bpe-output bpe_tokenizer.gob \
  --bpe-vocab-size 1000
```
- Saves the tokenizer as `bpe_tokenizer.gob`.

#### 2. Train the Model

```bash
go run your_program.go \
  --mode train \
  --bpe-path bpe_tokenizer.gob \
  --model-data path/to/your/training_data.txt \
  --validation-data path/to/your/validation_data.txt \
  --embedding-dim 128 \
  --gru-hidden-size 128 \
  --gru-layers 2 \
  --num-experts 4 \
  --batch-size 32 \
  --seq-length 100 \
  --epochs 10 \
  --lr 0.0005 \
  --wd 0.01
  # Optional: --checkpoint path/to/resume_checkpoint.gob
```

- Loads `bpe_tokenizer.gob`.
- If `--checkpoint` is specified, resumes from checkpoint.
- If not, starts fresh using the BPE vocab size.
- Saves checkpoints to `checkpoints/checkpoint_epoch_X.gob`.

#### 3. Chat with the Model

```bash
go run your_program.go \
  --mode chat \
  --bpe-path bpe_tokenizer.gob \
  --checkpoint checkpoints/checkpoint_epoch_9.gob \
  --max-response 300
```

- Loads tokenizer and model checkpoint.
- Starts interactive REPL-style chat.


---

## ⚙️ Configuration Flags

Run `./shoestring -help` for the full list. Highlights below:

| Flag                     | Type     | Default      | Description                                         |
|--------------------------|----------|--------------|-----------------------------------------------------|
| `-mode`                  | string   | `"chat"`     | Execution mode: `bpe-train`, `train`, or `chat`     |
| `-bpe-data`              | string   | `""`         | Corpus file for BPE training (used in `bpe-train`)  |
| `-bpe-output`            | string   | `""`         | Path to save trained BPE file (overrides `-bpe-path`) |
| `-bpe-path`              | string   | `bpe_tokenizer.gob` | Path to BPE state (load for `train/chat`, save for `bpe-train`) |
| `-bpe-vocab-size`        | int      | `850`        | BPE vocabulary size                                 |
| `-embedding-dim`         | int      | `96`         | Token embedding dimension                           |
| `-gru-hidden-size`       | int      | `96`         | Hidden units per GRU layer                          |
| `-gru-layers`            | int      | `2`          | Number of GRU layers                                |
| `-num-experts`           | int      | `6`          | Number of experts in MoE layers                     |
| `-seq-length`            | int      | `80`         | Sequence length                                     |
| `-batch-size`            | int      | `16`         | Batch size                                          |
| `-epochs`                | int      | `5`          | Training epochs                                     |
| `-lr`                    | float64  | `0.001`      | Learning rate (AdamW)                               |
| `-wd`                    | float64  | `0.01`       | Weight decay (AdamW)                                |
| `-eps-rmsnorm`           | float64  | `1e-5`       | RMSNorm epsilon                                     |
| `-eps-adamw`             | float64  | `1e-8`       | AdamW epsilon                                       |
| `-grad-clip`             | float64  | `5.0`        | Gradient clipping value                             |
| `-max-response`          | int      | `260`        | Max tokens to generate in chat mode                 |
| `-model-data`            | string   | `""`         | Training data path (required for `train`)           |
| `-validation-data`       | string   | `""`         | Optional validation set                             |
| `-checkpoint`            | string   | `""`         | Path to model checkpoint to load/resume from        |

