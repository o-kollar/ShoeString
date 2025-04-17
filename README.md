# Shoestring

This project implements a compact language model in Go, designed primarily for **next-token prediction** tasks, especially on **resource-constrained hardware**. It features a custom GRU architecture.

The model can be trained from text data and then used interactively via a command-line chat interface.

## Features

*   **BPE Tokenizer:** Uses Byte Pair Encoding for subword tokenization, trainable from a corpus. Special tokens (`[USER]`, `[BOT]`, `[EOS]`, `[PAD]`, `[UNK]`) are handled.
*   **Training:**
    *   Trains on plain text files.
    *   Uses AdamW optimizer with configurable hyperparameters (learning rate, weight decay, epsilon).
    *   Supports gradient clipping for training stability.
    *   Checkpointing system to save and resume training progress (includes model weights, optimizer state, BPE tokenizer state, and configuration).
*   **Inference:**
    *   Interactive command-line chat interface.
    *   Generates responses token by token based on the trained model.
    *   Configurable maximum response length.
*   **Cross-Platform:** Pre-compiled binaries are provided for common operating systems and architectures.

## Purpose

The main goal of this project is to provide a functional example of a modern, efficient language model architecture. 
It is specifically tailored for environments where computational resources (CPU, RAM) might be limited, making it potentially suitable for edge devices or low-power servers where large transformer models are infeasible. 
Its primary function is sequential prediction (predicting the next token given previous ones), which forms the basis for text generation.

## Getting Started (Using Pre-compiled Binaries)

1.  **Download:** Download the appropriate binary for your system from the releases section (or wherever you are distributing them).

    | OS      | Architecture | Target                | Binary Suffix (Example) |
    | :------ | :----------- | :-------------------- | :---------------------- |
    | Linux   | amd64        | 64-bit Linux          | `min-gru-linux-amd64`   |
    | Windows | amd64        | 64-bit Windows        | `min-gru-windows-amd64.exe` |
    | macOS   | amd64        | macOS (Intel)         | `min-gru-darwin-amd64`  |
    | macOS   | arm64        | macOS (Apple Silicon) | `min-gru-darwin-arm64`  |
    | Linux   | arm64        | Raspberry Pi / ARM    | `min-gru-linux-arm64`   |

    *(Replace `min-gru` with the actual name you give the executable)*

2.  **Make Executable (Linux/macOS):**
    ```bash
    chmod +x ./min-gru-<os>-<arch>
    ```

3.  **Prepare Data:** You will need plain text files (`.txt`) for training:
    *   **BPE Training Data:** A corpus of text used to train the BPE tokenizer (e.g., `bpe_corpus.txt`). This should be representative of the language/domain you want the model to understand.
    *   **Model Training Data:** A corpus of text used to train the language model itself (e.g., `model_corpus.txt`). This often includes conversational data or structured text relevant to the desired task. It should ideally be pre-formatted with `[USER]` and `[BOT]` tokens if intended for chat.

4.  **Usage:**

    *   **Train Everything (BPE + Model) from Scratch:**
        ```bash
        # On Linux/macOS
        ./min-gru-linux-amd64 -train \
                              -bpe-data path/to/bpe_corpus.txt \
                              -model-data path/to/model_corpus.txt \
                              -bpe-vocab-size 850 \
                              -embedding-dim 96 \
                              -gru-hidden-size 96 \
                              -gru-layers 2 \
                              -num-experts 6 \
                              -seq-length 80 \
                              -batch-size 16 \
                              -epochs 5 \
                              -lr 0.001

        # On Windows
        .\min-gru-windows-amd64.exe -train -bpe-data path\to\bpe_corpus.txt -model-data path\to\model_corpus.txt [other flags...]
        ```
        This command will:
        1.  Train the BPE tokenizer using `bpe_corpus.txt`.
        2.  Prepare the `model_corpus.txt` using the trained tokenizer.
        3.  Initialize and train the GRU model.
        4.  Save checkpoints (including BPE state) periodically in the `checkpoints/` directory.
        5.  After training finishes, it will automatically start the chat interface.

    *   **Resume Training from Checkpoint:**
        ```bash
        ./min-gru-linux-amd64 -train \
                              -checkpoint checkpoints/checkpoint_epoch_2.json \
                              -model-data path/to/model_corpus.txt \
                              -epochs 10 # Set total desired epochs
        ```
        *   Loads the model, optimizer state, and BPE tokenizer from the checkpoint.
        *   Continues training using `model_corpus.txt` from the saved epoch number up to the new `-epochs` value.
        *   Note: `-model-data` is still needed to prepare batches for the continuation. Configuration flags (like `-embedding-dim`, `-gru-layers`, etc.) from the *checkpoint* will generally override command-line flags for the model architecture.

    *   **Start Chat Interface using a Trained Model:**
        ```bash
        ./min-gru-linux-amd64 -checkpoint checkpoints/checkpoint_epoch_4.json
        ```
        *   Loads the specified checkpoint.
        *   Starts the interactive chat prompt.
        *   Type your message, press Enter. Type `exit` or `quit` to end.

## Architecture Details

*   **Tokenizer:** Byte Pair Encoding (BPE) splits words into common subword units.
*   **Embeddings:** Each token ID is mapped to a dense vector representation.
*   **GRU Layers:**
    *   Each layer contains multiple "expert" GRU networks.
    *   A gating network determines which experts to activate based on the input.
    *   The outputs of the experts are combined using weights from the gating network. This allows the model to increase capacity without proportionally increasing computation for every input.
    *   The GRU itself uses update and candidate gates to manage information flow over sequences. GELU activation is used internally.
*   **Residual Connections:** The input to a layer is added to its output (potentially after a projection if dimensions differ), helping with gradient flow and enabling deeper networks.
*   **RMSNorm:** Applied after the residual connection in each layer. It normalizes the activations based on the root mean square, providing stabilization during training with lower computational cost than standard Layer Normalization.
*   **Output Layer:** A final linear layer followed by Softmax converts the final hidden state into probabilities over the vocabulary for next-token prediction.
```mermaid
flowchart TD
    %% --- Style Definitions ---
    classDef Input    fill:#f9f,stroke:#333,stroke-width:2px;
    classDef Output   fill:#ccf,stroke:#333,stroke-width:2px;
    classDef Layer    fill:#e6ffe6,stroke:#66c266,stroke-width:1px;
    classDef Expert   fill:#fff0b3,stroke:#cca300,stroke-width:1px;
    classDef Gate     fill:#ffe6e6,stroke:#cc6666,stroke-width:1px;
    classDef Norm     fill:#e0ffff,stroke:#008080,stroke-width:1px;
    classDef Combine  fill:#f0f8ff,stroke:#4682b4,stroke-width:1px;

    %% --- Main Flow ---
    InputTokenID["Input Token ID (t)"]:::Input
    InputTokenID --> WE["Embedding Lookup (WE)"]
    WE --> X0["Embedding Vector x(t)"]

    %% External to Layer 1
    X0 --> |"Input to Layer 1"| X_in1
    H_prev1["Hidden h(t-1) from Layer 1"] --> |"Recurrence"| H_in1

    %% --- Layer 1 ---
    subgraph MoE_GRU_Layer1 [MoE-GRU Layer 1]
      direction TB
      X_in1["x(t)"]
      H_in1["h(t-1)"]

      subgraph Gating1 [Gating Network]
        direction TB
        X_in1 --> GLogits1["Wg*x + bg"]
        GLogits1 --> GWeights1["Softmax -> Gating Weights"]
      end
      class Gating1 Gate

      subgraph Experts1 
        direction TB
        Expert1_1["Expert 1 (GRU gates + GELU)"]
        ExpertN1["... Expert E ..."]
        X_in1 --> Expert1_1 & ExpertN1
        H_in1 --> Expert1_1 & ExpertN1
      end
      class Experts1 Expert

      Expert1_1 & ExpertN1 --> |"h_e(t)"| CombineExperts1
      GWeights1 --> |"weights"| CombineExperts1
      CombineExperts1["Combine Experts (Weighted Sum)"]:::Combine --> H_Combined1["h_comb(t)"]

      subgraph ResidualNorm1 [Residual & Norm]
        direction TB
        X_in1 --> ProjX1["Optional Projection (Wp)"]
        H_Combined1 --> Add1["+ Residual"]
        ProjX1 --> Add1
        Add1 --> RMSNorm1["RMSNorm"]
      end
      class ResidualNorm1 Norm

      RMSNorm1 --> X_out1["Layer 1 Output x(t)"]
    end
    class MoE_GRU_Layer1 Layer

    %% --- Recursion / Stacking ---
    X_out1 --> |"Feeds next"| X_inN
    X_prevN["x(t) from Layer N-1"] --> |"Input to Layer N"| X_inN
    H_prevN["h(t-1) from Layer N-1"] --> |"Recurrence"| H_inN

    %% --- Layer N (Simplified) ---
    subgraph MoE_GRU_LayerN [MoE-GRU Layer N]
      direction TB
      X_inN["x(t)"]
      H_inN["h(t-1)"]

      subgraph GatingN [Gating Network N]
        direction TB
        X_inN --> GLogitsN["Wg*x + bg"]
        GLogitsN --> GWeightsN["Softmax -> Gating Weights"]
      end
      class GatingN Gate

      subgraph ExpertsN [Experts N]
        direction TB
        ExpertsN_Out["Experts Output"]
        GWeightsN --> |"weights"| ExpertsN_Out
      end
      class ExpertsN Expert

      ExpertsN_Out --> AddN["+ Residual"]:::Norm
      X_inN --> AddN
      AddN --> RMSNormN["RMSNorm"]:::Norm
      RMSNormN --> X_outN["Final x(t)"]
    end
    class MoE_GRU_LayerN Layer

    %% --- Output ---
    X_outN --> OutputLayer["Output Projection (Whd, bd)"]
    OutputLayer --> Logits["Logits"]
    Logits --> Softmax["Softmax"]
    Softmax --> OutputProbs["P(token|context)"]:::Output
    OutputProbs --> SampledToken["Next Token ID"]:::Output

```


## Configuration Flags

*(Run `./min-gru-<os>-<arch> -h` to see all flags and default values)*

**Paths & Modes:**

*   `-checkpoint <path>`: Path to load/resume from a checkpoint file.
*   `-bpe-data <path>`: Path to data for BPE training.
*   `-model-data <path>`: Path to data for model training.
*   `-train`: Boolean flag to enable model training.

**Architecture Hyperparameters:**

*   `-bpe-vocab-size <int>`: Target BPE vocabulary size (default: 850).
*   `-embedding-dim <int>`: Dimension of token embeddings (default: 96).
*   `-gru-hidden-size <int>`: Hidden size for GRU layers (default: 96).
*   `-gru-layers <int>`: Number of GRU layers (default: 2).
*   `-num-experts <int>`: Number of experts per MoE layer (default: 6).

**Training Hyperparameters:**

*   `-seq-length <int>`: Sequence length for training steps (default: 80).
*   `-batch-size <int>`: Number of sequences per training batch (default: 16).
*   `-epochs <int>`: Total number of training epochs (default: 5).
*   `-lr <float>`: Learning rate for AdamW (default: 0.001).
*   `-wd <float>`: Weight decay for AdamW (default: 0.01).
*   `-grad-clip <float>`: Gradient clipping value (default: 5.0).
*   `-eps-rmsnorm <float>`: Epsilon for RMSNorm stability (default: 1e-05).
*   `-eps-adamw <float>`: Epsilon for AdamW stability (default: 1e-08).

**Inference:**

*   `-max-response <int>`: Maximum number of tokens to generate in chat mode (default: 260).
