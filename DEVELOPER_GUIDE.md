# Quantum Transformer: Developer & Researcher Guide

This document provides a detailed technical overview of the `quantum-transformer` architecture, its core classes, and the underlying quantum-inspired logic.

---

## 1. Core Philosophy

Quantum Transformer (QT) is designed to explore the intersection of quantum gate operations and autoregressive sequence modeling. Unlike standard transformers that rely on dot-product attention, QT utilizes simulated quantum transformations to model token dependencies.

## 2. Model Architecture

### `QuantumTransformerLM`
The top-level class for the language model.

**Constructor Parameters:**
- `config`: An instance of `QuantumTransformerConfig` containing hyperparameters.

**Key Methods:**
- `forward(input_ids, targets=None, past_key_values=None)`: Performs a forward pass. Supports KV caching for optimized inference.
- `generate(input_ids, max_new_tokens, ...)`: High-level generation method implementing top-k, top-p, temperature, and repetition penalty.
- `from_pretrained(path_or_repo_id)`: Static method to load models from a local path or Hugging Face Hub.

---

## 3. Quantum-Inspired Layers

### `QuantumTransformerBlock`
A single transformer layer containing a Quantum Attention mechanism followed by a Quantum Feed-Forward Network (QFFN).

### `QuantumAttention`
The heart of the architecture. It simulates quantum state evolution using:
1. **Rotation Gates**: Applied to input embeddings to map them into a Hilbert-like space.
2. **Entanglement Simulation**: Controlled transformations between tokens to simulate inter-token dependency.
3. **Measurement**: Projecting the quantum-like state back into a classical hidden dimension.

### `QuantumFFN`
A feed-forward network that uses rotation-based non-linearities instead of standard ReLU/GELU, mirroring quantum phase shifts.

---

## 4. Tokenization

### `BPETokenizer`
A Byte-Pair Encoding tokenizer built on top of the Hugging Face `tokenizers` library.

**Features:**
- **Byte-level BPE**: Prevents out-of-vocabulary (OOV) issues by falling back to byte representations.
- **Pre-trained Support**: Can load specific tokenizer configurations from model directories or remote repositories.

---

## 5. Inference Optimizations

### Key-Value (KV) Caching
The model implements a professional KV cache system. During generation, only the last token is processed through the transformer blocks, with previous keys and values retrieved from the cache. 

**Note on Absolute Positional Embeddings**: The implementation includes logic to cleanly recompute positional embeddings when the context window shifts, preventing positional degradation.

### Structural Confidence Guard
The generation loop includes a Shannon Entropy calculation:
$$ H(P) = -\sum p_i \log p_i $$
This metric is used to determine the "certainty" of the model at each step, allowing for dynamic pruning of low-confidence branches.

---

## 6. Training Pipeline

Training is handled via the `train.py` script, which supports:
- **Mixed Precision**: Utilizing `torch.amp` for faster training on modern GPUs.
- **Checkpointing**: Automatic saving of model weights, config, and tokenizer state.
- **Progress Tracking**: Real-time loss monitoring and epoch-level evaluation.

---

## 7. Configuration Example

```python
from quantum_transformer.config import QuantumTransformerConfig

config = QuantumTransformerConfig(
    num_qubits=4,
    hidden_dim=128,
    num_layers=4,
    num_heads=4,
    max_context_length=256,
    vocab_size=8192,
    circuit_depth=3
)
```

## 8. Research Directions

- **Qubit Scaling**: Testing the effect of increasing simulated qubits on model perplexity.
- **Hybrid Layers**: Mixing classical Self-Attention with Quantum Attention blocks.
- **Entanglement Patterns**: Implementing custom entanglement topologies (e.g., all-to-all vs. linear).

---
*For further technical queries, contact the Quantum AI team at reach.harishapc@gmail.com.*
