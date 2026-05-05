# Quantum Transformer

Quantum Transformer is a specialized framework designed for the development and deployment of quantum-inspired artificial intelligence models. It extends the standard Transformer architecture by integrating quantum gate simulations as a mechanism for attention and feature transformation, providing a novel approach to sequence modeling and representation learning.

## 🎥 Demo Video

[![Watch the video](https://img.youtube.com/vi/btVQvQEKVWM/maxresdefault.jpg)](https://youtu.be/btVQvQEKVWM)
## Overview

The core objective of the Quantum Transformer project is to provide researchers and developers with a robust toolset for building pre-trained Quantum AI models. By mapping quantum circuit operations (such as rotation gates and entanglement simulations) onto neural network layers, the framework achieves high-fidelity signal processing that mirrors quantum state evolution.

## Technical Architecture

The framework is built upon several key technical pillars:

### Quantum-Inspired Attention (QIA)
Unlike traditional scaled dot-product attention, QIA utilizes simulated quantum gates to calculate weight distributions. This allows the model to capture complex inter-token dependencies within a high-dimensional quantum-like state space.

### Structural Confidence Guard
The framework implements a Shannon Entropy-based monitoring system that evaluates model certainty during autoregressive generation. This provides a mathematically grounded metric for output reliability.

### Key-Value (KV) Caching
Optimized inference pathways are implemented using a sliding-window KV cache system, ensuring efficient memory utilization even with absolute positional embeddings.

## Installation

The package can be installed via pip:

```bash
pip install quantum-transformer
```

## Core Components

- **QuantumTransformerLM**: The primary language model class, supporting both training and inference.
- **BPETokenizer**: A professional-grade Byte-Pair Encoding tokenizer for sub-word processing.
- **QuantumTransformerConfig**: A comprehensive configuration system for defining model hyperparameters and quantum circuit depths.

## Usage Example

The following example demonstrates how to load a pre-trained model and perform inference:

```python
from quantum_transformer.model import QuantumTransformerLM

# Load a pre-trained model from the Hugging Face Hub
model = QuantumTransformerLM.from_pretrained("Harishapc01/quantum-qt-4q")

# Execute autoregressive generation
output = model.generate(
    prompt="The integration of quantum mechanics with neural networks",
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9
)

print(output)
```

## Research and Development

Quantum Transformer is intended for:
- Investigating the scalability of quantum-inspired layers in large-scale language models.
- Developing hybrid classical-quantum architectures for natural language processing.
- Benchmarking the efficiency of quantum gate simulations in high-dimensional feature spaces.

## License

This project is licensed under the MIT License.

## Contact


Quantum AI Team: Harisha P C
Email: reach.harishapc@gmail.com
website: www.harishapc.com
