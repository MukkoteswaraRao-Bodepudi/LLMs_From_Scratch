# LLMs From Scratch

A hands-on journey to understand and build Large Language Models from scratch.

This repository documents my learning and implementation journey, starting from the fundamentals of text processing and tokenization and gradually progressing toward building a Transformer-based language model.

The goal is to understand how LLMs work internally by implementing the major components step by step rather than treating them as a black box.

---

## 🎯 Project Goal

Build and understand a small Large Language Model from scratch.

The overall learning path is:

Raw Text → Tokenization → Vocabulary → Token IDs → Training Data → Embeddings → Self-Attention → Multi-Head Attention → Transformer Blocks → Language Model → Training → Next-Token Prediction → Text Generation

---

## 📚 Learning Progress

### Day 1 — Tokenization

Learned the fundamentals of tokenization and how raw text is converted into a representation that can be processed by a language model.

Topics covered:

- Tokens
- Tokenization
- Vocabulary
- Token IDs
- Token-to-ID mapping
- ID-to-token mapping
- Encoding
- Decoding
- Unknown tokens
- Special tokens
- `<|unk|>`
- `<|endoftext|>`
- Building a simple tokenizer
- Understanding the text-to-token pipeline

The tokenizer was implemented using Python and regular expressions to understand the basic mechanics of converting text into tokens and numerical IDs.

---

### Day 2 — Byte Pair Encoding (BPE)

Learned how Byte Pair Encoding is used for subword tokenization and explored GPT-style tokenization using `tiktoken`.

Topics covered:

- Byte Pair Encoding
- Subword tokenization
- Byte-level tokenization
- GPT-2 tokenizer
- Token IDs
- Encoding
- Decoding
- Token byte representations
- Leading-space tokens
- Multi-token words
- Token count analysis
- Encoding and decoding verification

An important concept learned during the experiments was:

**One word does not necessarily correspond to one token.**

For example, a word such as `unbelievable` can be represented using multiple subword tokens.

Related words such as:

- `play`
- `playing`
- `played`
- `player`
- `playground`
- `unplayable`

can be represented using combinations of common and different subword units.

---

## 🧪 Practical Experiments

The project includes practical experiments to reinforce the concepts learned during each stage.

### Tokenization

The tokenization experiments focused on:

- Processing raw text
- Splitting text into tokens
- Building a vocabulary
- Creating token-to-ID mappings
- Creating ID-to-token mappings
- Encoding text
- Decoding token IDs
- Handling unknown tokens
- Handling special tokens

### Byte Pair Encoding

The BPE experiments focused on:

- Using the GPT-2 tokenizer
- Encoding individual words
- Encoding complete sentences
- Inspecting token IDs
- Inspecting token byte representations
- Decoding token IDs
- Verifying decoded text
- Comparing token counts
- Identifying differences in tokenization between words and sentences

The experiments also verified that encoded token IDs can be decoded back into the original text.

---

## 🗂️ Project Structure

- `Data/`
  - `the-verdict.txt`

- `NoteBooks/`
  - `01_Tokenization.ipynb`
  - `02_byte_pair_encoding.ipynb`

- `Tasks/`
  - `Data/`
    - `llm_tokenization_data.txt`
  - `01_tokenization_task.ipynb`
  - `02_bpe_task.ipynb`

- `.gitignore`
- `.python-version`
- `main.py`
- `pyproject.toml`
- `README.md`
- `uv.lock`

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Regular Expressions
- tiktoken
- uv

---

## 📈 Current Progress

- ✅ LLM Fundamentals
- ✅ Tokenization
- ✅ Vocabulary
- ✅ Token IDs
- ✅ Encoding and Decoding
- ✅ Special Tokens
- ✅ Byte Pair Encoding
- ✅ GPT-2 Tokenizer Exploration
- ⬜ Embeddings
- ⬜ Positional Information
- ⬜ Self-Attention
- ⬜ Multi-Head Attention
- ⬜ Transformer Blocks
- ⬜ Language Model
- ⬜ Training
- ⬜ Next-Token Prediction
- ⬜ Text Generation

---

## 🧠 Key Concepts Learned

### Tokens

LLMs process text as tokens rather than directly processing raw human-readable text.

A token can represent:

- A complete word
- Part of a word
- Punctuation
- Whitespace-related text
- A subword or byte-level unit

Therefore:

**One word ≠ One token**

### Tokenization

Tokenization converts text into tokens and then represents those tokens using numerical IDs.

The simplified process is:

Text → Tokens → Token IDs

### Vocabulary

A vocabulary contains the tokens that a tokenizer knows and their corresponding numerical IDs.

The vocabulary provides the mapping between text tokens and numerical representations.

### Encoding and Decoding

Encoding converts text into token IDs.

Decoding converts token IDs back into text.

This creates the basic interface between human-readable text and the numerical representation used by language models.

### Byte Pair Encoding

BPE is a subword tokenization approach that allows words to be represented using smaller units.

This makes it possible to handle common words, rare words, and different word forms without requiring every complete word to exist as a separate vocabulary entry.

---


### Day 3 — Input-Target Pairs

Learned how tokenized text is converted into input-target pairs for language-model training.

The main objective is to prepare training examples so that a language model can learn **next-token prediction**.

Topics covered:

- Input sequences
- Target sequences
- Next-token prediction
- Context size
- Sliding-window approach
- Stride
- PyTorch Dataset
- PyTorch DataLoader
- Batching
- Tensor representation
- Training data preparation

The key relationship is:

**Target = Input shifted by one token**

For example:

Input:

`[5, 12, 25]`

Target:

`[12, 25, 41]`

The model uses the input sequence to learn which token should come next.

The input-target relationship can be represented as:

```text
Input:  I
Target: love

Input:  I love
Target: learning

Input:  I love learning
Target: LLMs 
```

# Day 4 Token Embeddings

## What Are Token Embeddings?

**Token embeddings convert token IDs into numerical vectors** that an LLM can process.

The basic flow is:

```text
Text
 ↓
Tokens
 ↓
Token IDs
 ↓
Token Embeddings
 ↓
Transformer
 ↓
Prediction
```

### Simple Example

Suppose:

```text
Token: "hello"
Token ID: 125
```

The model converts `125` into an embedding vector:

```text
125 → [0.21, -0.45, 0.78, 0.12, ...]
```

This vector is called the **token embedding**.

---

## Why Do We Need Token Embeddings?

A token ID is just a number.

```text
"hello" → 125
"world" → 892
```

The numbers `125` and `892` don't contain useful meaning by themselves.

So, the model converts them into vectors:

```text
125 → [0.21, -0.45, 0.78, ...]
892 → [0.15,  0.62, -0.31, ...]
```

These vectors are what the neural network works with.

---

## Embedding Matrix

All token embeddings are stored in an **embedding matrix**.

Its shape is:

```text
Vocabulary Size × Embedding Dimension
```

For example:

```text
Vocabulary Size = 5
Embedding Dimension = 4
```

The embedding matrix looks like:

```text
[
  [0.2,  0.5, -0.1, 0.7],   ← Token 0
  [0.4, -0.2,  0.8, 0.1],   ← Token 1
  [0.6,  0.3,  0.2, 0.9],   ← Token 2
  [0.1,  0.7, -0.4, 0.5],   ← Token 3
  [0.8, -0.1,  0.6, 0.2]    ← Token 4
]
```

Each **row** represents one token's embedding.

---

## Vocabulary Size

**Vocabulary size = Total number of tokens in the vocabulary.**

For example:

```text
Vocabulary Size = 50,000
```

means the model has 50,000 token entries.

Therefore, the embedding matrix has **50,000 rows**.

---

## Embedding Dimension

**Embedding dimension = Number of values in each token vector.**

For example:

```text
Embedding Dimension = 768
```

means each token is represented by a vector containing 768 numbers.

```text
Token → [x₁, x₂, x₃, ... x₇₆₈]
```

---

## How Does the Embedding Layer Work?

The embedding layer works like a **lookup table**.

```text
Token ID
   ↓
Find the corresponding row
   ↓
Return the embedding vector
```

For example:

```text
Token ID = 3

Embedding Matrix
       ↓
Row 3
       ↓
[0.1, 0.7, -0.4, 0.5]
```

So:

```text
Token ID → Embedding Vector
```

---

## Are Embeddings Trainable?

**Yes.**

Token embeddings are **trainable parameters** of the model.

During training:

```text
Input
  ↓
Token Embeddings
  ↓
Transformer
  ↓
Prediction
  ↓
Loss
  ↓
Update Parameters
```

The embedding values are gradually updated as the model learns.

---

## Token ID vs Token Embedding

| Token ID                      | Token Embedding                     |
| ----------------------------- | ----------------------------------- |
| A single number               | A vector of numbers                 |
| Identifies a token            | Represents a token numerically      |
| Example: `125`                | Example: `[0.21, -0.45, 0.78, ...]` |
| Used to look up the embedding | Used by the neural network          |

In short:

```text
Token
  ↓
Token ID
  ↓
Token Embedding
```

---

## Tensor Shape

Before embedding:

```text
Batch Size × Context Length
```

Example:

```text
4 × 16
```

After embedding:

```text
Batch Size × Context Length × Embedding Dimension
```

Example:

```text
4 × 16 × 768
```

This means:

* `4` → Number of sequences
* `16` → Number of tokens per sequence
* `768` → Numbers representing each token

---

## Easy Way to Remember

Think of token embeddings as a **lookup table**:

```text
Token ID
   ↓
┌─────────────────────┐
│ Embedding Matrix     │
│                     │
│ Row 0 → Vector      │
│ Row 1 → Vector      │
│ Row 2 → Vector      │
│ Row 3 → Vector      │
│ ...                 │
└─────────────────────┘
   ↓
Embedding Vector
   ↓
Transformer
```

### Key Takeaway

> **Token IDs identify tokens, while token embeddings represent those tokens as vectors that the LLM can process.**

```text
Text
 ↓
Tokenization
 ↓
Token IDs
 ↓
Token Embeddings
 ↓
Transformer
 ↓
Next-Token Prediction
```
# Day 5 Positional Embeddings

## What is Positional Embedding?

Positional embedding tells the model **where a token is located**.

```text
Token Embedding      → WHAT is the token?
Positional Embedding → WHERE is the token?
```

Example:

```text
I love Python

I       → Position 0
love    → Position 1
Python  → Position 2
```

---

## Input Batch

In this example:

```text
16 sequences
8 tokens per sequence
512 dimensions per token
```

So the token embeddings have:

```text
[16, 8, 512]
```

Meaning:

```text
16  → number of sequences
8   → tokens in each sequence
512 → dimensions for each token
```

---

## Positional Embeddings

There are 8 token positions:

```text
0  1  2  3  4  5  6  7
```

Each position has a 512-dimensional vector.

Therefore:

```text
[8, 512]
```

Meaning:

```text
8   → number of positions
512 → dimensions for each position
```

---

## Combining Them

```text
Token Embedding
[16, 8, 512]

        +

Positional Embedding
[8, 512]

        ↓

Input Embedding
[16, 8, 512]
```

The same 8 positional vectors are used for all 16 sequences.

```text
Sequence 1 → positions 0–7
Sequence 2 → positions 0–7
Sequence 3 → positions 0–7
...
Sequence 16 → positions 0–7
```

---

## Token ID vs Position ID

```text
Token ID     → identifies the token
Position ID  → identifies the token's position
```

Example:

```text
Python → Token ID = some number
Python → Position ID = 2
```

---

## Shape Summary

| Tensor              | Shape          | Meaning                 |
| ------------------- | -------------- | ----------------------- |
| Token IDs           | `[16, 8]`      | 16 sequences × 8 tokens |
| Token Embeddings    | `[16, 8, 512]` | 16 × 8 × 512            |
| Position Embeddings | `[8, 512]`     | 8 positions × 512       |
| Input Embeddings    | `[16, 8, 512]` | Token + Position        |

---

## Remember

```text
Token Embedding      = WHAT
Positional Embedding = WHERE

WHAT + WHERE
     ↓
Input Embedding
```

**`[16, 8, 512]` = 16 sequences × 8 tokens × 512 dimensions**

**`[8, 512]` = 8 positions × 512 dimensions**


## Day 6 Complete Data Preprocessing

The complete preprocessing pipeline covers:

1. Load Raw Text
2. Tokenization
3. Create Vocabulary
4. Convert Tokens → Token IDs
5. BPE Tokenization
6. Create Input-Target Pairs
7. Create Dataset
8. Create DataLoader
9. Create Batches
10. Token Embeddings
11. Positional Embeddings
12. Combine Token + Positional Embeddings
13. Final Input Embeddings

## Preprocessing Flow

```text
Raw Text
   ↓
Tokenization
   ↓
Token IDs
   ↓
Input-Target Pairs
   ↓
Dataset
   ↓
DataLoader
   ↓
Batches
   ↓
Token Embeddings
   ↓
Positional Embeddings
   ↓
Input Embeddings 
```
# Day 7 Simplified Attention Mechanism

This notebook demonstrates a **simplified attention mechanism without trainable weights** using PyTorch.

## What is Attention?

Attention helps a token determine **which other tokens are more relevant to it**.

The notebook uses the sentence:

> **Your Journey Starts with one step**

Each token is represented using a numerical vector.

## Steps

### 1. Input Vectors

Six tokens are represented as vectors:

```text
Your
Journey
Starts
with
one
step
```

Each token has a 3-dimensional vector.

### 2. Calculate Attention Scores

The **dot product** is used to measure how well two vectors are aligned.

For example, `Journey` is selected as the query and its dot product is calculated with every input vector.

```python
attention_scores[i] = torch.dot(x_i, query)
```

### 3. Normalize Attention Scores

The raw scores are converted into **attention weights**.

A simple normalization is:

```python
attention_weights = attention_scores / attention_scores.sum()
```

The notebook also demonstrates **Softmax**:

```python
attention_weights = torch.softmax(attention_scores, dim=0)
```

The attention weights sum to `1`.

### 4. Create the Context Vector

The attention weights are multiplied by their corresponding input vectors and then added together.

```python
context_vector += attention_weights[i] * x_i
```

This produces a **context vector** for the selected token.

### 5. Calculate Attention for All Tokens

Instead of calculating each dot product using loops:

```python
final_attn_scores = inputs @ inputs.T
```

This calculates the attention scores for all tokens at once.

### 6. Calculate Final Attention Weights

Softmax is applied across each row:

```python
final_attention_weights = torch.softmax(
    final_attention_scores,
    dim=1
)
```

Each row represents how one token attends to all the tokens.

### 7. Calculate Final Context Vectors

The final context vectors are calculated using matrix multiplication:

```python
final_context_vectors = final_attention_weights @ inputs
```

## Complete Flow

```text
Input Vectors
      ↓
Dot Product
      ↓
Attention Scores
      ↓
Softmax
      ↓
Attention Weights
      ↓
Weighted Sum
      ↓
Context Vectors
```

## Key Concepts

* Input vectors
* Query
* Dot product
* Attention scores
* Attention weights
* Softmax
* Context vector
* Matrix multiplication
* Attention without trainable weights

## Requirement

```bash
pip install torch
```

## Run

Open the notebook:

```text
07_simplified_attention(1).ipynb
```

and run the cells sequentially.

## Summary

This notebook builds attention step by step:

```text
Vectors
→ Similarity Scores
→ Attention Weights
→ Weighted Context
```

It provides a basic understanding of how attention works before moving to **trainable Query, Key, and Value matrices**.


# Day 8 Self-Attention with Trainable Weights

This notebook demonstrates the **Self-Attention mechanism with trainable weights** using PyTorch.

## Overview

Self-Attention allows each token in a sequence to understand its relationship with other tokens in the same sequence.

The notebook uses the sentence:

**Your Journey Starts with one step**

Each token is represented using an input vector.

## Main Steps

1. **Input Embeddings**
   Represent each token as a numerical vector.

2. **Query, Key, and Value**
   Transform the input vectors into Query, Key, and Value representations using trainable weight matrices.

3. **Attention Scores**
   Calculate the similarity between Query and Key vectors using the dot product.

4. **Scaling**
   Scale the attention scores using the square root of the key dimension to maintain stable values.

5. **Softmax**
   Convert the attention scores into attention weights.

6. **Context Vectors**
   Use the attention weights to calculate weighted combinations of the Value vectors.

## Attention Flow

**Input Embeddings → Query, Key, Value → Attention Scores → Scaling → Softmax → Attention Weights → Context Vectors**

## Key Concepts

* Self-Attention
* Query, Key, and Value
* Trainable Weights
* Attention Scores
* Softmax
* Attention Weights
* Context Vectors
* Scaled Dot-Product Attention

## Requirements

* Python
* PyTorch
* Jupyter Notebook

## Purpose

The purpose of this notebook is to understand the basic working of **Self-Attention with trainable Query, Key, and Value weights**, which is an important component of Transformer models.

# Day 9 Causal Self-Attention

This notebook demonstrates the **Causal Self-Attention mechanism** using PyTorch.

## Overview

Causal Self-Attention is a type of attention mechanism where each token can attend to the **current token and previous tokens**, but not to future tokens.

This prevents future information from being used when processing the current token.

## Main Steps

1. **Input Embeddings**
   The input sentence is represented as numerical vectors.

2. **Query, Key, and Value**
   Trainable weights are used to create Query, Key, and Value representations.

3. **Attention Scores**
   Attention scores are calculated between Query and Key vectors.

4. **Causal Masking**
   Future tokens are masked so that the current token cannot access information from future positions.

5. **Softmax**
   The masked attention scores are converted into attention weights.

6. **Dropout**
   Dropout is applied to the attention weights to reduce overfitting.

7. **Context Vectors**
   The attention weights are combined with the Value vectors to produce context vectors.

## Causal Masking

The attention matrix is masked in the upper triangular region.

This ensures:

* The first token can attend only to itself.
* The second token can attend to the first and second tokens.
* The third token can attend to the first, second, and third tokens.
* Future tokens are not accessible.

## Attention Flow

**Input Embeddings → Query, Key, Value → Attention Scores → Causal Masking → Softmax → Dropout → Context Vectors**

## Batch Processing

The notebook also demonstrates how to process multiple input sequences together using a batch.

The input is represented as a 3-dimensional tensor containing:

* Multiple input sequences
* Multiple tokens
* Vector representation of each token

## Key Concepts

* Causal Self-Attention
* Query, Key, and Value
* Attention Scores
* Causal Masking
* Softmax
* Dropout
* Context Vectors
* Batch Processing
* PyTorch

## Requirements

* Python
* PyTorch
* Jupyter Notebook

## Purpose

The purpose of this notebook is to understand how **Causal Self-Attention prevents future information leakage** and produces context vectors while processing a sequence.

# Day 10 Simple Multi-Head Attention

## Overview

Multi-Head Attention uses multiple attention heads instead of a single attention mechanism.

Each head performs its own attention operation and can learn different relationships between tokens.

In this task, we combine **Multi-Head Attention** with **Causal Self-Attention**.

## Why Multi-Head Attention?

A single attention head may focus on one type of relationship between tokens.

Multiple heads allow the model to learn different relationships at the same time.

For example:

Input
↓
Attention Head 1
Attention Head 2
↓
Concatenate
↓
Combined Output

Each attention head has its own Query, Key, and Value projections.

## Causal Attention

Causal attention prevents a token from attending to future tokens.

A token can attend to:

- Previous tokens
- The current token

A token cannot attend to:

- Future tokens

This is important for GPT-style language models because they predict the next token using only the information available so far.

## Multi-Head Causal Attention Flow

Input Embeddings
↓
Create Query, Key, and Value
↓
Split into Multiple Heads
↓
Calculate Attention Scores
↓
Apply Causal Mask
↓
Scale Attention Scores
↓
Apply Softmax
↓
Calculate Context Vectors
↓
Concatenate Head Outputs
↓
Final Multi-Head Attention Output

## Attention Head

Each attention head performs attention independently.

For example, with two heads:

Head 1
→ Query, Key, Value
→ Causal Attention
→ Context Vector

Head 2
→ Query, Key, Value
→ Causal Attention
→ Context Vector

The outputs from both heads are then concatenated.

## Main Steps

1. Create input embeddings.
2. Define multiple attention heads.
3. Create Query, Key, and Value projections.
4. Calculate attention scores.
5. Apply the causal mask.
6. Scale the attention scores.
7. Apply Softmax.
8. Calculate the context vector for each head.
9. Concatenate the outputs from all heads.
10. Produce the final multi-head attention output.

## Key Concepts

### Attention Head

An independent attention mechanism that learns relationships between tokens.

### Query

Represents what information a token is looking for.

### Key

Represents information that can be matched against a Query.

### Value

Contains the information used to create the final context representation.

### Causal Mask

Prevents tokens from attending to future tokens.

### Attention Scores

Measure the similarity between Queries and Keys.

### Attention Weights

Softmax converts attention scores into weights that determine how much attention each token receives.

### Context Vector

A weighted combination of Value vectors.

### Concatenation

Combines the outputs from all attention heads into a single representation.

## Simple Example

For the sentence:

The student reads a book.

When processing:

student

the model can attend to:

The
student

but it cannot attend to:

reads
a
book

This restriction is applied independently in every attention head.

## Summary

Multi-Head Attention allows the model to learn different relationships between tokens using multiple attention heads.

Causal masking ensures that each token can only use information from previous and current tokens.

Therefore:

Multi-Head Attention
+
Causal Attention
=
Simple Multi-Head Causal Attention



### Current Focus

Building a strong understanding of the fundamental components required to construct an LLM from scratch.

---

## 🎯 Long-Term Objective

The long-term objective of this repository is to build a small Transformer-based Large Language Model from scratch.

The project will progressively cover:

- Text processing
- Tokenization
- Vocabulary
- Token IDs
- Training data preparation
- Embeddings
- Positional information
- Self-attention
- Multi-head attention
- Feed-forward networks
- Transformer blocks
- Language-model architecture
- Loss calculation
- Training
- Next-token prediction
- Text generation

The primary objective is to understand **why each component is required, how it works, and how the components work together to form an LLM**.

---

## 📖 Learning Approach

Each stage of the project follows a practical learning process:

1. Understand the concept
2. Study the underlying idea
3. Implement the concept
4. Experiment with examples
5. Create practical tasks
6. Analyze the results
7. Document the learning

The goal is not simply to use existing LLMs, but to understand the fundamental mechanisms behind them.

---

## 👨‍💻 Author

**Mukkoteswara Rao Bodepudi**

Learning and building Large Language Models from the fundamentals