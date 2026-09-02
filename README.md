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

## 🚧 Project Status

**In Progress**

### Completed

- LLM fundamentals
- Tokenization
- Vocabulary
- Token IDs
- Encoding and decoding
- Special tokens
- Byte Pair Encoding
- GPT-2 tokenizer exploration
- Tokenization practical tasks
- BPE practical tasks

### Current Focus

Building a strong understanding of the fundamental components required to construct an LLM from scratch.

---

## 👨‍💻 Author

**Mukkoteswara Rao Bodepudi**

Learning and building Large Language Models from the fundamentals.