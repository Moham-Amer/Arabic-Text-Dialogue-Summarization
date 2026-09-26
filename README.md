# Arabic Dialogue Summarization: Fine-Tuning AraT5, mT5, and mBART

An end-to-end Natural Language Processing (NLP) project investigating abstractive summarization of Arabic conversational dialogue. This repository implements text normalization pipelines and fine-tunes three seq2seq transformer architectures (**AraT5v2**, **mT5-Small**, and **Tiny-mBART**) evaluated against **ROUGE** metrics.

- **Author**: Mohammad Amer Khalil (Student ID: 202110942)
- **Institution**: Arab International University (AIU) — Faculty of Informatics Engineering

---

## Architecture & Workflow

[Arabic SAMSum Dialogue] ──> [Normalization & Cleaning] ──> [Stemming vs Lemmatization]
                                                             (Snowball vs Qalsadi)
                                                                       │
                                                                       ▼
[ROUGE-1 / ROUGE-2 / ROUGE-L] <── [Hugging Face Trainer] <── [Seq2Seq Tokenization]
 (evaluate & rouge-score)           (AraT5 / mT5 / mBART)       (DataCollatorForSeq2Seq)

---

## Key Components

### 1. Preprocessing & Morphological Experiments
- **Cleaning & Normalization**: Stripped diacritics (`[\u064B-\u065F]`), cleaned URLs, tags, and non-Arabic noise while preserving conversational structure.
- **Morphology Comparison**: Evaluated the downstream impact of root stemming (`nltk.stem.SnowballStemmer`) versus lemmatization (`qalsadi.lemmatizer`).
- **Tokenization**: Analyzed sentence segmentation and subword tokenization across model-specific tokenizers.

### 2. Transformer Models Fine-Tuned
Using Hugging Face `Trainer` and `DataCollatorForSeq2Seq`, three sequence-to-sequence architectures were fine-tuned and compared:
- **`UBC-NLP/AraT5v2-base-1024`**: Dedicated Arabic-pretrained T5 model. Evaluated across both raw input and stemmed variants.
- **`google/mt5-small`**: Multilingual T5 model fine-tuned on the processed Arabic conversational corpus.
- **`Tiny-mBART`**: Compact sequence-to-sequence architecture evaluated for lower-latency inference.

### 3. Evaluation Metrics
Evaluated using the Hugging Face `evaluate` library:
- **ROUGE-1**: Unigram overlap between generated summary and gold reference.
- **ROUGE-2**: Bigram overlap measuring fluency and phrase capture.
- **ROUGE-L**: Longest Common Subsequence (LCS) scoring structural coherence.

---

## File Structure

├── Final_NLPL_Notebook.ipynb    # Full 206-cell notebook: preprocessing, training loops, ROUGE evals
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

---

## Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/Moham-Amer/Arabic-Dialogue-Summarization-Seq2Seq.git
cd Arabic-Dialogue-Summarization-Seq2Seq

# Install requirements
pip install torch transformers datasets evaluate rouge-score nltk qalsadi

Run in Jupyter Notebook or Google Colab:
jupyter notebook Final_NLPL_Notebook.ipynb

---
