# CS-4063 Assignment 2: BBC Urdu NLP Pipeline

Implementation of TF-IDF, PPMI, Word2Vec, BiLSTM, and Transformer models from scratch for BBC Urdu text processing.

## Overview

This project implements three core NLP components:

- Word embedding techniques (TF-IDF, PPMI, Word2Vec)
- Sequence labeling with BiLSTM
- Text classification using Transformer architecture



## Part 1: Embeddings

Implementation of three embedding approaches for Urdu text representation.

### How to run

```bash
python embeddings.py
```

## Part 2: Sequence Labeling

BiLSTM model for sequence labeling tasks on BBC Urdu dataset.

### How to run

```bash
python bi-LSTM.py
```

## Part 3: Transformer Classification

Transformer-based text classification for BBC Urdu articles.

### How to run

```bash
python transformer.py
```

## Jupyter Notebooks

A comprehensive Jupyter notebook (`i23-2607_Assignment2_DS_A.ipynb`) is provided that includes:
- Complete implementation of all assignment parts
- Data exploration and visualization
- Model training and evaluation
- Results and analysis

To run the notebook:

```bash
jupyter notebook i23-2607_Assignment2_DS_A.ipynb
```

## Project Structure

```
.
├── data/                               # Dataset files
├── embeddings/                         # Saved embedding models
├── models/                             # Trained model checkpoints
├── i23-2607_Assignment2_DS_A.ipynb    # Complete assignment notebook
├── embeddings.py
├── bilstm.py
├── transformer.py
└── utils.py                            # Helper functions
```

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib

## Author

CS-4063 NLP Course Assignment
