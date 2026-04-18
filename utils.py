import numpy as np
from collections import Counter
import json
import os


def load_data(filepath):
    """Read cleaned.txt line by line and return list of documents"""
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        return []
    
    documents = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                documents.append(line)
    
    print(f"Loaded {len(documents)} documents from {filepath}")
    return documents


def save_numpy(array, filename):
    """Save numpy array to .npy file"""
    np.save(filename, array)
    print(f"Saved array to {filename}")


def load_numpy(filename):
    """Load numpy array from .npy file"""
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        return None
    
    array = np.load(filename)
    print(f"Loaded array from {filename}")
    return array


def get_top_vocab(documents, vocab_size=10000):
    """Count word frequencies and return top vocab_size words"""
    # Count word frequencies
    word_counts = Counter()
    
    for doc in documents:
        words = doc.split()
        word_counts.update(words)
    
    # Get top words
    top_words = [word for word, count in word_counts.most_common(vocab_size)]
    
    print(f"Created vocabulary of {len(top_words)} words")
    return top_words


def create_word_to_idx(vocab):
    """Create word to index and index to word mappings"""
    # Simple dict mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    print(f"Created mappings for {len(vocab)} words")
    return word_to_idx, idx_to_word


if __name__ == "__main__":
    pass
