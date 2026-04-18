import numpy as np
import json
from utils import load_data, save_numpy, load_numpy, get_top_vocab, create_word_to_idx


def load_metadata(filepath):
    """Load metadata JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"Loaded metadata for {len(metadata)} documents")
    return metadata


def build_term_document_matrix(documents, vocab):
    """Build term-document matrix by counting word occurrences"""
    print("Building term-document matrix...")
    
    num_docs = len(documents)
    vocab_size = len(vocab)
    
    # Create word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Initialize matrix
    term_doc_matrix = np.zeros((num_docs, vocab_size))
    
    # Count word occurrences in each document
    for doc_idx, doc in enumerate(documents):
        words = doc.split()
        for word in words:
            if word in word_to_idx:
                word_idx = word_to_idx[word]
                term_doc_matrix[doc_idx, word_idx] += 1
    
    print("Done.")
    return term_doc_matrix


def compute_tfidf(term_doc_matrix, vocab):
    """Compute TF-IDF from term-document matrix"""
    print("Computing TF-IDF...")
    
    num_docs = term_doc_matrix.shape[0]
    vocab_size = term_doc_matrix.shape[1]
    
    # Initialize TF-IDF matrix
    tfidf_matrix = np.zeros((num_docs, vocab_size))
    
    # Compute document frequencies for each word
    # df = number of documents containing the word
    doc_freq = np.zeros(vocab_size)
    for word_idx in range(vocab_size):
        doc_freq[word_idx] = np.sum(term_doc_matrix[:, word_idx] > 0)
    
    # Compute IDF for each word
    # idf = log(N / (1 + df))
    idf_values = np.log(num_docs / (1 + doc_freq))
    
    # Compute TF-IDF for each document
    for doc_idx in range(num_docs):
        # Get word counts for this document
        word_counts = term_doc_matrix[doc_idx, :]
        
        # Compute document length
        doc_length = np.sum(word_counts)
        
        if doc_length > 0:
            # Compute TF: word_count / doc_length
            tf_values = word_counts / doc_length
            
            # Multiply TF × IDF
            tfidf_matrix[doc_idx, :] = tf_values * idf_values
    
    print("Done.")
    return tfidf_matrix


def get_top_words_per_category(tfidf_matrix, vocab, documents, metadata, top_n=10):
    """Get top N words for each category based on TF-IDF scores"""
    print("Finding top words per category...")
    
    # Group documents by category
    category_docs = {}
    for idx, (doc_id, meta) in enumerate(metadata.items()):
        category = meta.get('category', 'unknown')
        if category not in category_docs:
            category_docs[category] = []
        category_docs[category].append(idx)
    
    # For each category, find top words
    top_words = {}
    for category, doc_indices in category_docs.items():
        # Sum TF-IDF scores across all documents in this category
        category_scores = np.zeros(len(vocab))
        
        for doc_idx in doc_indices:
            category_scores += tfidf_matrix[doc_idx, :]
        
        # Average the scores
        category_scores = category_scores / len(doc_indices)
        
        # Get top N words
        top_indices = np.argsort(category_scores)[::-1][:top_n]
        top_words[category] = [(vocab[idx], category_scores[idx]) for idx in top_indices]
    
    print("Done.")
    return top_words


def print_top_words(top_words):
    """Print top words for each category"""
    print("\n" + "="*50)
    print("Top Words per Category (TF-IDF)")
    print("="*50)
    
    for category, words in top_words.items():
        print(f"\n{category}:")
        for word, score in words:
            print(f"  {word}: {score:.4f}")


if __name__ == "__main__":
    # Load data
    documents = load_data("data/cleaned.txt")
    metadata = load_metadata("data/metadata.json")
    vocab = get_top_vocab(documents, vocab_size=10000)
    
    # Build and compute
    term_doc_matrix = build_term_document_matrix(documents, vocab)
    tfidf_matrix = compute_tfidf(term_doc_matrix, vocab)
    
    # Save
    save_numpy(tfidf_matrix, "embeddings/tfidf_matrix.npy")
    print("TF-IDF matrix saved.")
    
    # Get top words
    top_words = get_top_words_per_category(tfidf_matrix, vocab, documents, metadata)
    print_top_words(top_words)
