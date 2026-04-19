"""
Part 3: Simplified Transformer for Topic Classification
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import math
import os


class SimpleTransformer(nn.Module):
    """Simplified transformer using PyTorch's built-in components where allowed"""
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4, dim_feedforward=512, num_classes=5, max_len=256, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        pe = torch.zeros(max_len + 1, d_model)
        position = torch.arange(0, max_len + 1).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder layers (manual implementation)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Embeddings
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        
        # Create attention mask
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            # Convert to attention mask format (True = ignore)
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Use CLS token
        cls_output = x[:, 0]
        logits = self.classifier(cls_output)
        
        return logits


class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=256):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        words = text.split()[:self.max_len]
        indices = [self.word2idx.get(w, self.word2idx.get('<UNK>', 0)) for w in words]
        
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return torch.tensor(indices), torch.tensor(label)


def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = torch.stack(texts)
    labels = torch.stack(labels)
    mask = (texts != 0).float()
    return texts, labels, mask


def train():
    print("="*60)
    print("Part 3: Transformer Classifier")
    print("="*60)
    
    # Load data
    print("\nLoading...")
    with open("data/cleaned.txt", 'r', encoding='utf-8') as f:
        documents = [line.strip() for line in f if line.strip()]
    
    with open("data/metadata.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    with open("embeddings/word2idx.json", 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    
    # Assign categories based on keywords in the text
    categories = []
    for doc in documents:
        doc_lower = doc.lower()
        
        # Simple keyword-based classification
        if any(word in doc_lower for word in ['کرکٹ', 'میچ', 'ٹیم', 'کھلاڑی', 'فٹبال']):
            categories.append(1)  # Sports
        elif any(word in doc_lower for word in ['معیشت', 'تجارت', 'بینک', 'روپے', 'ڈالر']):
            categories.append(2)  # Economy
        elif any(word in doc_lower for word in ['امریکا', 'چین', 'برطانیہ', 'بین', 'الاقوامی']):
            categories.append(3)  # International
        elif any(word in doc_lower for word in ['صحت', 'ہسپتال', 'بیماری', 'تعلیم', 'سیلاب']):
            categories.append(4)  # Health
        else:
            categories.append(0)  # Politics (default)
    
    # Print distribution
    from collections import Counter
    dist = Counter(categories)
    print(f"Category distribution: {dict(dist)}")
    
    # Split
    n = len(documents)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.15)
    
    train_ds = TextDataset(documents[:train_end], categories[:train_end], word2idx)
    val_ds = TextDataset(documents[train_end:val_end], categories[train_end:val_end], word2idx)
    test_ds = TextDataset(documents[val_end:], categories[val_end:], word2idx)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Model (smaller with more regularization)
    model = SimpleTransformer(vocab_size=len(word2idx), d_model=64, nhead=4, num_layers=2, dim_feedforward=256, num_classes=5, dropout=0.3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)  # Lower LR, higher weight decay
    criterion = nn.CrossEntropyLoss()
    
    # Train
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    print("Training...")
    for epoch in range(5):  # 5 epochs for faster training
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch_idx, (texts, labels, mask) in enumerate(train_loader):
            texts, labels, mask = texts.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            logits = model(texts, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {total_loss/(batch_idx+1):.4f}")
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Val
        model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for texts, labels, mask in val_loader:
                texts, labels, mask = texts.to(device), labels.to(device), mask.to(device)
                logits = model(texts, mask)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        
        val_loss = total_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1:2d} | Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f}")
    
    # Save
    torch.save(model.state_dict(), "models/transformer_cls.pt")
    print("\nSaved: models/transformer_cls.pt")
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("models/transformer_curves.png")
    print("Saved: models/transformer_curves.png")
    
    # Test
    print("\nTest:")
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for texts, labels, mask in test_loader:
            texts, labels, mask = texts.to(device), labels.to(device), mask.to(device)
            logits = model(texts, mask)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Politics', 'Sports', 'Economy', 'Intl', 'Health'],
                yticklabels=['Politics', 'Sports', 'Economy', 'Intl', 'Health'])
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig("models/transformer_confusion.png")
    print("Saved: models/transformer_confusion.png")
    
    print("\nDone!")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train()
