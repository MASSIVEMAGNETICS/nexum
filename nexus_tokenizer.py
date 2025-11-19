"""
Tokenizer - Character-level tokenization with vocabulary
Replacement for ord() % 256 anti-pattern from problem statement.
"""

import numpy as np
from typing import List, Dict, Optional
import json
import os


class CharacterTokenizer:
    """
    Character-level tokenizer with fixed vocabulary.
    Simple but effective replacement for ord() modulo patterns.
    """
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<START>": 2,
            "<END>": 3
        }
        self.next_id = len(self.special_tokens)
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.char_to_id[token] = idx
            self.id_to_char[idx] = token
    
    def train(self, texts: List[str]):
        """
        Build vocabulary from training texts.
        """
        # Collect all unique characters
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)
        
        # Add to vocabulary
        for char in sorted(unique_chars):
            if char not in self.char_to_id and self.next_id < self.vocab_size:
                self.char_to_id[char] = self.next_id
                self.id_to_char[self.next_id] = char
                self.next_id += 1
        
        print(f"[TOKENIZER] Vocabulary built: {len(self.char_to_id)} tokens")
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """
        Convert text to list of token IDs.
        """
        tokens = []
        for char in text:
            token_id = self.char_to_id.get(char, self.special_tokens["<UNK>"])
            tokens.append(token_id)
        
        # Apply max length if specified
        if max_length is not None:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            elif len(tokens) < max_length:
                # Pad with <PAD> token
                tokens.extend([self.special_tokens["<PAD>"]] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        chars = []
        for token_id in token_ids:
            # Skip padding
            if token_id == self.special_tokens["<PAD>"]:
                continue
            char = self.id_to_char.get(token_id, "<UNK>")
            chars.append(char)
        
        return "".join(chars)
    
    def tokens_to_embedding(self, token_ids: List[int], embed_dim: int = 512) -> np.ndarray:
        """
        Convert token IDs to embeddings.
        Simple lookup table approach.
        """
        # Initialize embedding matrix if not exists
        if not hasattr(self, 'embedding_matrix'):
            self.embedding_matrix = np.random.randn(self.vocab_size, embed_dim) * 0.01
        
        # Look up embeddings
        embeddings = []
        for token_id in token_ids:
            if token_id < self.vocab_size:
                embeddings.append(self.embedding_matrix[token_id])
            else:
                embeddings.append(np.zeros(embed_dim))
        
        return np.array(embeddings)
    
    def save(self, filepath: str):
        """Save tokenizer state."""
        data = {
            "vocab_size": self.vocab_size,
            "char_to_id": self.char_to_id,
            "id_to_char": {int(k): v for k, v in self.id_to_char.items()},
            "special_tokens": self.special_tokens,
            "next_id": self.next_id
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str) -> bool:
        """Load tokenizer state."""
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.vocab_size = data["vocab_size"]
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        self.special_tokens = data["special_tokens"]
        self.next_id = data["next_id"]
        
        return True


def train_tokenizer(texts: List[str], vocab_size: int = 256) -> CharacterTokenizer:
    """
    Convenience function to create and train a tokenizer.
    """
    tokenizer = CharacterTokenizer(vocab_size=vocab_size)
    tokenizer.train(texts)
    return tokenizer
