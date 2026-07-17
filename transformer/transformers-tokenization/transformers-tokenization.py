import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
    
    def build_vocab(self, texts: List[str]) -> None:
        self.word_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }

        self.id_to_word = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<BOS>",
            3: "<EOS>"
        }

        self.vocab_size = 4
        
        words = []

        for text in texts:
            words.extend(text.lower().split())
        words.sort()
        for word in words:
            if word not in self.word_to_id:
                idx = self.vocab_size
        
                self.word_to_id[word] = idx
                self.id_to_word[idx] = word
        
                self.vocab_size += 1
    
    def encode(self, text: str) -> List[int]:
        text = text.lower()
        list_words = text.split()

        result = []
        for word in list_words:
            if word not in self.word_to_id:
                result.append(self.word_to_id["<UNK>"])
            else: 
                result.append(self.word_to_id[word])
        return result
    
    def decode(self, ids: List[int]) -> str:
        words = []
        for id in ids:
            if id not in self.id_to_word:
                words.append(self.id_to_word[1])
            else:
                words.append(self.id_to_word[id])
        return " ".join(words)
