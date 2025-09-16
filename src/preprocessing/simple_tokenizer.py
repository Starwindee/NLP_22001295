import re
from typing import List
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()
