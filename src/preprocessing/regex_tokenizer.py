import re
from typing import List
from core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def __init__(self, partten: str = r'\w+|[^\w\s]'):
        super().__init__()
        self.partten = partten

    def tokenize(self, text: str) -> List[str]:
        return re.findall(self.partten, text.lower())
