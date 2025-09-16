import re
from typing import List
from src.core.interfaces import Tokenizer


class RegexTokenizer(Tokenizer):
    def __init__(self, partten: str = r'\w+|[^\w\s]'):
        super().__init__()
        self.partten = partten

    def tokenize(self, text: str) -> List[str]:
        return re.findall(self.partten, text.lower())
