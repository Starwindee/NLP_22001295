# src/main.py
from preprocessing.simple_tokenizer import SimpleTokenizer
from preprocessing.regex_tokenizer import SimpleTokenizer as RegexTokenizer
from core.interfaces import Tokenizer

if __name__ == "__main__":
    text = "Hello, World! This is a test."
    tokenizer1: Tokenizer = SimpleTokenizer()
    tokens1 = tokenizer1.tokenize(text)
    print("Simple Tokenizer:", tokens1)

    tokenizer2: Tokenizer = RegexTokenizer()
    tokens2 = tokenizer2.tokenize(text)
    print("Regex Tokenizer:", tokens2)
    
