
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.interfaces import Tokenizer
from src.representations.count_vectorizer import CountVectorizer
from src.core.dataset_loaders import load_raw_text_data

def test_tokenizers():
    text = "Hello, World! This is a test."
    print("Original Text:", text)

    tokenizer1: Tokenizer = SimpleTokenizer()
    tokens1 = tokenizer1.tokenize(text)
    print("Simple Tokenizer:", tokens1)

    tokenizer2: Tokenizer = RegexTokenizer()
    tokens2 = tokenizer2.tokenize(text)
    print("Regex Tokenizer:", tokens2)

def main():
    # 1. Load dataset
    dataset_path = r"E:\Năm 4\NLP\level 1\UD_English-EWT\en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)

    # 2. Lấy mẫu dữ liệu nhỏ để demo
    sample_text = raw_text[:500] 
    print("\nTokenizing Sample Text from UD_English-EWT")
    print(f"Original Sample: {sample_text[:100]}")  

    # 3. Khởi tạo tokenizers
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    # 4. Tokenize với SimpleTokenizer
    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")

    # 5. Tokenize với RegexTokenizer
    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")

if __name__ == "__main__":
    test_tokenizers()
    main()