# NLP Labs – Tokenization & Vectorization

## Mô tả công việc

(có sử dụng AI hỗ trợ)

### Lab 1: Tokenization

- Cài đặt **interface Tokenizer** (`src/core/interfaces.py`) sử dụng `abc.ABC`.
- Cài đặt **SimpleTokenizer** (`src/preprocessing/simple_tokenizer.py`):
  - Chuyển text về chữ thường.
  - Tách token theo khoảng trắng.
  - Xử lý dấu câu cơ bản (`. , ? !`) bằng cách tách riêng.
- Cài đặt **RegexTokenizer** (`src/preprocessing/regex_tokenizer.py`):
  - Sử dụng biểu thức chính quy để tách từ và dấu câu.
  - Xử lý tốt hơn SimpleTokenizer
- **Task 3:** Thử nghiệm trên dataset thật (UD_English-EWT).
  - Viết module `src/core/dataset_loaders.py` để load dataset.
  - Tokenize một đoạn văn bản (~500 ký tự đầu).

### Lab 2: Vectorization

- Cài đặt **interface Vectorizer** (`src/core/interfaces.py`) với 3 phương thức:
  - `fit`
  - `transform`
  - `fit_transform`
- Cài đặt **CountVectorizer** (`src/representations/count_vectorizer.py`):
  - Nhận một tokenizer (Simple hoặc Regex).
  - Học **vocabulary** từ corpus.
  - Biến đổi văn bản thành **document-term matrix**.
- Viết test (`test/lab2.py`) để thử nghiệm CountVectorizer trên corpus mẫu.

## Kết quả

### Ví dụ 1: Tokenizer

Input:"Hello, World! This is a test."

- **SimpleTokenizer Output:** ['hello,', 'world!', 'this', 'is', 'a', 'test.']
- **RegexTokenizer Output:** ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
- **UD_English-EWT Dataset:**
  - Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of
  - SimpleTokenizer Output (first 20 tokens): ['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani,', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim,', 'near', 'the']
  - RegexTokenizer Output (first 20 tokens): ['al', '-', 'zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al', '-', 'ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the']

### Ví dụ 2: CountVectorizer

```python
corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]

Vocabulary: {'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}
Document-term matrix:
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]

```
