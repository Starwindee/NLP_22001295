# Data Description

## 1. Thư mục `data/hwu/`

### Mô tả chung

Bộ dữ liệu HWU64 (Harbin Institute of Technology and Wuhan University) dùng cho bài toán **Intent Detection** - phát hiện ý định người dùng từ câu văn bản.

### Cấu trúc files

#### `categories.json`

- **Mô tả:** Tệp JSON chứa danh sách 64 nhãn intent/categories
- **Định dạng:** JSON array hoặc dictionary
- **Ví dụ:** `["alarm_query", "alarm_remove", "alarm_set", ...]`

#### `train.csv`

- **Mô tả:** Dữ liệu huấn luyện đầy đủ
- **Số lượng:** ~8,954 samples
- **Cột:**
  - `text` (string): Câu văn bản đầu vào
  - `intent` (string): Nhãn intent (1 trong 64 classes)
- **Ví dụ:**
  ```csv
  text,intent
  "set an alarm for 7am",alarm_set
  "what's the weather today",weather_query
  ```

#### `train_5.csv`

- **Mô tả:** Phiên bản rút gọn của train.csv với 5 nhãn chính
- **Mục đích:** Dùng cho thử nghiệm nhanh hoặc học với ít classes

#### `train_10.csv`

- **Mô tả:** Phiên bản rút gọn với 10 nhãn chính
- **Mục đích:** Thử nghiệm với độ phức tạp trung bình

#### `val.csv`

- **Mô tả:** Dữ liệu validation
- **Số lượng:** ~1,076 samples
- **Định dạng:** Tương tự train.csv
- **Mục đích:** Đánh giá mô hình trong quá trình huấn luyện

#### `test.csv`

- **Mô tả:** Dữ liệu test
- **Số lượng:** ~1,076 samples
- **Định dạng:** Tương tự train.csv
- **Mục đích:** Đánh giá cuối cùng hiệu suất mô hình

---

## 2. Thư mục `data/twitter-financial-news-sentiment/`

### Mô tả

Bộ dữ liệu phân tích cảm xúc trên Twitter trong lĩnh vực tài chính.

### Files

#### `sent_train.csv`

- **Mô tả:** Dữ liệu huấn luyện
- **Cột:**
  - `text` (string): Nội dung tweet
  - `sentiment` (string/int): Nhãn cảm xúc (positive/negative/neutral hoặc 0/1/2)

#### `sent_valid.csv`

- **Mô tả:** Dữ liệu validation
- **Định dạng:** Tương tự sent_train.csv

### Mục đích

- Sentiment Analysis
- Financial NLP
- Social Media Text Analysis

---

## 3. Thư mục `data/UD_English-EWT/`

### Mô tả

Universal Dependencies English Web Treebank - Bộ dữ liệu có chú thích ngữ pháp cho tiếng Anh.

### Files

#### `en_ewt-ud-dev.conllu`

- **Định dạng:** CoNLL-U format
- **Mô tả:** Dữ liệu development set
- **Annotations:**
  - POS tags (Part-of-Speech)
  - Dependency parsing
  - Morphological features

#### `en_ewt-ud-train.conllu`

- **Mô tả:** Dữ liệu huấn luyện
- **Định dạng:** CoNLL-U format

#### `en_ewt-ud.json.zip`

- **Mô tả:** Phiên bản JSON của UD English-EWT
- **Định dạng:** Compressed JSON
- **Cần giải nén trước khi sử dụng**

### Mục đích

- POS Tagging
- Dependency Parsing
- Named Entity Recognition
- Syntactic Analysis

### Cấu trúc CoNLL-U

```
# sent_id = 1
# text = I love NLP.
1   I       I       PRON    PRP     _   2   nsubj   _   _
2   love    love    VERB    VBP     _   0   root    _   _
3   NLP     NLP     PROPN   NNP     _   2   obj     _   _
4   .       .       PUNCT   .       _   2   punct   _   _
```

---

## 4. File `data/c4-train.00000-of-01024-30K.json`

### Mô tả

Một phần của bộ dữ liệu **C4 (Colossal Clean Crawled Corpus)** - corpus web-crawled khổng lồ.

### Thông tin

- **Định dạng:** JSON Lines (mỗi dòng là một document JSON)
- **Kích thước:** 30,000 documents
- **Phần:** Shard 0/1024 của tập train
- **Cấu trúc:**
  ```json
  {
    "text": "Content of the document...",
    "url": "https://example.com",
    "timestamp": "2019-01-01"
  }
  ```

### Mục đích

- Pre-training Language Models
- Text Generation
- Masked Language Modeling

---

## 5. File `data/sentiments.csv`

### Mô tả

Bộ dữ liệu tổng quát cho bài toán phân loại cảm xúc.

### Cấu trúc

- **Cột:**
  - `text` (string): Văn bản đầu vào
  - `sentiment` (string/int): Nhãn cảm xúc
    - Có thể là: `positive`, `negative`, `neutral`
    - Hoặc: `0`, `1`, `2`

### Ví dụ

```csv
text,sentiment
"This product is amazing!",positive
"Terrible experience.",negative
"It's okay.",neutral
```

### Mục đích

- Sentiment Classification
- Text Classification
- Binary/Multi-class Classification

---
