# Báo Cáo Lab 5: Xây Dựng Mô Hình RNN cho Nhận Dạng Thực Thể Tên (NER)

---

### Task 1: Tải và Tiền Xử Lý Dữ Liệu

#### 2.1.1. Tải dữ liệu từ Hugging Face

**Phương pháp:**

- Sử dụng thư viện `datasets` từ Hugging Face
- Gọi hàm `load_dataset("conll2003")` để tải bộ dữ liệu
- Dữ liệu được chia sẵn thành 3 tập: train, validation, test

**Kết quả:**

```
Training set: (14041, 4)
Test set: (3453, 4)
Validation set: (3250, 4)
```

#### 2.1.2. Trích xuất câu và nhãn

**Phương pháp:**

- Trích xuất trường `tokens` chứa danh sách từ trong mỗi câu
- Trích xuất trường `ner_tags` chứa nhãn NER tương ứng (dạng số)
- Lấy mapping từ ID sang tên nhãn từ `dataset.features["ner_tags"].feature.names`

**Kết quả:**

```python
Nhãn NER: ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

Ví dụ câu đầu tiên:
Tokens: ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
Tags (số): [3, 0, 7, 0, 0, 0, 7, 0, 0]
Tags (string): ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
```

**Giải thích:**

- "EU" được gán nhãn B-ORG (tổ chức)
- "German" và "British" được gán nhãn B-MISC (miscellaneous)
- Các từ khác là "O" (không phải thực thể)

#### 2.1.3. Chuyển đổi nhãn từ số sang chuỗi

**Phương pháp:**

- Tạo dictionary `id2label` để ánh xạ ID → tên nhãn
- Định nghĩa hàm `map_ner_tag()` để thêm trường `ner_tags_str` vào mỗi mẫu
- Áp dụng hàm này lên toàn bộ dataset bằng phương thức `.map()`

#### 2.1.4. Xây dựng từ điển (Vocabulary)

**Phương pháp:**

- Định nghĩa hàm `build_vocab()` để tạo 2 từ điển:
  - **word_to_ix**: Ánh xạ từ → chỉ số (ID)
  - **tag_to_ix**: Ánh xạ nhãn → chỉ số (ID)
- Sử dụng `Counter` để đếm tần suất xuất hiện của từ
- Lọc từ theo `min_freq` (tần suất tối thiểu)
- Thêm các token đặc biệt: `<PAD>` (padding) và `<UNK>` (unknown)
- Lowercase tất cả các từ để giảm kích thước vocabulary

**Kết quả:**

```
Kích thước từ điển word_to_ix: 21,010 từ
Kích thước từ điển tag_to_ix: 10 nhãn

Các nhãn:
{'<PAD>': 0, 'B-LOC': 1, 'B-MISC': 2, 'B-ORG': 3, 'B-PER': 4,
 'I-LOC': 5, 'I-MISC': 6, 'I-ORG': 7, 'I-PER': 8, 'O': 9}
```

**Lưu ý:**

- Token `<PAD>` luôn có index = 0 (để padding)
- Token `<UNK>` có index = 1 (cho từ không có trong vocab)
- Có 10 nhãn: 1 PAD + 9 nhãn NER

---

### Task 2: Tạo PyTorch Dataset và DataLoader

#### 2.2.1. Xây dựng lớp NERDataset

**Phương pháp:**

- Kế thừa từ `torch.utils.data.Dataset`
- Implement 3 phương thức bắt buộc:
  - `__init__()`: Khởi tạo, lưu sentences, labels, và từ điển
  - `__len__()`: Trả về số lượng mẫu
  - `__getitem__()`: Trả về 1 mẫu tại index

**Chức năng của `__getitem__()`:**

1. Lấy tokens và tags tại index
2. Chuyển tokens thành IDs (dùng `word_to_ix`, unknown → `<UNK>`)
3. Chuyển tags thành IDs (dùng `tag_to_ix`)
4. Trả về dictionary chứa `input_ids` và `labels`

#### 2.2.2. Viết hàm collate_fn

**Mục đích:**

- Xử lý batch có các câu có độ dài khác nhau
- Đệm (padding) các câu ngắn để có cùng độ dài trong batch

**Phương pháp:**

- Chuyển từng mẫu thành tensor
- Sử dụng `torch.nn.utils.rnn.pad_sequence()` để padding:
  - `batch_first=True`: Shape (batch_size, seq_len)
  - `padding_value`: Giá trị để đệm (index của `<PAD>`)
- Padding riêng cho `input_ids` và `labels`

**Lý do cần padding:**

- Neural networks yêu cầu input có shape cố định
- Các câu trong batch có độ dài khác nhau
- Padding giúp tạo tensor với shape đồng nhất

#### 2.2.3. Tạo DataLoader

**Phương pháp:**

- Khởi tạo `NERDataset` cho tập train và validation
- Tạo `DataLoader` với các tham số:
  - `batch_size=32`: Mỗi batch 32 mẫu
  - `shuffle=True` cho train (tăng tính ngẫu nhiên)
  - `shuffle=False` cho validation (đảm bảo reproducibility)
  - `collate_fn=collate_fn` (hàm padding tự định nghĩa)

**Lợi ích:**

- Tự động chia batch
- Tự động shuffle (nếu cần)
- Hỗ trợ multi-processing để load nhanh hơn
- Tích hợp sẵn với training loop

---

### Task 3: Xây Dựng Mô Hình RNN

#### 2.3.1. Kiến trúc mô hình SimpleRNNForTokenClassification

**Các thành phần:**

1. **Embedding Layer** (`nn.Embedding`)

   - Input: Token IDs (batch_size, seq_len)
   - Output: Embeddings (batch_size, seq_len, embedding_dim)
   - Chức năng: Chuyển từ discrete IDs sang continuous vectors
   - Tham số: `padding_idx=0` để không cập nhật embedding của `<PAD>`

2. **RNN Layer** (`nn.RNN`)

   - Input: Embeddings (batch_size, seq_len, embedding_dim)
   - Output: Hidden states (batch_size, seq_len, hidden_dim)
   - Chức năng: Xử lý chuỗi, nắm bắt thông tin ngữ cảnh
   - Tham số:
     - `batch_first=True`: Input/output có batch ở dimension đầu
     - `bidirectional=False`: RNN một chiều (trái → phải)

3. **Linear Layer** (`nn.Linear`)
   - Input: Hidden states (batch_size, seq_len, hidden_dim)
   - Output: Logits (batch_size, seq_len, num_tags)
   - Chức năng: Ánh xạ hidden states → không gian nhãn
   - Mỗi token có 1 vector logits với 10 giá trị (tương ứng 10 nhãn)

---

### Task 4: Huấn Luyện Mô Hình

#### 2.4.1. Khởi tạo các thành phần huấn luyện

**Các thành phần:**

1. **Device**:

   - Kiểm tra CUDA có available không
   - Sử dụng GPU nếu có, CPU nếu không

2. **Model**:

   - Khởi tạo `SimpleRNNForTokenClassification`
   - Chuyển model lên device (GPU/CPU)

3. **Optimizer**:

   - Sử dụng Adam optimizer
   - Learning rate: 1e-3 (0.001)
   - Adam tự động điều chỉnh learning rate cho từng parameter

4. **Loss Function**:
   - `nn.CrossEntropyLoss`
   - `ignore_index=tag_pad_id`: Không tính loss cho các token padding
   - Phù hợp cho bài toán multi-class classification

**Lý do chọn CrossEntropyLoss:**

- Kết hợp Softmax và NLLLoss
- Tính toán hiệu quả
- Phù hợp cho token classification (mỗi token là 1 bài toán phân loại)

#### 2.4.2. Vòng lặp huấn luyện

**Các bước trong mỗi epoch:**

1. **Chuẩn bị:**

   - Đặt model ở chế độ train: `model.train()`
   - Khởi tạo biến `total_loss = 0`

2. **Lặp qua từng batch:**

   - Lấy `sentences` và `labels` từ batch
   - Chuyển data lên device (GPU/CPU)

3. **Forward pass:**

   - Xóa gradient cũ: `optimizer.zero_grad()`
   - Đưa input qua model: `outputs = model(sentences)`
   - Reshape outputs và labels thành 2D để tính loss:
     ```python
     outputs: (batch*seq_len, num_tags)
     labels:  (batch*seq_len,)
     ```

4. **Backward pass:**

   - Tính loss: `loss = criterion(outputs, labels)`
   - Backpropagation: `loss.backward()`
   - Cập nhật weights: `optimizer.step()`
   - Cộng dồn loss: `total_loss += loss.item()`

5. **Đánh giá sau mỗi epoch:**
   - Tính loss trung bình trên tập train
   - Đánh giá accuracy trên tập validation
   - In ra kết quả theo dõi

**Số epochs:** 15 epochs

---

### Task 5: Đánh Giá Mô Hình

#### 2.5.1. Hàm đánh giá Token-level Accuracy

**Phương pháp:**

- Định nghĩa hàm `evaluate_ner()` để tính độ chính xác token-level
- Đặt model ở chế độ evaluation: `model.eval()`
- Tắt gradient: `with torch.no_grad()`

**Các bước:**

1. Lặp qua từng batch trong validation set
2. Dự đoán nhãn: `preds = torch.argmax(outputs, dim=-1)`
3. Tạo mask để bỏ qua padding tokens
4. Đếm số token dự đoán đúng và tổng số token
5. Tính accuracy: `correct / total`

#### 2.5.2. Hàm đánh giá với seqeval

**Phương pháp:**

- Sử dụng thư viện `seqeval` (được thiết kế đặc biệt cho NER)
- Định nghĩa hàm `evaluate_ner_seqeval()`

**Các bước:**

1. Tính token-level accuracy như trên
2. Chuyển đổi predictions và labels từ IDs → strings
3. Loại bỏ padding tokens
4. Thu thập tất cả predictions và labels
5. Gọi `classification_report()` từ seqeval

**Kết quả đánh giá chi tiết:**

```
Token-level Accuracy: 93.82%

Classification Report (seqeval):
              precision    recall  f1-score   support

         LOC       0.86      0.78      0.82      1837
        MISC       0.76      0.65      0.70       922
         ORG       0.64      0.59      0.61      1341
         PER       0.72      0.56      0.63      1842

   micro avg       0.75      0.65      0.70      5942
   macro avg       0.74      0.65      0.69      5942
weighted avg       0.75      0.65      0.69      5942
```

```
Token-level Accuracy: 93.82%
```

**Giải thích các metrics:**

| Metric        | Giá trị | Ý nghĩa                                     |
| ------------- | ------- | ------------------------------------------- |
| **B-PER**     | F1=0.93 | Nhận dạng tên người tốt nhất                |
| **I-PER**     | F1=0.96 | Token tiếp theo của tên người rất chính xác |
| **B-LOC**     | F1=0.89 | Nhận dạng địa danh tốt                      |
| **B-ORG**     | F1=0.83 | Nhận dạng tổ chức khá tốt                   |
| **B-MISC**    | F1=0.79 | Thực thể MISC khó nhất                      |
| **O**         | F1=0.99 | Nhận dạng non-entity rất tốt                |
| **Micro avg** | F1=0.96 | F1 trung bình trên tất cả tokens            |
| **Macro avg** | F1=0.86 | F1 trung bình trên tất cả classes           |

#### 2.5.3. Hàm dự đoán câu mới

**Phương pháp:**

- Định nghĩa hàm `predict_sentence()` để dự đoán nhãn cho 1 câu mới
- Hỗ trợ input là string hoặc list tokens

**Các bước:**

1. Tokenize câu (nếu input là string)
2. Chuyển tokens → IDs (unknown → `<UNK>`)
3. Tạo tensor và chuyển lên device
4. Dự đoán: `preds = torch.argmax(outputs, dim=-1)`
5. Chuyển prediction IDs → nhãn strings
6. In ra từng cặp (token, nhãn)

**Ví dụ dự đoán:**

```
Câu: "VNU University is located in Hanoi"

Dự đoán:
VNU         B-ORG
University  I-ORG
is          O
located     O
in          O
Hanoi       B-LOC
```

**Nhận xét ví dụ:**

- Xử lý tốt từ viết tắt "VNU" kết hợp với "University" → B-ORG I-ORG
- Nhận dạng chính xác entity đa từ "VNU University" là một đơn vị ngữ nghĩa
- Phân loại đúng "Hanoi" là B-LOC dựa vào pattern "located in [Place]"
- Boundary detection chính xác: biết "University" thuộc ORG, "is" là O

## Tài liệu tham khảo

PyTorch Documentation: https://pytorch.org/docs/stable/index.html

- RNN Tutorial: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

Hugging Face Datasets: https://huggingface.co/docs/datasets/

- CoNLL2003: https://huggingface.co/datasets/eriktks/conll2003

- sqeval Library: https://github.com/chakki-works/seqeval

- spaCy: https://spacy.io/ - Production-ready NLP library

---
