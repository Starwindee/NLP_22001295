# Report Lab 5: RNN for POS Tagging

**Ngày:** 17/11/2025

---

## 1. Giới thiệu

### 1.1. Mục tiêu

Lab này tập trung vào bài toán **Part-of-Speech (POS) Tagging** - gán nhãn từ loại cho từng từ trong câu. Mục tiêu là xây dựng mô hình RNN với PyTorch để:

1. Đọc và xử lý dữ liệu CoNLL-U format
2. Xây dựng PyTorch Dataset và DataLoader với padding
3. Thiết kế mô hình Simple RNN cho token classification
4. Huấn luyện và đánh giá mô hình trên tập UD_English-EWT

---

## 2. Task 1: Tải và Tiền xử lý Dữ liệu

### 2.1. Giải thích các bước làm

**Bước 1: Đọc file CoNLL-U**

- Viết hàm `load_conllu(file_path)` để đọc dữ liệu định dạng CoNLL-U.
- Xử lý từng dòng trong file:
  - Bỏ qua dòng comment (bắt đầu bằng `#`).
  - Dòng trống đánh dấu kết thúc một câu.
  - Parse dòng dữ liệu: tách bằng `\t`, lấy cột 1 (word) và cột 3 (UPOS tag).
  - Chuyển word về lowercase để giảm kích thước vocabulary.
- Trả về list các câu, mỗi câu là list of `(word, upos_tag)` tuples.

**Bước 2: Xây dựng Vocabulary**

- Viết hàm `build_vocab(sentences, min_freq)` để tạo từ điển word_to_ix và tag_to_ix.
- Đếm tần suất xuất hiện của từng từ bằng `Counter`.
- Thu thập tất cả các UPOS tags vào set.
- Xây dựng `word_to_ix`:
  - Thêm token đặc biệt: `<PAD>` (index=0), `<UNK>` (index=1).
  - Chỉ thêm các từ có tần suất >= `min_freq` vào vocabulary.
- Xây dựng `tag_to_ix`:
  - Thêm `<PAD>` tag (index=0).
  - Thêm các UPOS tags theo thứ tự sorted để đảm bảo nhất quán.
- Trả về hai dictionaries: word_to_ix và tag_to_ix.

**Kết quả:**

- **Vocabulary size:** 17115 từ
- **Số UPOS tags:** 19 nhãn

## 3. Task 2: Tạo PyTorch Dataset và DataLoader

### 3.1. Giải thích các bước làm

**Bước 1: Xây dựng POSDataset**

- Tạo class `POSDataset` kế thừa từ `torch.utils.data.Dataset`.
- Constructor nhận:
  - `sentences`: list các câu từ bước 1.
  - `word_to_ix`, `tag_to_ix`: vocabularies từ bước 1.
- Implement `__len__()`: trả về số lượng câu.
- Implement `__getitem__(idx)`:
  - Lấy câu tại index `idx`.
  - Chuyển words thành indices: dùng `word_to_ix.get(word, <UNK>)` để xử lý OOV words.
  - Chuyển tags thành indices: dùng `tag_to_ix[tag]`.
  - Convert sang `torch.LongTensor`.
  - Trả về tuple `(sentence_indices, tag_indices)`.

**Bước 2: Viết Collate Function**

- Viết hàm `collate_fn(batch)` để xử lý padding cho batch.
- Nhận input: list of `(sentence_indices, tag_indices)` tuples.
- Xử lý:
  - Tách sentences và tags từ batch.
  - Lưu độ dài thực của mỗi câu vào tensor `lengths`.
  - Sử dụng `pad_sequence()` để pad về cùng độ dài:
    - `padding_value=0` cho sentences (tương ứng `<PAD>` token).
    - `padding_value=-100` cho tags (để `CrossEntropyLoss` ignore padding).
  - `batch_first=True` để shape là `(batch_size, max_len)`.
- Trả về: `(sentences_padded, tags_padded, lengths)`.

**Bước 3: Khởi tạo DataLoader**

- Tạo `train_dataset` và `dev_dataset` từ `POSDataset`.
- Khởi tạo `DataLoader` với:
  - `batch_size=32` cho training.
  - `shuffle=True` cho train (random batches), `False` cho dev.
  - `collate_fn=collate_fn` để áp dụng padding.

## 4. Task 3: Xây dựng Mô hình RNN

### 4.1. Giải thích các bước làm

**Kiến trúc mô hình SimpleRNNForTokenClassification**

- Tạo class `SimpleRNNForTokenClassification` kế thừa từ `nn.Module`.
- Constructor nhận:
  - `vocab_size`: kích thước vocabulary.
  - `embedding_dim`: chiều của embedding vectors.
  - `hidden_dim`: số units trong RNN hidden state.
  - `num_tags`: số lượng POS tags.
  - `padding_idx=0`: index của `<PAD>` token.

**Các layers:**

1. **Embedding Layer:**

   - `nn.Embedding(vocab_size, embedding_dim, padding_idx=0)`.
   - Chuyển word indices -> dense vectors (embedding_dim chiều).

2. **RNN Layer:**

   - `nn.RNN(embedding_dim, hidden_dim, batch_first=True)`.
   - Xử lý sequence embedding vectors, tạo hidden states tại mỗi time step.

3. **Linear Layer:**
   - `nn.Linear(hidden_dim, num_tags)`.
   - Ánh xạ hidden states -> logits cho num_tags classes.

---

## 5. Task 4: Huấn luyện Mô hình

### 5.1. Giải thích các bước làm

**Bước 1: Khởi tạo mô hình và optimizer**

- Khởi tạo model `SimpleRNNForTokenClassification` với các hyperparameters.
- Chuyển model lên device (GPU nếu có, CPU nếu không).
- Khởi tạo optimizer: `Adam(lr=0.001)`.
- Khởi tạo loss function: `CrossEntropyLoss(ignore_index=-100)` để ignore padding.

**Bước 2: Viết hàm train_epoch()**

- Đặt model ở chế độ training: `model.train()`.
- Loop qua từng batch trong train_loader:
  - Chuyển data lên device.
  - **5 bước kinh điển:**
    1. `optimizer.zero_grad()`: Xóa gradient cũ.
    2. Forward pass: `logits = model(words_padded)`.
    3. Tính loss: Reshape logits và tags về 2D rồi tính `criterion()`.
    4. Backward pass: `loss.backward()` để tính gradient.
    5. Update weights: `optimizer.step()`.
  - Lưu loss để tính trung bình.
- In loss mỗi 100 batches để theo dõi quá trình training.
- Trả về average loss của epoch.

**Bước 3: Training loop**

- Loop qua `num_epochs=10`:
  - Gọi `train_epoch()` để huấn luyện.
  - Gọi `evaluate()` trên train và dev set để đánh giá.
  - Lưu lịch sử (train_loss, train_acc, dev_acc).
  - In kết quả sau mỗi epoch.
  - Lưu model tốt nhất (best dev accuracy).

---

## 6. Task 5: Đánh giá Mô hình

### 6.1. Giải thích các bước làm

**Bước 1: Viết hàm evaluate()**

- Đặt model ở chế độ evaluation: `model.eval()`.
- Tắt gradient computation: `with torch.no_grad()` để tiết kiệm memory.
- Loop qua từng batch trong loader:
  - Chuyển data lên device.
  - Forward pass: `logits = model(words_padded)`.
  - Lấy predictions: `argmax(logits, dim=-1)` để lấy tag có xác suất cao nhất.
  - Tạo mask: `tags_padded != -100` để bỏ qua padding tokens.
  - Đếm số token dự đoán đúng: `(predictions == tags_padded) & mask`.
- Tính accuracy: `correct_predictions / total_tokens`.
- Trả về accuracy.

**Bước 2: Viết hàm predict_sentence()**

- Hàm để dự đoán POS tags cho câu mới (inference).
- Input: câu văn bản dạng string.
- Xử lý:
  - Tách câu thành list words bằng `split()`.
  - Chuyển words về lowercase.
  - Map words → indices, dùng `<UNK>` cho OOV words.
  - Convert thành tensor và chuyển lên device.
  - Forward pass để lấy predictions.
  - Decode predictions về tag names bằng `ix_to_tag`.
- Trả về list of `(word, predicted_tag)` tuples.

**Bước 3: Test trên câu mới**

- Test hàm `predict_sentence()` trên các câu example:
  - "The cat is sleeping on the couch"
  - "I love programming in Python"
  - "She quickly ran to the store"
  - "The beautiful flowers are blooming in the garden"

### 6.2. Kết quả thực nghiệm và nhận xét

**Kết quả huấn luyện (10 epochs):** (chi tiết trong notebook)

**Best Dev Accuracy:** 0.8719 (Epoch 6)

**Nhận xét:**

**Quan sát:**

- **Loss giảm đều:** Training loss giảm từ 0.0377 -> 0.0154
- **Accuracy có tăng nhẹ**

**Kết quả dự đoán trên câu test:**

**Câu 1: "The cat is sleeping on the couch"**

```
the         : DET
cat         : NOUN
is          : AUX
sleeping    : VERB
on          : ADP
the         : DET
couch       : NOUN
```

**Đánh giá:** Chính xác 100%. Model nhận diện đúng DET-NOUN, AUX-VERB, ADP structure.

**Câu 2: "I love programming in Python"**

```
i           : PRON
love        : VERB
programming : NOUN
in          : ADP
python      : PROPN
```

**Đánh giá:** 100% chính xác. Nhận diện đúng "programming" là NOUN (gerund), "Python" là PROPN.

**Câu 3: "She quickly ran to the store"**

```
she         : PRON
quickly     : ADV
ran         : VERB
to          : ADP
the         : DET
store       : NOUN
```

**Đánh giá:** Nhận diện đúng adverb "quickly" modify verb "ran".

**Câu 4: "The beautiful flowers are blooming in the garden"**

```
the         : DET
beautiful   : ADJ
flowers     : NOUN
are         : AUX
blooming    : VERB
in          : ADP
the         : DET
garden      : NOUN
```

**Đánh giá:** Nhận diện đúng cấu trúc ADJ-NOUN, progressive tense với AUX-VERB.

---

## 7. Vấn đề gặp phải và Cách giải quyết

### 7.1. Vấn đề 1: Padding trong batch

**Vấn đề:**

- Các câu trong batch có độ dài khác nhau, không thể stack thành tensor.
- Padding tokens không nên được tính vào loss và accuracy.

**Giải quyết:**

- Sử dụng `pad_sequence()` để pad về cùng độ dài.
- Set `padding_value=-100` cho tags để `CrossEntropyLoss` ignore khi tính loss.
- Tạo mask `tags != -100` để bỏ qua padding khi tính accuracy.

---

# 8. Tài liệu tham khảo

1. **Universal Dependencies:**

   - [UD English EWT Dataset](https://universaldependencies.org/treebanks/en_ewt/index.html)
   - [CoNLL-U Format Specification](https://universaldependencies.org/format.html)

2. **PyTorch Documentation:**

   - [torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
   - [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
   - [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)
   - [pad_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html)

---
