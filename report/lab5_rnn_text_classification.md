# Report Lab 5: RNN Text Classification - Intent Detection

**Ngày:** 17/11/2025

**Notebook:** [../notebook/lab5_rnn_text_classification.ipynb]

---

## 1. Giới thiệu

### 1.1. Mục tiêu

Lab này tập trung vào bài toán **Intent Detection** - phát hiện ý định người dùng từ câu văn bản. Mục tiêu là so sánh hiệu quả của các phương pháp:

1. **TF-IDF + Logistic Regression** (baseline truyền thống)
2. **Word2Vec + Dense Layer** (sử dụng word embeddings)
3. **LSTM với Pre-trained Embedding** (mô hình chuỗi với kiến thức từ Word2Vec)
4. **LSTM với Embedding học từ đầu** (mô hình chuỗi tự học representation)

### 1.2. Dataset

- **Dataset:** HWU64 (64 intents khác nhau)
- **Train:** 8,954 câu
- **Validation:** 1,076 câu
- **Test:** 1,076 câu
- **Số lượng intents:** 64 classes

---

## 2. Nhiệm vụ 1: Pipeline TF-IDF + Logistic Regression

### 2.1. Giải thích các bước làm

**Bước 1: Tiền xử lý dữ liệu**

- Sử dụng `LabelEncoder` để chuyển đổi nhãn intent (string) thành số nguyên.
- Fit encoder trên toàn bộ các nhãn từ train, val và test để đảm bảo ánh xạ nhất quán.
- Transform các nhãn trong các tập dữ liệu thành label số.

**Bước 2: Xây dựng Pipeline**

- Khởi tạo pipeline với 2 bước:
  - `TfidfVectorizer(max_features=5000)`: Chuyển văn bản thành vector sparse dựa trên TF-IDF, chỉ giữ 5000 từ phổ biến nhất.
  - `LogisticRegression(max_iter=1000)`: Mô hình phân loại tuyến tính phù hợp với high-dimensional sparse data.

**Bước 3: Huấn luyện và Đánh giá**

- Fit pipeline trên tập train.
- Dự đoán nhãn trên tập test.
- Đánh giá kết quả bằng accuracy và F1-score.

### 2.2. Kết quả thực nghiệm và nhận xét

**Kết quả:**

- **Test Accuracy:** 0.8355

**Nhận xét:**

**Ưu điểm:**

- Đơn giản, dễ triển khai
- Training nhanh (< 1 phút)
- Hiệu quả tốt với câu ngắn và từ khóa rõ ràng
- Giải thích được (feature importance)

**Nhược điểm:**

- Không nắm bắt được thứ tự từ
- Bỏ qua ngữ cảnh và quan hệ giữa các từ
- Yếu với câu phức tạp, phủ định
- Không xử lý được synonyms
- Bag-of-words mất thông tin cấu trúc câu

---

## 3. Nhiệm vụ 2: Word2Vec + Dense Layer

### 3.1. Giải thích các bước làm

**Bước 1: Huấn luyện Word2Vec**

- Tách các câu trong tập train thành danh sách từ để chuẩn bị corpus.
- Huấn luyện mô hình Word2Vec với:
  - `vector_size=128`: mỗi từ được biểu diễn bằng vector 200 chiều.
  - `window=5`: cửa sổ ngữ cảnh rộng để học semantic meaning.
  - `min_count=1`: giữ tất cả các từ (kể cả từ hiếm).
  - Huấn luyện 150 epochs để tối ưu word embeddings.

**Bước 2: Chuyển câu thành Vector Trung bình**

- Định nghĩa hàm `sentence_to_avg_vector()` để chuyển đổi câu:
  - Lấy vector của từng từ trong câu từ Word2Vec model.
  - Tính trung bình các word vectors → vector đại diện cho câu.
  - Xử lý trường hợp không có từ nào trong vocabulary bằng zero vector.
- **Hạn chế:** Phương pháp averaging mất thông tin về thứ tự từ.

**Bước 3: Xây dựng Mô hình Dense Neural Network**

- Xây dựng mạng neural network với kiến trúc:
  - Input layer: nhận vector 200 chiều từ Word2Vec.
  - 2 hidden layers: 128 -> 64 neurons với activation ReLU.
  - Dropout (0.5, 0.3) để giảm overfitting.
  - Output layer: 64 neurons với softmax cho 64 intents.
- Compile model với optimizer Adam, loss categorical_crossentropy.

**Bước 4: Training với EarlyStopping**

- Khởi tạo callback EarlyStopping với:
  - `monitor='val_loss'`: theo dõi validation loss.
  - `patience=5`: dừng nếu không cải thiện sau 5 epochs.
  - `restore_best_weights=True`: khôi phục weights tốt nhất.
- Huấn luyện model với batch_size=32, tối đa 50 epochs.
- Sử dụng validation data để theo dõi quá trình training.

### 3.2. Kết quả thực nghiệm và nhận xét

**Kết quả:**

- **Test Accuracy:** 0.8234
- **Test Loss:** 0.7011

**Nhận xét:**

**Ưu điểm:**

- Có semantic understanding (từ đồng nghĩa)
- Vector embeddings capture word relationships

**Nhược điểm:**

- Vẫn mất thứ tự từ do averaging
- Training lâu hơn (cần train Word2Vec + Neural Net)
- Cần nhiều dữ liệu để Word2Vec hiệu quả

---

## 4. Nhiệm vụ 3: LSTM với Pre-trained Embedding

### 4.1. Giải thích các bước làm

**Bước 1: Tokenization và Padding**

- Khởi tạo `Tokenizer` với `num_words=10000` (giới hạn vocabulary) và `oov_token="<UNK>"` để xử lý từ ngoài vocabulary.
- Fit tokenizer trên tập train để xây dựng word-to-index mapping.
- Chuyển đổi text thành sequences of integers bằng `texts_to_sequences()`.
- Padding các sequences về cùng độ dài (`maxlen=50`) để đưa vào mô hình.

**Bước 2: Tạo Embedding Matrix từ Word2Vec**

- Tính `vocab_size_actual` từ tokenizer (số từ trong vocabulary + 1).
- Khởi tạo ma trận embedding với shape `(vocab_size, embedding_dim)`.
- Duyệt qua các từ trong tokenizer word_index:
  - Nếu từ có trong Word2Vec model, lấy vector tương ứng.
  - Gán vector vào hàng tương ứng trong embedding matrix.
- Ma trận này sẽ được dùng để khởi tạo Embedding layer.

**Bước 3: Xây dựng Mô hình LSTM**

- Xây dựng Sequential model với các layers:
  - **Embedding Layer:** Khởi tạo weights từ embedding_matrix (Word2Vec), set `trainable=False` để đóng băng embedding.
  - **Bidirectional LSTM:** 128 units, xử lý sequence từ cả 2 chiều (forward + backward) để nắm bắt context tốt hơn. Dropout (0.2) để giảm overfitting.
  - **Dense Layer:** 64 neurons với ReLU activation.
  - **Dropout:** 0.3 để regularization.
  - **Output Layer:** 64 neurons với softmax cho phân loại multi-class.
- Compile với optimizer Adam, loss categorical_crossentropy.

**Bước 4: Training**

- Sử dụng EarlyStopping callback để dừng sớm khi validation loss không cải thiện (patience=5).
- Huấn luyện model với batch_size=32, tối đa 50 epochs.

### 4.2. Kết quả thực nghiệm và nhận xét

**Kết quả:**

- **Test Accuracy:** 0.8643 
- **Test Loss:** 0.4986

**Nhận xét:**

**Ưu điểm:**

- **Hiểu thứ tự từ:** LSTM xử lý sequential data
- **Nắm bắt long-range dependencies:** nhớ thông tin từ xa
- **Xử lý tốt phủ định:** "not want" khác "want not"
- Kết hợp ưu điểm: Pre-trained embeddings + Sequential processing

**Nhược điểm:**

- Training lâu hơn
- Phụ thuộc vào chất lượng Word2Vec

--

## 5. Nhiệm vụ 4: LSTM với Embedding học từ đầu

### 5.1. Giải thích các bước làm

**Xây dựng Mô hình**

- Xây dựng Sequential model với kiến trúc tương tự LSTM Pre-trained nhưng có sự khác biệt ở Embedding layer:
  - **Embedding Layer:**
    - `output_dim=100`: chiều embedding
    - `trainable=True`: embedding weights được học từ đầu qua backpropagation thay vì khởi tạo từ Word2Vec.
    - Không có pre-trained weights, mô hình tự học representation phù hợp với task.
  - **Bidirectional LSTM:** 128 units, dropout 0.2, xử lý sequence từ 2 chiều.
  - **Dense layers:** 64 neurons với ReLU, dropout 0.3.
  - **Output:** 64 neurons với softmax.

**Training**

- Sử dụng cùng hyperparameters với LSTM Pre-trained:
  - Optimizer Adam, loss categorical_crossentropy.
  - Batch_size=32, tối đa 50 epochs.
  - EarlyStopping với patience=5, theo dõi validation loss.
- Mô hình học cả embedding và LSTM weights đồng thời, phù hợp khi có đủ dữ liệu training.

### 5.2. Kết quả thực nghiệm và nhận xét

**Kết quả:**

- **Test Accuracy:** 0.8076
- **Test Loss:** 0.7322

**Nhận xét:**

**Ưu điểm:**

- **Tự học representation** phù hợp với domain
- Không phụ thuộc vào Word2Vec bên ngoài

**Nhược điểm:**

- Cần nhiều dữ liệu hơn để học tốt
- Không có kiến thức ngôn ngữ ban đầu
- Training từ đầu mất thời gian

---

## 6. So sánh và Đánh giá

### 6.1. Bảng so sánh kết quả định lượng

| Pipeline                           | F1-score (Macro) | Test Loss |
| ---------------------------------- | ---------------- | --------- |
| **TF-IDF + Logistic Regression**   | 0.8353           | N/A       |
| **Word2Vec (Avg) + Dense**         | 0.8202           | 0.7011    |
| **Embedding (Pre-trained) + LSTM** | 0.8420           | 0.5182    |
| **Embedding (Scratch) + LSTM**     | 0.7944           | 0.8450    |

**Kết luận từ bảng:**

Embedding (Pre-trained) + LSTM đạt F1-score cao nhất (0.8420), vượt trội hơn các phương pháp còn lại.

TF-IDF + Logistic Regression đạt 0.8353, cho thấy phương pháp đơn giản vẫn hiệu quả với bài toán này.

Word2Vec + Dense và Embedding Scratch + LSTM có hiệu suất thấp hơn, do averaging vector mất thông tin thứ tự và việc học embedding từ đầu cần nhiều dữ liệu hơn.

### 6.2. Phân tích định tính - Các câu phức tạp

#### Câu Test 1: Phủ định phức tạp

```
Câu: "can you remind me to not call my mom"
Ý định đúng: reminder_create
```

**Dự đoán:**

- TF-IDF + LR: `calendar_set` (SAI)
- Word2Vec + Dense: `email_sendemail` (SAI)
- LSTM Pre-trained: `calendar_set` (SAI)
- LSTM Scratch: `calendar_set` (SAI)

**Phân tích:**

Không mô hình nào dự đoán đúng. Câu chứa cấu trúc phủ định "to not call" kết hợp với "remind", tạo ngữ nghĩa phức tạp. TF-IDF chỉ nhận diện từ khóa "remind", "call" mà không hiểu mối quan hệ giữa chúng, dẫn đến nhầm lẫn với calendar_set. Word2Vec + Dense còn sai nhiều hơn khi dự đoán email_sendemail do averaging mất hoàn toàn cấu trúc câu. Cả hai LSTM đều thất bại, cho thấy training data thiếu mẫu về reminder với phủ định, và sự tương đồng cao giữa reminder_create và calendar_set khiến mô hình khó phân biệt.

#### Câu Test 2: Lựa chọn với từ khóa rõ ràng

```
Câu: "is it going to be sunny or rainy tomorrow"
Ý định đúng: weather_query
```

**Dự đoán:**

- TF-IDF + LR: `weather_query` (ĐÚNG)
- Word2Vec + Dense: `weather_query` (ĐÚNG)
- LSTM Pre-trained: `weather_query` (ĐÚNG)
- LSTM Scratch: `weather_query` (ĐÚNG)

**Phân tích:**

Tất cả mô hình đều dự đoán đúng nhờ từ khóa đặc trưng mạnh: "sunny", "rainy", "tomorrow". Ba từ này hiếm xuất hiện ngoài intent weather_query, tạo tín hiệu phân biệt rõ ràng. TF-IDF dễ dàng học pattern này từ training data. Word2Vec + Dense thành công vì "sunny" và "rainy" có semantic gần nhau trong không gian embedding, và averaging vẫn giữ được thông tin domain. LSTM không gặp khó khăn vì cả hai từ đều thuộc semantic field về thời tiết. Kết quả xác nhận rằng với từ khóa mạnh, phương pháp đơn giản cũng đủ hiệu quả.

#### Câu Test 3: Cấu trúc điều kiện dài và phức tạp

```
Câu: "find a flight from new york to london but not through paris"
Ý định đúng: flight_search
```

**Dự đoán:**

- TF-IDF + LR: `general_negate` (SAI)
- Word2Vec + Dense: `transport_query` (SAI)
- LSTM Pre-trained: `transport_query` (SAI)
- LSTM Scratch: `social_post` (SAI)

**Phân tích:**

Không mô hình nào dự đoán đúng. Câu có cấu trúc phức tạp "A from B to C but not through D" với nhiều thành phần: hành động (find), đối tượng (flight), điểm đi/đến, và ràng buộc phủ định. TF-IDF nhầm general_negate do focus vào "not" và "but", không hiểu "not through paris" là constraint của hành động tìm chuyến bay. Word2Vec + Dense và LSTM Pre-trained cùng dự đoán transport_query - nhận ra câu về di chuyển nhưng không phân biệt được "tìm chuyến bay cụ thể" và "hỏi thông tin di chuyển chung". LSTM Scratch cho kết quả tệ nhất (social_post) vì thiếu dữ liệu và representation kém cho "flight", "new york", "london".

**Kết luận:**

1. Từ khóa mạnh -> mọi phương pháp đều tốt (câu 2)
2. Cấu trúc phức tạp với pattern mới -> 2 mô hình LSTM hiện tại thất bại -> cần cải tiến cấu trúc mô hình, cần nhiều dữ liệu hơn
3. Pre-trained embeddings vượt trội hơn học từ đầu khi dữ liệu hạn chế

---

## Tài liệu tham khảo

1. [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
2. [Keras LSTM Documentation](https://keras.io/api/layers/recurrent_layers/lstm/)

---
