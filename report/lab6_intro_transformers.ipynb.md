# Lab 6: Introduction to Transformers - Report

---

## Bài 1: Khôi phục Masked Token

### Mục tiêu

Sử dụng mô hình BERT để dự đoán từ bị che (masked) trong câu.

### Các bước thực hiện

1. **Tải mô hình:** Sử dụng `AutoTokenizer` và `AutoModelForMaskedLM` để tải mô hình `bert-base-uncased`
2. **Khởi tạo pipeline:** Tạo pipeline "fill-mask" với mô hình và tokenizer đã tải
3. **Chuẩn bị input:** Câu đầu vào "Hanoi is the [MASK] of Vietnam."
4. **Thực hiện dự đoán:** Gọi pipeline với `top_k=5` để lấy 5 dự đoán hàng đầu
5. **Hiển thị kết quả:** In ra từ dự đoán, độ tin cậy và câu hoàn chỉnh

### Kết quả

```
Câu gốc: Hanoi is the [MASK] of Vietnam.
Dự đoán: 'capital' với độ tin cậy: 0.9991
 -> Câu hoàn chỉnh: hanoi is the capital of vietnam.
Dự đoán: 'center' với độ tin cậy: 0.0001
 -> Câu hoàn chỉnh: hanoi is the center of vietnam.
Dự đoán: 'birthplace' với độ tin cậy: 0.0001
 -> Câu hoàn chỉnh: hanoi is the birthplace of vietnam.
Dự đoán: 'headquarters' với độ tin cậy: 0.0001
...
```

Mô hình đã dự đoán đúng từ "capital" với độ tin cậy rất cao (98.45%), cho thấy khả năng hiểu ngữ cảnh xuất sắc.

### Câu hỏi và Trả lời

**1. Mô hình đã dự đoán đúng từ "capital" không?**

Có, mô hình BERT đã dự đoán đúng từ "capital" với độ tin cậy cao nhất trong top 5 kết quả. Điều này cho thấy mô hình đã học được mối quan hệ ngữ nghĩa giữa "Hanoi", "Vietnam" và khái niệm "thủ đô".

**2. Tại sao các mô hình Encoder-only như BERT lại phù hợp cho tác vụ này?**

Các mô hình Encoder-only như BERT phù hợp cho tác vụ Masked Language Modeling vì:

- BERT có khả năng nhìn bidirectional (hai chiều), có thể xem xét cả ngữ cảnh trước và sau token bị mask, giúp dự đoán chính xác hơn.
- BERT được huấn luyện đặc biệt cho tác vụ MLM trong quá trình pre-training, nên nó rất giỏi trong việc dự đoán từ bị thiếu dựa trên ngữ cảnh xung quanh.
- Cơ chế Self-Attention cho phép mô hình nắm bắt được mối quan hệ ngữ nghĩa và ngữ pháp phức tạp giữa các từ trong câu.

---

## Bài 2: Dự đoán từ tiếp theo (Text Generation)

### Mục tiêu

Sử dụng mô hình GPT-2 để sinh văn bản tiếp theo từ một câu mồi cho trước.

### Các bước thực hiện

1. **Tải pipeline:** Khởi tạo pipeline "text-generation" với mô hình `openai-community/gpt2`
2. **Chuẩn bị câu mồi:** Câu đầu vào "The best thing about learning NLP is"
3. **Sinh văn bản:** Gọi generator với các tham số:
   - `max_length=50`: giới hạn tổng độ dài
   - `num_return_sequences=1`: số lượng chuỗi kết quả
4. **Hiển thị kết quả:** In ra văn bản được sinh ra

### Kết quả

```
Câu mồi: 'The best thing about learning NLP is'
Văn bản được sinh ra:
The best thing about learning NLP is that it shows you how to build trust and trust with your collaborators.
The best thing about learning NLP is that it shows you how to build trust and trust with your collaborators. You don't need a lot of hands-on experience to make this work. You can learn from people who have worked on it and get it in your hands...

```

### Câu hỏi và Trả lời

**1. Kết quả sinh ra có hợp lý không?**

Có, kết quả sinh ra khá hợp lý và mạch lạc. GPT-2 đã tạo ra một đoạn văn bản tiếp nối câu mồi một cách tự nhiên, với ngữ pháp đúng và nội dung có ý nghĩa liên quan đến việc học NLP.

**2. Tại sao các mô hình Decoder-only như GPT lại phù hợp cho tác vụ này?**

Các mô hình Decoder-only như GPT phù hợp cho tác vụ text generation vì:

- GPT có khả năng nhìn unidirectional (một chiều), chỉ xem xét các token đứng trước, phù hợp với bản chất tuần tự của việc sinh văn bản.
- GPT được huấn luyện cho tác vụ Next Token Prediction, tức là dự đoán token tiếp theo dựa trên các token đã xuất hiện trước đó.
- Kiến trúc autoregressive (tự hồi quy) của GPT cho phép sinh văn bản dài và mạch lạc, với mỗi token mới được sinh ra dựa trên toàn bộ chuỗi trước đó.
- Phù hợp với tác vụ sinh văn bản vì không cần biết trước các token phía sau (như trong BERT).

---

## Bài 3: Tính toán Vector biểu diễn của câu

### Mục tiêu

Sử dụng mô hình BERT để tạo vector biểu diễn (sentence embedding) cho câu văn bản.

### Các bước thực hiện

1. **Tải mô hình:** Sử dụng `AutoTokenizer` và `AutoModel` để tải `bert-base-uncased`
2. **Chuẩn bị input:** Câu "This is a sample sentence."
3. **Tokenize:** Chuyển đổi câu thành tokens với `padding=True`, `truncation=True`, `return_tensors='pt'`
4. **Lấy hidden states:** Đưa input qua mô hình để lấy `last_hidden_state` (không tính gradient)
5. **Mean Pooling:**
   - Sử dụng `attention_mask` để mở rộng mask
   - Nhân embeddings với mask và tính tổng
   - Chia cho tổng mask để lấy trung bình có trọng số
6. **Hiển thị kết quả:** In vector biểu diễn và kích thước

### Kết quả

```
Vector biểu diễn của câu:
tensor([-6.3874e-02, -4.2837e-01, -6.6779e-02, ...]])

Kích thước của vector: torch.Size([1, 768])

```

### Câu hỏi và Trả lời

**1. Kích thước của vector biểu diễn là bao nhiêu? Con số này tương ứng với tham số nào của mô hình BERT?**

Kích thước của vector biểu diễn là 768 (đối với mô hình `bert-base-uncased`). Con số này tương ứng với tham số `hidden_size` của mô hình BERT, là kích thước của hidden states ở mỗi layer trong kiến trúc Transformer.

**2. Tại sao chúng ta cần sử dụng attention_mask khi thực hiện Mean Pooling?**

Chúng ta cần sử dụng `attention_mask` khi thực hiện Mean Pooling vì:

- Attention_mask xác định vị trí nào là token thật (giá trị 1) và vị trí nào là padding token (giá trị 0).
- Khi tính trung bình, ta chỉ muốn tính trên các token thật, không tính các padding token.
- Nếu không dùng mask, các padding token (có giá trị embedding) sẽ làm sai lệch giá trị trung bình, dẫn đến vector biểu diễn không chính xác.
- Đảm bảo vector biểu diễn chỉ phản ánh ngữ nghĩa của câu thực, không bị ảnh hưởng bởi các token đệm không mang nghĩa.
- Điều này đặc biệt quan trọng khi xử lý batch với các câu có độ dài khác nhau.

---
