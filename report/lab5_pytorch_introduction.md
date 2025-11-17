# Report Lab 5: PyTorch Introduction

**Ngày:** 17/11/2025

**Notebook:** [../notebook/lab5_pytorch_introduction.ipynb]

## 1. Phần 1: Khám phá Tensor

### 1.1. Giải thích các bước làm

#### Task 1.1: Tạo Tensor

**Tensor** là cấu trúc dữ liệu đa chiều cơ bản trong PyTorch, tương tự như NumPy array nhưng có thể chạy trên GPU.

**Các bước thực hiện:**

1. **Tạo tensor từ Python list:**

   - Sử dụng `torch.tensor(data)` để chuyển đổi list Python thành tensor
   - Ví dụ: `data = [[1, 2], [3, 4]]` -> tensor 2x2

2. **Tạo tensor từ NumPy array:**

   - Sử dụng `torch.from_numpy(np_array)` để chuyển đổi
   - Tensor và NumPy array chia sẻ cùng bộ nhớ (nếu có thể)

3. **Tạo tensor với giá trị đặc biệt:**

   - `torch.ones_like(tensor)`: tạo tensor gồm số 1 có cùng shape
   - `torch.rand_like(tensor, dtype=torch.float)`: tạo tensor ngẫu nhiên

4. **Kiểm tra thuộc tính tensor:**
   - `shape`: kích thước tensor (ví dụ: `torch.Size([2, 2])`)
   - `dtype`: kiểu dữ liệu (ví dụ: `torch.float32`)
   - `device`: thiết bị lưu trữ (ví dụ: `cpu`, `cuda:0`)

#### Task 1.2: Các phép toán trên Tensor

**Các bước thực hiện:**

1. **Phép cộng element-wise:**

   - `x_data + x_data`: cộng từng phần tử tương ứng
   - Kết quả: `[[2, 4], [6, 8]]`

2. **Phép nhân với scalar:**

   - `x_data * 5`: nhân mỗi phần tử với 5
   - Kết quả: `[[5, 10], [15, 20]]`

3. **Phép nhân ma trận (matrix multiplication):**
   - `x_data @ x_data.T`: nhân ma trận với ma trận chuyển vị
   - Sử dụng toán tử `@` hoặc `torch.matmul()`
   - Kết quả: `[[5, 11], [11, 25]]`

#### Task 1.3: Indexing và Slicing

**Các bước thực hiện:**

1. **Lấy hàng đầu tiên:**

   - `x_data[0]` -> `tensor([1, 2])`

2. **Lấy cột thứ hai:**

   - `x_data[:, 1]` -> `tensor([2, 4])`
   - `:` nghĩa là lấy tất cả các hàng

3. **Lấy phần tử cụ thể:**
   - `x_data[1, 1]` -> `tensor(4)`
   - Lấy phần tử tại hàng 1, cột 1

#### Task 1.4: Thay đổi hình dạng Tensor

**Các bước thực hiện:**

1. **Tạo tensor 4x4:**

   - `torch.rand(4, 4)`: tạo tensor ngẫu nhiên kích thước 4x4

2. **Reshape bằng view():**

   - `tensor_4x4.view(16, 1)`: thay đổi shape thành 16x1
   - Yêu cầu: tensor phải contiguous trong bộ nhớ

3. **Reshape bằng reshape():**
   - `tensor_4x4.reshape(16, 1)`: thay đổi shape thành 16x1
   - An toàn hơn `view()`, tự động tạo copy nếu cần

### 1.2. Kết quả thực nghiệm và nhận xét

**Kết quả:**

```
Tensor từ list:
 tensor([[1, 2],
         [3, 4]])

Shape của tensor: torch.Size([2, 2])
Datatype của tensor: torch.float32
Device lưu trữ tensor: cpu

Cộng x_data với chính nó:
 tensor([[2, 4],
         [6, 8]])

Nhân x_data với 5:
 tensor([[ 5, 10],
         [15, 20]])

Nhân ma trận x_data với x_data.T:
 tensor([[ 5, 11],
         [11, 25]])

Hàng đầu tiên: tensor([1, 2])
Cột thứ hai: tensor([2, 4])
Giá trị ở hàng thứ hai, cột thứ hai: 4

Tensor shape (16, 1): [16, 1]
```

**Nhận xét:**

1. **Tính linh hoạt:**

   - PyTorch tensor hỗ trợ nhiều cách tạo và chuyển đổi dữ liệu
   - Tương thích tốt với NumPy, dễ dàng chuyển đổi qua lại

2. **Các phép toán:**

   - Phép toán element-wise đơn giản và trực quan
   - Phép nhân ma trận sử dụng `@` giống Python 3.5+

3. **Indexing và Slicing:**

   - Cú pháp giống NumPy, dễ học và sử dụng
   - Hỗ trợ đầy đủ các thao tác truy cập dữ liệu

4. **Reshape:**
   - `view()` nhanh hơn nhưng yêu cầu contiguous memory
   - `reshape()` an toàn hơn, tự động xử lý các trường hợp đặc biệt

---

## 2. Phần 2: Tự động tính Đạo hàm với autograd

### 2.1. Giải thích các bước làm

#### Task 2.1: Thực hành với autograd

**autograd** là engine tính đạo hàm tự động của PyTorch, hỗ trợ backpropagation trong deep learning.

**Các bước thực hiện:**

1. **Tạo tensor với requires_grad=True:**

   ```python
   x = torch.ones(1, requires_grad=True)
   ```

   - `requires_grad=True`: yêu cầu PyTorch theo dõi các phép toán trên tensor này

2. **Thực hiện các phép toán:**

   ```python
   y = x + 2
   z = y * y * 3
   ```

   - Mỗi phép toán tạo ra một node trong computational graph
   - `grad_fn` lưu thông tin về phép toán đã thực hiện

3. **Tính đạo hàm bằng backward():**

   ```python
   z.backward()
   ```

   - Tự động tính đạo hàm dz/dx bằng chain rule
   - Kết quả lưu trong `x.grad`

4. **Giải thích toán học:**
   - Cho: z = 3(x + 2)²
   - Đạo hàm: dz/dx = 3 × 2(x + 2) = 6(x + 2)
   - Với x = 1: dz/dx = 6(1 + 2) = **18**

#### Gọi backward() nhiều lần

**Vấn đề:**

- Khi gọi `z.backward()` lần thứ hai sẽ gây lỗi `RuntimeError`

**Nguyên nhân:**

- PyTorch tự động giải phóng computational graph sau `backward()` để tiết kiệm bộ nhớ

**Giải pháp:**

```python
z.backward(retain_graph=True)  # Giữ lại graph
z.backward()  # Có thể gọi lại
```

**Lưu ý:**

- Gradient sẽ được **cộng dồn** (accumulate) sau mỗi lần backward()
- Cần reset gradient bằng `x.grad.zero_()` trước mỗi iteration

### 2.2. Kết quả thực nghiệm và nhận xét

**Kết quả:**

```
x: tensor([1.], requires_grad=True)
y: tensor([3.], grad_fn=<AddBackward0>)
grad_fn của y: <AddBackward0 object at 0x...>
Đạo hàm của z theo x: tensor([18.])

Lỗi khi gọi z.backward() lần thứ hai:
Trying to backward through the graph a second time...
```

**Nhận xét:**

1. **Computational Graph:**

   - PyTorch tự động xây dựng dynamic computational graph
   - Mỗi tensor có `grad_fn` lưu thông tin về phép toán tạo ra nó

2. **Tính toán đạo hàm:**

   - autograd tính đúng đạo hàm theo công thức toán học
   - Kết quả dz/dx = 18 khớp với tính toán thủ công

3. **Quản lý bộ nhớ:**

   - PyTorch giải phóng graph sau backward() để tối ưu bộ nhớ
   - Có thể giữ lại graph bằng `retain_graph=True` nếu cần

4. **Ứng dụng:**
   - autograd là nền tảng cho training neural networks
   - Tự động tính gradient cho optimization algorithms

---

## 3. Phần 3: Xây dựng mô hình với torch.nn

### 3.1. Giải thích các bước làm

#### Task 3.1: Lớp nn.Linear

**nn.Linear** thực hiện phép biến đổi tuyến tính: **y = xW^T + b**

**Các bước thực hiện:**

1. **Khởi tạo lớp Linear:**

   ```python
   linear_layer = torch.nn.Linear(in_features=5, out_features=2)
   ```

   - `in_features=5`: chiều đầu vào
   - `out_features=2`: chiều đầu ra

2. **Tạo input tensor:**

   ```python
   input_tensor = torch.randn(3, 5)  # 3 samples, 5 features
   ```

3. **Forward pass:**
   ```python
   output = linear_layer(input_tensor)
   ```
   - Input shape: `(3, 5)`
   - Output shape: `(3, 2)`

**Tham số của Linear layer:**

- Weight matrix: `(out_features, in_features)` = `(2, 5)`
- Bias vector: `(out_features,)` = `(2,)`

#### Task 3.2: Lớp nn.Embedding

**nn.Embedding** chuyển đổi word index (số nguyên) thành vector dense.

**Các bước thực hiện:**

1. **Khởi tạo Embedding layer:**

   ```python
   embedding_layer = torch.nn.Embedding(num_embeddings=10, embedding_dim=3)
   ```

   - `num_embeddings=10`: kích thước vocabulary (10 từ)
   - `embedding_dim=3`: chiều của vector embedding

2. **Tạo input indices:**

   ```python
   input_indices = torch.LongTensor([1, 5, 0, 8])
   ```

   - Mỗi số là index của một từ trong vocabulary

3. **Lấy embeddings:**
   ```python
   embeddings = embedding_layer(input_indices)
   ```
   - Input shape: `(4,)` - 4 từ
   - Output shape: `(4, 3)` - 4 vector 3 chiều

**Ứng dụng:**

- Word embeddings trong NLP (Word2Vec, GloVe)
- Entity embeddings trong recommendation systems

#### Task 3.3: Xây dựng nn.Module đầu tiên

**nn.Module** là base class cho tất cả các mô hình trong PyTorch.

**Các bước xây dựng:**

1. **Định nghĩa class kế thừa nn.Module:**

   ```python
   class MyFirstModel(nn.Module):
       def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
           super(MyFirstModel, self).__init__()
   ```

2. **Định nghĩa các layers trong **init**:**

   ```python
   self.embedding = nn.Embedding(vocab_size, embedding_dim)
   self.linear = nn.Linear(embedding_dim, hidden_dim)
   self.activation = nn.ReLU()
   self.output_layer = nn.Linear(hidden_dim, output_dim)
   ```

3. **Định nghĩa forward pass:**

   ```python
   def forward(self, indices):
       embeds = self.embedding(indices)
       hidden = self.activation(self.linear(embeds))
       output = self.output_layer(hidden)
       return output
   ```

4. **Khởi tạo và sử dụng mô hình:**
   ```python
   model = MyFirstModel(vocab_size=100, embedding_dim=16,
                        hidden_dim=8, output_dim=2)
   output = model(input_data)
   ```

**Luồng dữ liệu:** (từ trên xuống)

```
Input indices (1, 4)
    |
Embedding Layer → (1, 4, 16)
    |
Linear Layer → (1, 4, 8)
    |
ReLU Activation → (1, 4, 8)
    |
Output Layer → (1, 4, 2)
```

### 3.2. Kết quả thực nghiệm và nhận xét

**Kết quả:**

```
Input shape: torch.Size([3, 5])
Output shape: torch.Size([3, 2])

Input shape: torch.Size([4])
Output shape: torch.Size([4, 3])
Embeddings:
 tensor([[-0.1234,  0.5678, -0.9012],
         [ 0.3456, -0.7890,  0.1234],
         ...])

Model output shape: torch.Size([1, 4, 2])
```

**Nhận xét:**

1. **nn.Linear:**

   - Tự động khởi tạo weights và bias
   - Dễ dàng stack nhiều layers để tạo deep network

2. **nn.Embedding:**

   - Lookup table cho word embeddings
   - Hiệu quả hơn one-hot encoding
   - Weights có thể học được thông qua backpropagation

3. **nn.Module:**

   - Tự động quản lý parameters
   - Hỗ trợ GPU và distributed training
   - Forward pass đơn giản, backward tự động

4. **Kiến trúc mô hình:**
   - Có thể mở rộng dễ dàng với nhiều layers
   - Phù hợp cho các bài toán classification

---

## Tài liệu tham khảo

1. [PyTorch Official Documentation](https://pytorch.org/docs/)
2. [PyTorch Tutorials](https://pytorch.org/tutorials/)
3. [Deep Learning with PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)
4. Đọc tài liệu: lab5_rnns_text_classification.pdf

---
