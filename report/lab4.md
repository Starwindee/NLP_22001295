# Report - Lab 4: Word Embeddings

## 1. Task 1 + 2: Tải và sử dụng model có sẵn (gensim), và nhúng câu/văn bản

### 1.1. Giải thích các bước thực hiện

**Bước 1: Cài đặt thư viện gensim**

```bash
pip install gensim
```

**Bước 2: Triển khai class WordEmbedder**

- Tạo file `src/representations/word_embedder.py`
- Sử dụng `gensim.downloader` để tải pretrained model
- Triển khai các phương thức chính:
  - `get_vector()`: Lấy vector từ một từ
  - `get_similarity()`: Tính độ tương đồng cosine giữa hai từ
  - `most_similar()`: Tìm top-n từ đồng nghĩa
  - `embed_document()`: Nhúng một câu/văn bản thành vector trung bình của các từ
    - Bước đầu tiên: Tokenize câu thành các từ
    - Sau đó lấy vector từng từ và tính trung bình

**Bước 3: Test class WordEmbedder**

- Tạo file `test/lab4_test.py`
- Viết các test case để kiểm tra các phương thức `get_vector()`, `get_similarity()`, `most_similar()`, với:
  - Pretrained model: `glove-wiki-gigaword-50`
  - Lấy vector 1 từ: `king`
  - Tính độ tương đồng giữa 2 từ: `king` và `queen`; `king` và `man`
  - Tìm top-10 từ đồng nghĩa với `computer`
  - Nhúng câu mẫu: "The cat is sleeping peacefully on the sunny windowsill."

### 1.2. Kết quả thực nghiệm

**Kết quả lấy vector từ 'king':**

```markdown
Vector for 'king':
[-0.32307 -0.87616 0.21977 0.25268 0.22976 0.7388 -0.37954
-0.35307 -0.84369 -1.1113 -0.30266 0.33178 -0.25113 0.30448
-0.077491 -0.89815 0.092496 -1.1407 -0.58324 0.66869 -0.23122
-0.95855 0.28262 -0.078848 0.75315 0.26584 0.3422 -0.33949
0.95608 0.065641 0.45747 0.39835 0.57965 0.39267 -0.21851
0.58795 -0.55999 0.63368 -0.043983 -0.68731 -0.37841 0.38026
0.61641 -0.88269 -0.12346 -0.37928 -0.38318 0.23868 0.6685
-0.43321 -0.11065 0.081723 1.1569 0.78958 -0.21223 -2.3211
-0.67806 0.44561 0.65707 0.1045 0.46217 0.19912 0.25802
0.057194 0.53443 -0.43133 -0.34311 0.59789 -0.58417 0.068995
0.23944 -0.85181 0.30379 -0.34177 -0.25746 -0.031101 -0.16285
0.45169 -0.91627 0.64521 0.73281 -0.22752 0.30226 0.044801
-0.83741 0.55006 -0.52506 -1.7357 0.4751 -0.70487 0.056939
-0.7132 0.089623 0.41394 -1.3363 -0.61915 -0.33089 -0.52881
0.16483 -0.98878 ]
```

**Kết quả tính độ tương đồng giữa 'king' và 'queen':**

```markdown
Similarity between 'king' and 'queen': 0.7507691
```

- **Nhận xét:** Độ tương đồng cao (gần 1) cho thấy `king` và `queen` có ngữ nghĩa liên quan chặt chẽ (có thể là quan hệ ngữ nghĩa về royalty), phù hợp với thực tế.

```markdown
Similarity between 'king' and 'man': 0.5118681
```

- **Nhận xét:** Độ tương đồng trung bình (khoảng 0.5) cho thấy `king` và `man` có liên hệ ngữ nghĩa (đều chỉ giới tính nam hoặc vai trò nam giới), nhưng ít liên quan hơn so với cặp `king` – `queen`.

**Kết quả tìm top-10 từ đồng nghĩa với 'computer':**

```markdown
Most similar words to 'computer':
[('computers', 0.8751984238624573),
('software', 0.8373122215270996),
('technology', 0.7642159461975098),
('pc', 0.7366448640823364),
('hardware', 0.7290390729904175),
('internet', 0.7286775708198547),
('desktop', 0.7234441637992859),
('electronic', 0.7221828699111938),
('systems', 0.7197922468185425),
('computing', 0.7141730785369873)]
```

- **Nhận xét:** Các từ đồng nghĩa như `computers`, `software`, `technology` đều liên quan đến lĩnh vực máy tính - công nghệ, cho thấy model đã học được các mối quan hệ ngữ nghĩa tốt.

**Kết quả nhúng câu mẫu:**

```markdown
Document embedding for sentence:
[-1.70427799e-01 6.63375929e-02 5.12149990e-01 -7.62629956e-02
-8.57415944e-02 3.90449971e-01 -1.14693180e-01 3.94869983e-01
-1.71310917e-01 -8.89820978e-02 4.34519984e-02 9.17638987e-02
4.00394022e-01 -1.04710601e-01 2.23587081e-01 -2.47156814e-01
1.31761715e-01 -2.19885968e-02 -1.74138278e-01 -3.03126991e-01
2.24873587e-01 -8.02365988e-02 3.50619167e-01 2.07757987e-02
3.77558589e-01 9.78751034e-02 -1.27796710e-01 -3.90764982e-01
-2.18382001e-01 5.06768823e-02 -2.00227693e-01 1.99799612e-01
2.57236093e-01 2.48154193e-01 4.51723859e-02 3.94370615e-01
1.09889671e-01 7.51530007e-02 2.60905921e-01 -4.23067570e-01
-2.79865444e-01 -9.85172987e-02 1.34854913e-01 -1.52658999e-01
1.57997012e-04 9.44599509e-03 -2.37888005e-02 -4.61669974e-02
-3.66334617e-02 -3.32768977e-01 -6.49879649e-02 -2.20051005e-01
2.13972211e-01 9.41075027e-01 -2.34671995e-01 -2.00726390e+00
-2.07846016e-02 -8.40370879e-02 1.13496304e+00 2.19453007e-01
-1.29831001e-01 7.78317988e-01 -1.96782395e-01 -1.82390064e-02
6.06468379e-01 2.16688007e-01 3.04827392e-01 -7.17839971e-02
2.08125059e-02 -2.14456707e-01 -1.92906693e-01 -2.35222384e-01
6.83334619e-02 -3.45234901e-01 1.87417731e-01 3.10201257e-01
-1.88526705e-01 -9.16182026e-02 -5.36607563e-01 1.50859401e-01
3.33710998e-01 1.34376347e-01 -3.52062017e-01 2.67804086e-01
-6.88329160e-01 -2.18072012e-01 2.81905923e-02 -1.41832884e-02
-1.33834496e-01 -1.31722599e-01 1.30415084e-02 -1.97637796e-01
1.51704894e-02 4.91348803e-01 -4.83259439e-01 1.49025887e-01
-3.46081406e-01 -1.21453404e-01 5.29408932e-01 1.01277888e-01]
```

---

## 2. Task 3: Tự train word2vec model trên corpus nhỏ (Gensim)

### 2.1. Giải thích các bước thực hiện

**Bước 1:** Tạo file `test/lab4_embedder_training_demo.py` để demo việc train word2vec model

**Bước 2: Chuẩn bị corpus nhỏ**

- Sử dụng file `UD_English-EWT/en_ewt-ud-train.txt` lấy từ lab 1
- Viết hàm `read_sentences()` để đọc file và yield từng câu đã tokenized

**Bước 3: Train word2vec model**

- Sử dụng `gensim.models.Word2Vec` để train model trên tập dữ liệu
- Thiết lập các tham số:
  - `vector_size=100`
  - `window=5`
  - `min_count=2`
  - `workers=4`
  - `epochs=10`
- Lưu model đã train vào file `results/word2vec_ewt.model`

**Bước 4: Test model đã train**

- Test tìm 5 từ đồng nghĩa với từ `dog`
- Giải bài toán quan hệ từ: `king` - `man` + `woman` = ?

### 2.2. Kết quả thực nghiệm

**Kết quả tìm 5 từ đồng nghĩa với 'dog':**

```markdown
Top 5 words similar to 'dog':
car: 0.9961
walk: 0.9953
house: 0.9951
drive: 0.9948
move: 0.9948
```

- **Nhận xét:** Các từ đồng nghĩa như `car`, `walk`, `house` đều không thực sự đồng nghĩa với `dog` về mặt ngữ nghĩa. Điều này có thể do corpus nhỏ và không đủ đa dạng để học được các mối quan hệ ngữ nghĩa chính xác.

**Kết quả giải bài toán quan hệ từ:**

```markdown
Analogy example (king - man + woman): ('job', 0.991863489151001)
```

- **Nhận xét:** Kết quả trả về là `job` với độ tương đồng cao 0.9918, nhưng `job` không liên quan rõ ràng đến quan hệ giữa `king`, `man`, và `woman`. Điều này cho thấy model chưa học được các mối quan hệ ngữ nghĩa phức tạp do corpus hạn chế (vocab = 11,290 từ).

---

## 3. Advance Task - Task 4: Huấn luyện model trên tập dữ liệu lớn (Spark)

### 3.1. Giải thích các bước thực hiện

**Bước 1: Cài đặt Apache Spark và PySpark**

- Tải và cài đặt Apache Spark từ trang chủ
- Cài đặt PySpark qua pip:

```bash
pip install pyspark
```

**Bước 2:** Tạo file `test/lab4_spark_word2vec_demo.py` để demo việc train word2vec model sử dụng Spark

**Bước 3:** Chuẩn bị tập dữ liệu lớn: sử dụng tập dữ liệu `c4-train.00000-of-01024-30K.json`

**Bước 4: Triển khai**

- Khởi tạo SparkSession
- Đọc và tiền xử lý dữ liệu:
  - Đọc file JSON vào DataFrame
  - Lấy cột `text` và tokenize thành các câu
  - Tiền xử lý: chuyển về chữ thường, loại bỏ ký tự đặc biệt, tách token theo khoảng trắng
- Huấn luyện Word2Vec model:
  - Sử dụng `pyspark.ml.feature.Word2Vec` để huấn luyện model trên tập dữ liệu đã tokenized
  - Thiết lập các tham số:
    - `vectorSize=100`
    - `minCount=5`
- Lưu model đã huấn luyện vào thư mục `results/spark_word2vec_model`

**Bước 5: Test model đã huấn luyện**

- Tìm 5 từ đồng nghĩa với từ `computer`

### 3.2. Kết quả thực nghiệm

**Kết quả tìm 5 từ đồng nghĩa với 'computer':**

```markdown
Finding synonyms for 'computer':
+---------+------------------+
|word |similarity |
+---------+------------------+
|desktop |0.7037132382392883|
|computers|0.6869197487831116|
|uwowned |0.6821119785308838|
|device |0.654563844203949 |
|laptop |0.6494661569595337|
+---------+------------------+
```

- **Nhận xét:** Các từ đồng nghĩa như `desktop`, `computers`, `device`, và `laptop` đều liên quan chặt chẽ đến `computer`, cho thấy model đã học được các mối quan hệ ngữ nghĩa tốt từ tập dữ liệu lớn.

---

## 4. Task 5: Trực quan hóa Embeddings

### 4.1. Giải thích các bước thực hiện

**Bước 1:** Tạo notebook `notebook/lab4_visualize.ipynb`

**Bước 2:** Tải GloVe embeddings (50 chiều): `glove.6B.50d.txt`

**Bước 3: Chuyển embeddings sang DataFrame**

- Đọc file GloVe và chuyển sang DataFrame với các cột `word` và `vector` (dạng list)

**Bước 4: Sử dụng PCA để giảm chiều từ 50 xuống 2**

- Sử dụng `sklearn.decomposition.PCA` để giảm chiều vector

**Bước 5: Trực quan hóa với matplotlib**

- Vẽ scatter plot các từ trên mặt phẳng 2D
- Gắn nhãn các từ để dễ nhận diện

### 4.2. Kết quả trực quan hóa

Kết quả trực quan hóa embeddings nằm trong notebook `notebook/lab4_visualize.ipynb`

### 4.3. Hướng dẫn chạy notebook

**Cài đặt các thư viện cần thiết:**

```bash
pip install pandas numpy matplotlib scikit-learn
```

**Các bước chạy:**

1. Thay đổi đường dẫn `glove_file` trong notebook thành đường dẫn tới file `glove.6B.50d.txt` trên máy bạn
2. Chạy từng cell trong notebook để thực hiện các bước từ tải dữ liệu, xử lý, giảm chiều, và trực quan hóa

---

## 5. Một số vấn đề gặp phải

**Vấn đề:** Load toàn bộ GloVe file (6GB) vào RAM

**Giải pháp:** Chỉ load một phần (ví dụ 20,000 từ phổ biến nhất để demo)

---

## 6. Tài liệu tham khảo

- Gensim Documentation: https://radimrehurek.com/gensim/
- PySpark Documentation: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Word2Vec.html
- GloVe Embeddings: https://nlp.stanford.edu/projects/glove/
