# Report lab 5: Text_Classification

## 1. Task 1 + 2: Data Preparation (with scikit-learn) + Task 2: Basic Test Case

### 1.1 Giải thích các bước thực hiện

**Task 1: Scikit-learn TextClassifier**

- Tạo file `src/models/text_classifier.py`.
- Triển khai class `TextClassifier` với các phương thức chính:
  - `fit(X_train, y_train`: Huấn luyện mô hình LogisticRegression trên dữ liệu huấn luyện.
  - `predict(X_test)`: Dự đoán nhãn cho dữ liệu kiểm thử.
  - `evaluate(X_test, y_test)`: Tính toán các chỉ số đánh giá (accuracy, precision, recall, f1-score).

**Task 2: Basic Test Case**

- Tạo file `test/lab5_test.py`.
- Viết hàm `load_data()` để tạo dữ liệu văn bản và nhãn mẫu.
- Chia dữ liệu thành tập huấn luyện và kiểm thử bằng `train_test_split`.
- Khởi tạo tokenizer (`RegexTokenizer`) và vectorizer (`CountVectorizer`).
- Khởi tạo class `TextClassifier` với vectorizer.
- Huấn luyện mô hình với phương thức `fit`.
- Dự đoán nhãn trên tập kiểm thử với phương thức `predict`.
- Đánh giá kết quả dự đoán bằng phương thức `evaluate`
- In ra kết quả dự đoán và các chỉ số đánh giá.

### 1.2. Kết quả thực nghiệm và nhận xét

**Kết quả thực nghiệm:**

- Kết quả dự đoán: `Predictions: [1, 1]`
- Các chỉ số đánh giá:
  - `accuracy: 0.5000`
  - `precision: 0.2500`
  - `recall: 0.5000`
  - `f1_score: 0.3333`

**Nhận xét:**

- Mô hình đã huấn luyện và dự đoán thành công trên tập kiểm thử.
- Độ chính xác đạt 50%, các chỉ số precision, recall, f1-score ở mức trung bình, chưa cao.
- Do dữ liệu chỉ là 6 câu đánh giá phim (với nhãn tích cực/tiêu cực), số lượng mẫu nhỏ và nội dung đơn giản nên chưa phản ánh đầy đủ hiệu quả của mô hình.

## 2. Task 3: Task 3: Running the Spark Example

### 2.1. Giải thích các bước thực hiện

- Tạo file `test/lab5_spark_sentiment_analysis.py` để thực hiện phân loại cảm xúc với Spark ML.
- Viết hàm `load_data(spark, data_path)`:
  - Đọc dữ liệu từ file CSV bằng Spark DataFrame.
  - Loại bỏ các dòng thiếu nhãn sentiment.
  - Chuyển nhãn sentiment sang label nhị phân (0/1).
- Viết hàm `build_pipeline()`:
  - Khởi tạo các bước xử lý dữ liệu gồm:
    - `Tokenizer`: tách trường văn bản thành các từ.
    - `StopWordsRemover`: loại bỏ các từ dừng.
    - `HashingTF`: chuyển danh sách từ thành vector đặc trưng bằng phương pháp hashing.
    - `IDF`: chuẩn hóa trọng số đặc trưng bằng phương pháp TF-IDF.
    - `LogisticRegression`: xây dựng mô hình phân loại cảm xúc.
  - Gom các bước trên vào một pipeline để thực hiện tuần tự.
- Viết hàm `evaluate_model(predictions)`:
  - Đánh giá mô hình bằng các chỉ số như accuracy và f1-score dựa trên kết quả dự đoán.
- Trong hàm `main()`:
  - Khởi tạo SparkSession.
  - Đọc dữ liệu và tiền xử lý bằng hàm `load_data`.
  - Chia dữ liệu thành tập huấn luyện và kiểm thử bằng `randomSplit`.
  - Xây dựng pipeline và huấn luyện mô hình trên tập train.
  - Dự đoán trên tập test và đánh giá kết quả bằng hàm `evaluate_model`.
  - Đóng SparkSession sau khi hoàn thành.

### 2.2. Kết quả thực nghiệm và nhận xét

**Kết quả thực nghiệm:**

- Script thực thi thành công, Spark session khởi tạo và xử lý dữ liệu đúng quy trình.
- Kết quả đánh giá trên tập kiểm thử:
  - `Accuracy: 0.7277`
  - `F1 Score: 0.7245`

**Nhận xét:**

- Mô hình LogisticRegression trong pipeline Spark ML đạt độ chính xác và f1-score khá tốt trên dữ liệu sentiment.csv.
- Quy trình tiền xử lý và huấn luyện tự động hóa giúp xử lý dữ liệu lớn hiệu quả.
- Kết quả cho thấy Spark ML phù hợp cho các bài toán phân loại cảm xúc với dữ liệu quy mô lớn.

## 3. Task 4: Task 4: Model Improvement Experiment

### 3.1 Giải thích các bước thực hiện

- Tạo file `test/lab5_improvement_test.py` để thực hiện các thí nghiệm cải tiến mô hình phân loại cảm xúc với dữ liệu từ `https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment`.
- Viết hàm `load_data(spark, path)`:
  - Đọc dữ liệu từ file CSV bằng Spark DataFrame.
  - Loại bỏ các dòng thiếu trường "text" hoặc "label".
  - Chỉ giữ lại các dòng có giá trị label hợp lệ (0, 1, 2).
  - Ép kiểu label sang integer để phù hợp với các mô hình phân loại.
  - Làm sạch văn bản: chuyển về chữ thường, loại bỏ link, tag HTML, ký tự đặc biệt và khoảng trắng thừa.
- Khởi tạo các bước tiền xử lý chung:
  - `RegexTokenizer`: tách văn bản thành các từ.
  - `StopWordsRemover`: loại bỏ stopwords khỏi danh sách từ.
- Thử nghiệm nhiều mô hình và đặc trưng khác nhau:
  1. **TF-IDF + Logistic Regression**: Sử dụng HashingTF và IDF để sinh đặc trưng, huấn luyện với Logistic Regression.
  2. **CountVectorizer + Naive Bayes**: Sử dụng CountVectorizer để sinh đặc trưng, huấn luyện với Naive Bayes.
  3. **Word2Vec + Logistic Regression**: Sử dụng Word2Vec để sinh đặc trưng, huấn luyện với Logistic Regression.
  4. **Word2Vec + Random Forest**: Sử dụng Word2Vec để sinh đặc trưng, huấn luyện với Random Forest.
- Viết hàm `evaluate_model(name, pipeline, train_df, valid_df)` để huấn luyện từng pipeline, dự đoán và đánh giá bằng các chỉ số accuracy, f1-score.
- In ra kết quả so sánh các mô hình để lựa chọn phương pháp tối ưu.
- Đóng Spark session sau khi hoàn thành.

### 3.2. Kết quả thực nghiệm và nhận xét

**Kết quả thực nghiệm:**

- `TF-IDF + Logistic Regression: Accuracy 0.6766, F1 0.6821`
- `CountVectorizer + Naive Bayes: Accuracy 0.7896, F1 0.7861`
- `Word2Vec + Logistic Regression: Accuracy 0.6771, F1 0.5952`
- `Word2Vec + Random Forest: Accuracy 0.7045, F1 0.6360`

**Nhận xét:**

- Mô hình tốt nhất: CountVectorizer + Naive Bayes (Accuracy 0.7896, F1 0.7861). Đây là mô hình đơn giản nhưng phù hợp nhất với dữ liệu text dạng tweet, cân bằng giữa precision và recall.

- TF-IDF + Logistic Regression: hiệu quả trung bình, F1 hơi cao hơn accuracy nhưng kém hơn Naive Bayes.

- Word2Vec + Logistic Regression: F1 thấp (0.5920), không tận dụng tốt embedding, Logistic Regression quá đơn giản cho không gian vector phức tạp.

- Word2Vec + Random Forest: cải thiện so với LR nhưng vẫn kém Naive Bayes; cho thấy Random Forest khai thác embedding tốt hơn nhưng chưa tối ưu.

### 4. Vấn đề gặp phải:

Tại task 4: với dữ liệu từ hugging face:

- Dữ liệu bị nhiễu trong file CSV: Các tweet chứa link, khiến trình đọc CSV bị lệch cột, label thỉnh thoảng nhận toàn bộ text tweet thay vì giá trị số.
- Cách xử lý: Lọc lại các dòng chỉ giữ label hợp lệ (0, 1, 2) và ép kiểu label sang số nguyên để đảm bảo dữ liệu đầu vào đúng định dạng:
  - Từ (Train: 9938, Validation: 2486) -> (Train: 9479, Validation: 2372)

## 5. Hướng dẫn chạy

**Hướng dẫn chạy:**

1. Cài đặt các thư viện cần thiết bằng lệnh:

```bash
pip install -r requirements.txt
```

2. Chuẩn bị dữ liệu đầu vào ở các đường dẫn phù hợp, ví dụ:
   - `data/twitter-financial-news-sentiment/sent_train.csv`
   - `data/twitter-financial-news-sentiment/sent_valid.csv`
   - `data/sentiments.csv`
3. Chạy script bằng lệnh sau trong terminal:

   **Task 2:**

```bash
python -m test.lab5_test
```

**Task 3:**

```bash
python -m test.lab5_spark_sentiment_analysis
```

**Task 4:**

```bash
python -m lab5_improvement_test
```

## 6. Tài liệu tham khảo

**Nguồn tham khảo và công cụ sử dụng:**

- Thư viện [PySpark](https://spark.apache.org/docs/latest/api/python/).
- Các mô hình và công cụ của Spark ML: `LogisticRegression`, `NaiveBayes`, `RandomForestClassifier`, `HashingTF`, `IDF`, `Word2Vec`, `CountVectorizer`, `RegexTokenizer`, `StopWordsRemover`.
- Bộ dữ liệu: `twitter-financial-news-sentiment` (cho task 4)
- Các hàm xử lý dữ liệu của Spark SQL: `col`, `lower`, `regexp_replace`.
