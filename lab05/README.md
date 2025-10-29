# Lab 5: Phân tích Cảm xúc Văn bản với PySpark

Dự án này thực hiện và so sánh các pipeline PySpark để phân tích cảm xúc. Trọng tâm là đánh giá hiệu suất của mô hình baseline so với các kỹ thuật cải tiến trên một bộ dữ liệu lớn.

So sánh 3 phương pháp:
1.  **Baseline:** TF-IDF + Logistic Regression
2.  **Cải tiến 1:** TF-IDF + Naive Bayes
3.  **Cải tiến 2:** Word2Vec + Logistic Regression

---

## Báo cáo và Phân tích Lab 5

### 1. Giải thích các bước thực thi

1.  **Chuẩn bị dữ liệu:** Tải dataset `zeroshot/twitter-financial-news-sentiment` (Hugging Face) bằng script `prepare_dataset.py`, vì bộ dữ liệu 6 mẫu trong lab quá nhỏ.
2.  **Tải và Tiền xử lý:** Tiền xử lý dữ liệu với PySpark trong `test/lab5_spark_advanced_analysis.py`: làm sạch (xóa URL, @mentions, nhiễu), xử lý các tweet nhiều dòng, và chia 80/20 (train/test).
3.  **Xây dựng & Đánh giá Pipeline:** Xây dựng 3 pipeline (Baseline, Cải tiến 1, Cải tiến 2). Huấn luyện trên tập train và đánh giá trên tập test bằng Accuracy và F1-Score.

### 2. Hướng dẫn chạy mã nguồn

1.  **Cài đặt thư viện:**
    ```bash
    pip install pyspark pandas scikit-learn datasets
    ```
2.  **Chuẩn bị dữ liệu (1 lần):**
    ```bash
    python prepare_dataset.py
    ```
3.  **Chạy phân tích (Chính):**
    ```bash
    python test/lab5_spark_advanced_analysis.py
    ```
4.  **Xem kết quả:** Bảng so sánh hiệu suất sẽ được in ra ở cuối console.

### 3. Phân tích kết quả

| Mô hình | Accuracy | F1-Score |
| :--- | :--- | :--- |
| **Baseline (TF-IDF + LR)** | **0.7273** | **0.7263** |
| **Improvement (TF-IDF + NB)** | 0.6881 | 0.7010 |
| **Improvement (Word2Vec + LR)** | 0.6678 | 0.5621 |

---

**Phân tích:**
Kết quả cho thấy **các phương pháp "cải tiến" không phải lúc nào cũng hoạt động tốt hơn baseline nếu không được tinh chỉnh cẩn thận.**

1.  **Baseline (TF-IDF + LR) là tốt nhất:** Mô hình `LogisticRegression` (LR) là một baseline tuyến tính mạnh mẽ. Kết hợp với `TF-IDF`, nó cho hiệu quả tốt nhất vì tần suất từ khóa là chỉ báo mạnh mẽ cho văn bản ngắn (tweet).
2.  **Tại sao `Naive Bayes` (NB) kém hơn?** `NaiveBayes` hoạt động dựa trên giả định "ngây thơ" về tính độc lập của các từ, điều này không đúng với ngôn ngữ tự nhiên.
3.  **Tại sao `Word2Vec` kém nhất?** Rất có thể do việc **lấy trung bình** vector câu đã làm mất đi các sắc thái quan trọng. Hơn nữa, các giá trị siêu tham số mặc định rõ ràng là chưa tối ưu.

### 4. Thử thách và giải pháp
  **Vấn đề:** Các mô hình "cải tiến" cho kết quả kém hơn baseline.
  **Giải pháp:** Phân tích nguyên nhân kỹ thuật (như trong Mục 3) thay vì cho rằng đã làm sai.

### 5. Tài liệu tham khảo

* Tài liệu hướng dẫn `lab5_text_classification.pdf`.
* Tài liệu tiêu chí `criteria.pdf`.
* Bộ dữ liệu: `zeroshot/twitter-financial-news-sentiment` (Nguồn: Hugging Face).
* Thư viện: `pyspark`, `pandas`, `datasets` (Hugging Face), `scikit-learn`.
