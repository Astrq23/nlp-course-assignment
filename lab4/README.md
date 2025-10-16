#  Lab 04 – Word Embeddings

**Họ và tên:** Nguyễn Hoàng Việt  
 

---

##  Task 1 – Cài đặt và chuẩn bị dữ liệu
**Mục tiêu:**  
Chuẩn bị môi trường Python, cài đặt thư viện (`gensim`, `nltk`, `pyspark`...), và đọc dữ liệu văn bản từ tập huấn luyện (ví dụ `en_ewt-ud-train.txt`).

**Thực hiện:**  
- Đọc dữ liệu thô, loại bỏ ký tự đặc biệt, tách từ bằng `nltk`.  
- Lưu danh sách câu để huấn luyện.  

**Nhận xét:**  
Dữ liệu được xử lý thành công, các câu văn bản đã sẵn sàng cho mô hình embedding.  

---

##  Task 2 – Huấn luyện Word2Vec bằng Gensim
**Mục tiêu:**  
Huấn luyện mô hình Word2Vec cơ bản bằng thư viện **Gensim** trên tập dữ liệu đã xử lý.

**Thực hiện:**  
- Dùng `gensim.models.Word2Vec` với tham số: `vector_size=100`, `window=5`, `min_count=2`.  
- Lưu và tải lại mô hình.  

**Nhận xét:**  
Mô hình Word2Vec được huấn luyện thành công, các từ phổ biến có vector biểu diễn ổn định.  
Ví dụ: từ “computer” có các từ gần nghĩa như “machine”, “software”, “device”.  

---

##  Task 3 – Huấn luyện trên tập dữ liệu nhỏ
**Mục tiêu:**  
Kiểm tra hoạt động của mô hình trên tập dữ liệu nhỏ để hiểu cơ chế huấn luyện và kiểm tra độ tương đồng.  

**Thực hiện:**  
- Huấn luyện Word2Vec với dữ liệu nhỏ (vài câu).    

**Nhận xét:**  
Mô hình chạy nhanh, nhưng kết quả chưa ổn định vì dữ liệu nhỏ.  
---

##  Task 4 – Huấn luyện model trên tập dữ liệu lớn (Spark)
**Mục tiêu:**  
Ứng dụng **PySpark MLlib Word2Vec** để huấn luyện mô hình trên tập dữ liệu lớn, khai thác sức mạnh tính toán phân tán.  

**Thực hiện:**  
- Dùng `SparkSession` để đọc file JSON lớn.  
- Tiền xử lý văn bản (lowercase, bỏ ký tự đặc biệt, tách từ bằng `split`).  
- Huấn luyện `Word2Vec` của `pyspark.ml.feature`.  
- In ra các từ gần nghĩa của “computer”.  

**Nhận xét:**  
Huấn luyện thành công trên Spark, mô hình hoạt động ổn định, phù hợp với dữ liệu lớn.  
So với Gensim, Spark Word2Vec cho phép xử lý tập dữ liệu nặng và song song.  

---

##  Task 5 – Trực quan hóa Embedding
**Mục tiêu:**  
Giảm chiều và trực quan hóa các vector từ bằng **PCA** và **t-SNE** để quan sát mối quan hệ ngữ nghĩa.  

**Thực hiện:**  
- Lấy vector của các từ mẫu.  
- Giảm chiều bằng `PCA` và `TSNE`.  
- Vẽ hai biểu đồ so sánh (PCA và t-SNE).  

**Nhận xét:**  
- PCA cho thấy cấu trúc tuyến tính tổng quát.  
- t-SNE thể hiện rõ hơn các cụm từ có ý nghĩa tương tự (ví dụ: “king–queen”, “paris–france”).  
- Trực quan hóa giúp đánh giá chất lượng embedding một cách trực quan.  

---

##  Tổng kết
- Hiểu được quy trình **tiền xử lý văn bản → huấn luyện embedding → trực quan hóa**.  
- Nắm vững cách dùng **Word2Vec** trong cả môi trường nhỏ (Gensim) và lớn (Spark).  
- Thấy rõ vai trò của embedding trong việc biểu diễn ngữ nghĩa từ vựng.  
