## Lab 2: Triển khai Count Vectorization
### Quy trình thực hiện
- Định nghĩa một interface có tên `Vectorizer` tại `src/core/interfaces.py`.
- Xây dựng lớp `CountVectorizer` tại `src/representations/count_vectorizer.py`.
- Chuẩn bị tệp `test/test.py` để kiểm thử với:
  - Một bộ corpus mẫu nhỏ.
  - Bộ dữ liệu UD_English-EWT.
### Hướng dẫn thực thi và ghi nhận kết quả
  ```
    python -m test.test
  ```
### Kết quả chạy
```
--------------------------------------------------

Vocabulary: {'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}
Document-Term Matrix:
[1, 0, 0, 1, 0, 1, 1, 0, 0, 0]
[1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
[1, 1, 1, 0, 1, 0, 1, 1, 0, 1]

--------------------------------------------------

TRANSFORM EXAMPLE LINES FROM UD_English-EWT INTO VECTOR FORM
Size of vocabulary: 65
Tokens in vocabulary: ['!', ',', '-', '.', '2', '3', ':', '[', ']', 'a', 'abdullah', 'al', 'american', 'ani', 'announced', 'at', 'authorities', 'baghdad', 'be', 'being', 'border', 'busted', 'by', 'causing', 'cells', 'cleric', 'come', 'dpa', 'for', 'forces', 'had', 'in', 'interior', 'iraqi', 'killed', 'killing', 'ministry', 'moi', 'mosque', 'near', 'of', 'officials', 'operating', 'preacher', 'qaim', 'respected', 'run', 'shaikh', 'syrian', 'terrorist', 'that', 'the', 'them', 'they', 'this', 'to', 'town', 'trouble', 'two', 'up', 'us', 'were', 'will', 'years', 'zaman']
Document-Term Matrix:
[0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
[0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
```
### Phân tích kết quả
- Với corpus mẫu (nhỏ): Ma trận thể hiện (document-term) có cấu trúc đơn giản, dễ dàng quan sát và nắm bắt.
- Với bộ dữ liệu lớn (UD_English-EWT): Kích thước từ vựng (vocabulary) tăng lên đáng kể, dẫn đến ma trận biểu diễn rất thưa (sparse matrix) và làm gia tăng chi phí tính toán.
- Nhận xét về Bag-of-Words (BoW):
  - **Ưu điểm**: Là phương pháp đơn giản và mang lại hiệu quả đối với các tác vụ phân loại văn bản cơ bản.
  - **Nhược điểm**: Bỏ qua thông tin về trật tự từ và ngữ cảnh, dẫn đến việc làm mất mát thông tin ngữ nghĩa quan trọng.

## Thách thức và giải pháp
- **Vấn đề**: Việc `import` các module/package ban đầu gây nhiều bối rối và tốn thời gian để tìm hiểu, khắc phục lỗi.
- **Giải pháp/Kinh nghiệm**: Cần phải đặc biệt chú ý đến cấu trúc thư mục (package) và cách thiết lập đường dẫn (path) sao cho chính xác.
