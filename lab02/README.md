# lab02
## Lab 2: Count Vectorization
### Các bước triển khai
- Tạo interface `Vectorizer` trong `src/core/interfaces.py`.
- Cài đặt `CountVectorizer` trong `src/representations/count_vectorizer.py`.
- Tạo file `test/test.py` để chạy trên:
  - Corpus mẫu.
  - UD_English-EWT dataset.
### Cách chạy code và ghi log kết quả
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
### Giải thích kết quả
- Corpus nhỏ: Ma trận document-term gọn, dễ hiểu, trực quan.
- Dataset lớn (UD_English-EWT): Vocabulary lên tới hàng chục nghìn token, ma trận rất thưa, tăng độ phức tạp tính toán.
- Bag-of-Words:
+ Ưu điểm: Đơn giản, hiệu quả cho bài toán cơ bản.
+ Hạn chế: Không giữ được ngữ cảnh hoặc thứ tự từ, làm mất thông tin ngữ nghĩa.

## Khó khăn và cách giải quyết
- import module/package: Gây lúng túng lúc đầu và mất khá nhiều thời gian tìm hiểu về lỗi. Kinh nghiệm rút ra được là cần chú ý kỹ về cấu trúc package cũng như đường dẫn
