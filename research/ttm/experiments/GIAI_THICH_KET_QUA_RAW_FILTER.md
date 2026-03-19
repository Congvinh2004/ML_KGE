# 📊 Giải Thích Các Loại Kết Quả trong Link Prediction

## 1. **Raw vs Filter**

### **Raw (Không Filter)**
- **Định nghĩa:** Kết quả **KHÔNG** loại bỏ các triple đã tồn tại trong training/test set khi xếp hạng
- **Đặc điểm:**
  - Model có thể xếp hạng các triple đã biết lên top
  - Kết quả thường **thấp hơn** vì bị "phạt" bởi các triple đã tồn tại
  - **Không phản ánh** khả năng dự đoán triple mới
- **Ví dụ:** Nếu test triple là `(Barack_Obama, born_in, ?)`, model có thể xếp `(Barack_Obama, born_in, Hawaii)` lên top dù triple này đã có trong training set

### **Filter (Đã Filter)**
- **Định nghĩa:** Kết quả **ĐÃ** loại bỏ các triple đã tồn tại trong training/test set khi xếp hạng
- **Đặc điểm:**
  - Chỉ xếp hạng các triple **chưa tồn tại**
  - Kết quả **cao hơn** và **chính xác hơn**
  - **Phản ánh** khả năng dự đoán triple mới
  - **Đây là tiêu chuẩn** để so sánh giữa các model
- **Ví dụ:** Khi test `(Barack_Obama, born_in, ?)`, model sẽ loại bỏ tất cả các triple `(Barack_Obama, born_in, X)` đã có trong training/test set trước khi xếp hạng

## 2. **Left (l) vs Right (r)**

### **Left (l) - Dự đoán Head Entity**
- **Nhiệm vụ:** Cho biết `(?, relation, tail)` → Dự đoán `head entity`
- **Ví dụ:** 
  - Input: `(?, born_in, Hawaii)`
  - Output: Xếp hạng các entities có thể là head (Barack_Obama, ...)

### **Right (r) - Dự đoán Tail Entity**
- **Nhiệm vụ:** Cho biết `(head, relation, ?)` → Dự đoán `tail entity`
- **Ví dụ:**
  - Input: `(Barack_Obama, born_in, ?)`
  - Output: Xếp hạng các entities có thể là tail (Hawaii, ...)

### **Averaged - Trung bình**
- **Định nghĩa:** Trung bình của Left và Right
- **Công thức:** `(left_metric + right_metric) / 2`
- **Lý do:** Đánh giá khả năng dự đoán cả 2 hướng (head và tail)

## 3. **Kết Quả Tổng (Final Results)**

### **Kết quả chính thức: `averaged(filter)`**

Từ code trong `openke/base/Test.h` (dòng 273-277):
```c
mrr = (l_filter_reci_rank+r_filter_reci_rank) / 2;
mr = (l_filter_rank+r_filter_rank) / 2;
hit10 = (l_filter_tot+r_filter_tot) / 2;
hit3 = (l3_filter_tot+r3_filter_tot) / 2;
hit1 = (l1_filter_tot+r1_filter_tot) / 2;
```

**Kết quả cuối cùng được tính từ `averaged(filter)`**, không phải `raw`!

## 4. **Ví Dụ Từ Output Của Bạn**

```
TransT FINAL RESULTS on FB15K237:
MRR: 0.2533
MR: 192.8
Hits@10: 0.4145  ← Đây là kết quả chính!
Hits@3: 0.2744
Hits@1: 0.1719
```

Tương ứng với:
```
averaged(filter): MRR: 0.252431, MR: 193.793808, hit@10: 0.413686, hit@3: 0.274309, hit@1: 0.170795
```

## 5. **Tóm Tắt**

| Loại Kết Quả | Mô Tả | Khi Nào Dùng |
|-------------|-------|--------------|
| **raw** | Không filter | Chỉ để tham khảo, **KHÔNG** dùng để so sánh |
| **filter** | Đã filter | **Tiêu chuẩn** để so sánh giữa các model |
| **l (left)** | Dự đoán head | Để xem chi tiết từng hướng |
| **r (right)** | Dự đoán tail | Để xem chi tiết từng hướng |
| **averaged** | Trung bình left + right | **Kết quả chính thức** |

## 6. **Kết Luận**

✅ **Kết quả tổng = `averaged(filter)`**
- Đây là kết quả được báo cáo trong papers
- Đây là kết quả để so sánh với các model khác
- Đây là kết quả hiển thị trong "FINAL RESULTS"

❌ **KHÔNG dùng `raw` để so sánh**
- Raw chỉ để tham khảo
- Raw không phản ánh khả năng dự đoán triple mới


