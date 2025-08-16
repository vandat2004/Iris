🌸 Ứng dụng Web dự đoán loài hoa Iris bằng KNN
Giới thiệu

Ứng dụng được xây dựng nhằm minh họa cách áp dụng học máy (Machine Learning) vào thực tế.
Người dùng chỉ cần nhập bốn thông số đặc trưng của một bông hoa Iris:

Chiều dài đài hoa (Sepal Length)

Chiều rộng đài hoa (Sepal Width)

Chiều dài cánh hoa (Petal Length)

Chiều rộng cánh hoa (Petal Width)

Hệ thống sẽ sử dụng mô hình đã huấn luyện để dự đoán loài hoa thuộc nhóm nào trong ba loại:
Iris-setosa, Iris-versicolor hoặc Iris-virginica.

Kết quả được hiển thị ngay trên giao diện web, giúp người dùng quan sát một cách trực quan và dễ hiểu.

Công nghệ sử dụng

Flask: Framework web nhẹ trong Python, dùng để triển khai server và định nghĩa các route xử lý dữ liệu.

Scikit-learn: Thư viện học máy, dùng để xây dựng và huấn luyện mô hình K-Nearest Neighbors (KNN).

Pandas: Hỗ trợ xử lý dữ liệu đầu vào từ file hoặc dataset.

HTML/CSS: Tạo giao diện nhập liệu và hiển thị kết quả dự đoán.

Thuật toán

Ứng dụng sử dụng thuật toán K-Nearest Neighbors (KNN) với các bước chính:

Với một mẫu mới (dữ liệu người dùng nhập), tính khoảng cách Euclid đến tất cả các mẫu trong tập huấn luyện.

Chọn ra k điểm gần nhất (ở đây chọn k = 5).

Dự đoán nhãn của mẫu mới dựa trên đa số phiếu trong k hàng xóm đó.

Nhờ cách hoạt động đơn giản nhưng hiệu quả, KNN thường được dùng cho các bài toán phân loại cơ bản như bộ dữ liệu Iris.

Ngôn ngữ lập trình

Python: để xử lý dữ liệu, huấn luyện mô hình, kết nối giữa backend và giao diện web.

HTML/CSS: để xây dựng form nhập dữ liệu và phần hiển thị kết quả cho người dùng.

Trải nghiệm người dùng

Người dùng mở ứng dụng web → nhập 4 tham số đặc trưng của hoa vào form.

Nhấn nút Dự đoán.

Hệ thống xử lý dữ liệu, chạy mô hình KNN và trả về kết quả ngay lập tức:
👉 Ví dụ: “Loài hoa dự đoán: Iris-versicolor”.

<img width="330" height="361" alt="{247E370C-560F-4EAF-8155-B82ACC2F108C}" src="https://github.com/user-attachments/assets/f84c173f-a8e6-4ced-abe3-70b28a6b8b10" />

