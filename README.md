# Báo cáo Bài tập 2: Tối ưu hoá với Differential Evolution (DE) và Cross Entropy Method (CEM)

## 1. Thông tin chung
- **Môn học:** Tối ưu hoá / Thuật toán Tiến hoá
- **Sinh viên:** ... (Vui lòng điền tên)
- **MSSV:** 23520108

---

## 2. Giới thiệu & Yêu cầu
Bài tập yêu cầu cài đặt và đánh giá hiệu năng của hai kim chỉ nam phổ biến trong họ thuật toán tiến hoá: **Differential Evolution (DE)** và **Cross Entropy Method (CEM)**. Cả hai thuật toán có mục tiêu là tìm kiếm điểm cực tiểu toàn cục (global minima) cho 5 hàm kiểm thử benchmark đa chiều: Sphere, Griewank, Rosenbrock, Rastrigin và Ackley.

Quy định khắt khe của bài toán đặt ra là: **Bắt buộc sử dụng giới hạn số lần gọi hàm mục tiêu (fitness max_evals) làm điều kiện dừng** để đảm bảo quá trình so sánh cực kỳ công bằng, độc lập với số thế hệ hay kích thước quần thể thay đổi.

---

## 3. Cài đặt các Hàm mục tiêu (Objective Functions)
Các hàm mục tiêu được xây dựng dưới định dạng tính toán vector cho không gian $d$ chiều của Numpy, bao gồm đầy đủ cấu hình cực tiểu (opt_val = 0) và khoảng giới hạn tìm kiếm:
1. **Sphere**: Hàm lồi hình nón cơ bản. Không có local minima. Tọa độ Global Minimum tại trung tâm. Limit: `[-5.12, 5.12]`.
2. **Rastrigin**: Hàm có cực kỳ nhiều local minima phụ thuộc bề mặt dạng sóng cosine, dễ gây "chim mồi" đánh lừa thuật toán. Limit: `[-5.12, 5.12]`.
3. **Rosenbrock**: Thường gọi là hàm thung lũng trái chuối, độ dốc rất nhỏ ở ranh giới cực tiểu. Global Min tại tọa độ $(1, 1, \dots)$. Limit: `[-5.0, 10.0]`.
4. **Ackley**: Hàm gồm mặt phẳng rộng với một cái hố rất sâu và hẹp đổ xuống 0 tại tâm. Limit: `[-32.0, 32.0]`.
5. **Griewank**: Hàm có nhiễu sóng nhỏ nhưng với bề mặt phẳng vĩ mô. Limit: `[-600.0, 600.0]`.

---

## 4. Phương pháp Giải thuật

### 4.1. Thuật toán Differential Evolution (DE)
- Thuật toán DE được setup bằng đột biến thông dụng `rand/1/bin`.
- Tham số siêu thám hiểm được dùng: Đột biến vector $F = 0.8$, Xác suất lai ghép $CR = 0.9$.
- **Cơ chế đếm `max_evals` ngặt nghèo (Strict Evaluations):** Sau bước sinh quần thể gốc ban đầu (`evals = popsize`), thuật toán chạy từng `trial` vector vào hàm đích qua vòng lặp cá thể. Khi đánh giá xong 1 hàm, bộ đếm tăng 1. Ngay khi quỹ `evals` bị chạm mốc giới hạn, một lệnh `break` sẽ đóng quá trình và trả về lời giải tinh túy nhất tại sát biên giới hạn Evaluation.

### 4.2. Thuật toán Cross Entropy Method (CEM) cải tiến
- CEM vận hành dựa trên cơ chế ước lượng phân phối xác suất. Mỗi thế hệ vẽ mẫu (sampling array) từ phân phối chuẩn Gaussian theo một kỳ vọng (Mean: $\mu$) và ma trận hiệp phương sai (tinh giản bằng độ lệch chuẩn `sigma` độc lập trên mỗi chiều).
- **Tuyển chọn Elite:** Hệ thống thiết lập lọc 20% cá thể tinh hoa (`num_elites`). Mean và Std mới trong lưới quần thể sinh ra sẽ được áp đặt dựa hoàn toàn vào 20% cá thể này.
- **Bảo đảm giới hạn Evals:** Thay vì mù quáng sinh kích thước $N$ (`popsize`), CEM được lập trình để check xem quỹ `max_evals` còn bao nhiêu. Nó sinh giới hạn cực đại đúng bằng số `evals` còn dư (`current_popsize = min(N, max_evals - evals)`), khiến việc chạm mốc đếm hoàn hảo 100%.

---

## 5. Kịch Bản Thực Nghiệm và Thống Kê
Toàn cảnh kịch bản vét cạn thực nghiệm:
- Lặp **5** Hàm đa chiều $\times$ **2** không gian số chiều (d=2, d=10) $\times$ **5** kích thước quần thể (N=8, 16, 32, 64, 128) $\times$ **10** lần chạy có Seed độc lập.
- Seed được tự động hóa qua công thức: `Seed = MSSV + i` (ví dụ `23520108 + 0..9`).

**Hệ sinh thái Báo Cáo:**
- **In Bảng Markdown Tóm Tắt:** Script cấu thành một DataFrame qua pandas, in ra bảng tóm tắt với đầy đủ các trụ cột thuật toán. Đồng thời so khớp Independent T-Test qua module `scipy.stats.ttest_ind`.
- **T-Test P-Value:** Nếu độ tự tin của hàm vượt trội > 95% ($p\_value < 0.05$), cột của thuật toán ưu thế được **in đậm (bold)**. Ngược lại đánh giá "Tie" (Hoà) giữa độ hội tụ của CEM và DE.
- **Biểu Đồ Hội Tụ Lịch Sử:** Lưu các chặng `history` có giá trị `best_fit` theo số evaluations. Sử dụng kĩ năng `plt.fill_between` của `matplotlib` mô tả bề dày Standard Deviation - chứng minh sự phân mảnh hay nhất quán giữa những lần lặp bằng vùng bóng đổ màu.

---

## 6. Trực Quan Hoá Động (GIF Animations)
- Kịch bản demo tĩnh: Không gian $d=2$, Số Quần Thể $N=32$, Quỹ hàm $max\_evals=2000$.
- Mã thiết kế vẽ bản đồ contour địa hình không gian bên dưới thông qua $Z$-meshgrid. 
- Thiết lập Neo cực trị (Global Min) bằng điểm hình thoi (Red Diamond).
- Array Tracking: Lưu trữ lưới tọa độ theo từng thế hệ của không gian tìm kiếm, sử dụng `scatter/set_offsets` vẽ đè quần thể và save thành format hình động `.gif` qua plugin `Pillow/matplotlib`. Nhằm phản ánh cực quan khả năng đàn hồi và cụm vào tâm của cả 2 hệ sinh thái thuật toán.

---

## 7. Hướng dẫn chạy môi trường
1. Toàn bộ mã nguồn cốt lõi đã được lưu dưới dạng file duy nhất `BT2_23520108.ipynb`.
2. Khởi động môi trường bằng Jupyter Notebook, JupyterLab hoặc VSCode Jupyter Extension.
3. Ở khung Notebook trên cùng, biến `MSSV` có thể thay đổi bằng MSSV cá nhân.
4. Chọn **Run All Cells / Restart & Run All**.
5. Nhâm nhi tách trà từ 1-2 phút (do bài toán vét cạn hàng tỷ vòng quét array/phân phối Gauss Numpy). Cuối kịch bản sẽ in thành bản log báo cáo T-Test và ảnh GIF tĩnh tự động embed cho bạn!
