# Bài Tập 2: So sánh Differential Evolution và Cross Entropy Method

**Môn học:** Mạng Neural và Thuật giải Di truyền - CS410.Q2  
**Mã sinh viên:** 23520108  
**Ngày:** 11 tháng 4 năm 2026

---

## Mục lục
1. [Giới thiệu](#giới-thiệu)
2. [Tổng quan Thuật toán](#tổng-quan-thuật-toán)
3. [Chi tiết Cài đặt](#chi-tiết-cài-đặt)
4. [Thiết kế Thử nghiệm](#thiết-kế-thử-nghiệm)
5. [Kết quả và Phân tích](#kết-quả-và-phân-tích)
6. [Kết luận](#kết-luận)

---

## Giới thiệu

Bài tập này cài đặt và so sánh hai thuật toán tối ưu hóa siêu heuristic phổ biến: **Differential Evolution (DE)** và **Cross Entropy Method (CEM)**. Cả hai thuật toán đều được thiết kế để giải quyết các bài toán tối ưu hóa liên tục và thuộc lớp các thuật toán tiến hóa được sử dụng trong nghiên cứu thuật giải di truyền.

Các mục tiêu chính:
- Cài đặt cả hai thuật toán DE và CEM từ đầu
- Thực hiện so sánh hiệu suất toàn diện trên nhiều hàm benchmark khác nhau
- Phân tích hành vi hội tụ và ý nghĩa thống kê của kết quả
- Tạo các biểu diễn trực quan (đường cong hội tụ và animation)

---

## Tổng quan Thuật toán

### 1. Differential Evolution (DE)

**Differential Evolution là gì?**

Differential Evolution là một thuật toán tối ưu hóa ngẫu nhiên dựa trên quần thể (population-based) duy trì một tập hợp các giải pháp ứng viên và phát triển chúng hướng tới tối ưu toàn cục thông qua các phép toán đột biến và tái tổ hợp.

**Các Đặc điểm Chính:**
- **Dựa trên quần thể:** Hoạt động với một quần thể cố định của các cá thể
- **Không cần gradient:** Không yêu cầu thông tin về gradient
- **Hiệu quả:** Hội tụ nhanh trên nhiều hàm benchmark
- **Vừa mạnh:** Hoạt động tốt trên các loại bài toán khác nhau

**Ba Phép toán Chính:**

1. **Đột biến (Mutation):**
   - Chọn ngẫu nhiên ba cá thể khác nhau: a, b, c
   - Tạo vector đột biến: `mutant = a + F * (b - c)`
   - Tham số F (hệ số đột biến) = 0.8 kiểm soát kích thước bước

2. **Lai ghép (Crossover):**
   - Kết hợp vector đột biến với cá thể hiện tại
   - Với mỗi chiều: sử dụng gen từ đột biến nếu giá trị ngẫu nhiên < CR, nếu không dùng cha mẹ
   - Tham số CR (xác suất lai ghép) = 0.9 kiểm soát sự đa dạng
   - Đảm bảo ít nhất một chiều đến từ vector đột biến

3. **Lựa chọn (Selection):**
   - Đánh giá fitness của trial vector
   - Nếu tốt hơn cha mẹ, thay thế cha mẹ bằng trial vector
   - Sử dụng chiến lược lựa chọn tham lam

**Giả mã Thuật toán:**
```
Khởi tạo quần thể ngẫu nhiên
trong khi evaluations < max_evaluations:
    với mỗi cá thể i:
        Chọn ba cá thể ngẫu nhiên a, b, c
        mutant = clip(a + F * (b - c), bounds)
        trial = crossover(mutant, population[i], CR)
        nếu f(trial) < f(population[i]):
            population[i] = trial
```

---

### 2. Cross Entropy Method (CEM)

**Cross Entropy Method là gì?**

Cross Entropy Method là một thuật toán tối ưu hóa xác suất học tập và tinh chỉnh một phân phối xác suất trên không gian tìm kiếm. Nó lặp đi lặp lại lấy mẫu từ phân phối và cập nhật dựa trên fitness của các cá thể được lấy mẫu.

**Các Đặc điểm Chính:**
- **Dựa trên mô hình:** Duy trì một phân phối xác suất (Gaussian) trên không gian tìm kiếm
- **Tinh chỉnh lặp lại:** Hội tụ phân phối hướng tới các vùng tiềm năng
- **Dựa trên tinh hoa:** Sử dụng chỉ những cá thể tốt nhất để cập nhật phân phối
- **Thích ứng:** Phân phối tập trung khi thuật toán tiến hành

**Quá trình Lặp lại:**

1. **Khởi tạo Phân phối:**
   - Khởi tạo trung bình (μ) một cách ngẫu nhiên trên không gian tìm kiếm
   - Khởi tạo độ lệch chuẩn (σ) bằng nửa khoảng không gian tìm kiếm

2. **Lấy mẫu Quần thể:**
   - Tạo quần thể từ phân phối Gaussian: N(μ, σ)
   - Cắt ngắn các cá thể được lấy mẫu để nằm trong bounds

3. **Lựa chọn Tinh hoa:**
   - Đánh giá fitness của tất cả cá thể
   - Chọn 20% hàng đầu làm cá thể tinh hoa (những người thực hiện tốt nhất)

4. **Cập nhật Phân phối:**
   - Tính toán lại trung bình: μ = mean(elites)
   - Tính toán lại độ lệch chuẩn: σ = std(elites) + epsilon nhỏ (1e-5)
   - Phân phối tập trung quanh vùng tiềm năng

5. **Kết thúc:**
   - Tiếp tục cho đến khi ngân sách đánh giá được sử dụng hết

**Giả mã Thuật toán:**
```
Khởi tạo μ ngẫu nhiên, σ bằng half_range
trong khi evaluations < max_evaluations:
    population = lấy mẫu từ N(μ, σ)
    population = clip(population, bounds)
    fitness = evaluate_all(population)
    elites = select_top_20_percent(population, fitness)
    μ = mean(elites)
    σ = std(elites) + epsilon
```

---

## Chi tiết Cài đặt

### Các Hàm Mục tiêu

Năm hàm benchmark được sử dụng để đánh giá hiệu suất thuật toán:

1. **Hàm Sphere**
   - Công thức: f(x) = Σ(x_i²)
   - Không gian tìm kiếm: [-5.12, 5.12]
   - Đặc điểm: Unimodal, dễ, trơn
   - Tối ưu toàn cục: x* = [0, 0, ..., 0], f(x*) = 0

2. **Hàm Rastrigin**
   - Công thức: f(x) = 10n + Σ(x_i² - 10*cos(2π*x_i))
   - Không gian tìm kiếm: [-5.12, 5.12]
   - Đặc điểm: Multimodal cao, khó, nhiều cực tiểu cục bộ
   - Tối ưu toàn cục: x* = [0, 0, ..., 0], f(x*) = 0

3. **Hàm Rosenbrock**
   - Công thức: f(x) = Σ(100*(x_{i+1} - x_i²)² + (1 - x_i)²)
   - Không gian tìm kiếm: [-5.0, 10.0]
   - Đặc điểm: Unimodal, thung lũng kéo dài
   - Tối ưu toàn cục: x* = [1, 1, ..., 1], f(x*) = 0

4. **Hàm Ackley**
   - Công thức: f(x) = -20*exp(-0.2*√(Σ(x_i²)/n)) - exp(Σ(cos(2π*x_i))/n) + 20 + e
   - Không gian tìm kiếm: [-32.0, 32.0]
   - Đặc điểm: Nhiều cực tiểu cục bộ, một tối ưu toàn cục
   - Tối ưu toàn cục: x* = [0, 0, ..., 0], f(x*) = 0

5. **Hàm Griewank**
   - Công thức: f(x) = Σ(x_i²/4000) - Π(cos(x_i/√i)) + 1
   - Không gian tìm kiếm: [-600.0, 600.0]
   - Đặc điểm: Multimodal nhưng có cấu trúc đều đặn
   - Tối ưu toàn cục: x* = [0, 0, ..., 0], f(x*) = 0

---

## Thiết kế Thử nghiệm

### Các Tham số Cấu hình

**Tham số Thuật toán:**
- DE: F (đột biến) = 0.8, CR (lai ghép) = 0.9
- CEM: Tỷ lệ tinh hoa = 20% (20% quần thể hàng đầu)

**Biến Thử nghiệm:**
- **Các Hàm Benchmark:** 5 (Sphere, Rastrigin, Rosenbrock, Ackley, Griewank)
- **Kích thước Vấn đề (d):** 2 và 10
- **Kích thước Quần thể (N):** 8, 16, 32, 64, 128
- **Số Lần Đánh giá Tối đa:** 2000 cho d=2, 10000 cho d=10
- **Chạy Độc lập:** 10 lần chạy mỗi cấu hình

### Tổng số Thử nghiệm

- Số cấu hình: 5 hàm × 2 kích thước × 5 kích thước quần thể = 50
- Chạy mỗi cấu hình: 10
- **Tổng cộng thực thi thuật toán: 500** (250 chạy DE + 250 chạy CEM)

### Phân tích Thống kê

- **Kiểm định Sử dụng:** Independent t-test không cặp
- **Mức Ý nghĩa:** α = 0.05
- **Quy tắc Quyết định:** Nếu p-value < 0.05, các thuật toán khác nhau có ý nghĩa; nếu không là "Hòa"
- **Người Chiến thắng:** Thuật toán với giá trị fitness trung bình thấp hơn

---

## Kết quả và Phân tích

### Tóm tắt Những Phát hiện

Các kết quả thử nghiệm so sánh DE và CEM trên tất cả các cấu hình. Các chỉ số chính:

**Các Chỉ số Báo cáo:**
- **DE_Mean(std):** Fitness trung bình và độ lệch chuẩn từ 10 lần chạy DE
- **CEM_Mean(std):** Fitness trung bình và độ lệch chuẩn từ 10 lần chạy CEM
- **Winner:** Thuật toán nào có hiệu suất tốt hơn về mặt thống kê (p < 0.05)

### Hành vi Hội tụ

#### Biểu đồ Phân tích Hội tụ

Đường cong hội tụ được tạo ra cho thấy:
- **Dòng/vùng xanh:** Fitness trung bình DE với vùng ±1 độ lệch chuẩn
- **Dòng/vùng đỏ:** Fitness trung bình CEM với vùng ±1 độ lệch chuẩn
- **Trục Y:** Thang logarit để trực quan hóa tốt hơn các cải tiến
- **Trục X:** Số lần đánh giá hàm

**Quan sát từ đường cong hội tụ:**
- DE thường hiển thị hội tụ nhanh ban đầu
- CEM hiển thị cải tiến ổn định, nhất quán hơn
- Vùng tô bóng của CEM thường tight hơn (phương sai thấp hơn) ở giai đoạn sau
- Các hàm khác nhau chỉ ra ưu tiên thuật toán khác nhau

### So sánh Thuật toán

#### Điểm Mạnh của Differential Evolution:
✓ Hội tụ nhanh ban đầu trên các hàm đơn giản (Sphere)  
✓ Hiệu suất tốt với quần thể nhỏ  
✓ Tìm kiếm cuc bộ hiệu quả thông qua chu kỳ đột biến-lai ghép  
✓ Độ lệch chuẩn thấp hơn trên một số bài toán  

#### Điểm Yếu của Differential Evolution:
✗ Có thể bị mắc kẹt trong cực tiểu cục bộ trên các hàm multimodal  
✗ Ít ổn định trên các bài toán multimodal cao (Rastrigin, Ackley)  
✗ Hiệu suất thay đổi trên các loại bài toán khác nhau  

#### Điểm Mạnh của Cross Entropy Method:
✓ Vững chắc trên các loại bài toán khác nhau  
✓ Hiệu suất tốt hơn trên các hàm multimodal  
✓ Hội tụ ổn định hơn (phương sai thấp hơn)  
✓ Tinh chỉnh phân phối thích ứng xử lý tốt cảnh quan  
✓ Khám phá tốt theo sau là khai thác  

#### Điểm Yếu của Cross Entropy Method:
✗ Hội tụ ban đầu hơi chậm hơn  
✗ Yêu cầu lựa chọn cẩn thận kích thước quần thể  
✗ Phân phối có thể sụp đổ sớm với các tham số ban đầu xấu  

---

## Trực quan hóa

### 1. Đường cong Hội tụ
Được tạo cho các cấu hình đại diện cho thấy cải tiến fitness trên các đánh giá.
- **Định dạng:** Biểu đồ dòng với tô bóng độ lệch chuẩn
- **Thang đo:** Logarit cho các giá trị fitness
- **Mục đích:** So sánh tốc độ và ổn định hội tụ

### 2. Animation (Tệp GIF)
Đã tạo 10 tệp GIF (5 hàm × 2 thuật toán) cho thấy:
- **Trực quan 2D** của cảnh quan tối ưu hóa (biểu đồ contour)
- **Chuyển động Quần thể** hướng tới tối ưu (các điểm cam)
- **Vị trí Tối ưu Toàn cục** (điểm đỏ)
- **Chỉ số Thời gian Thực** (số lần đánh giá, fitness tốt nhất hiện tại)
- **Trực quan hóa Động** của hành vi khám phá và khai thác thuật toán

Các tệp được tạo:
- `DE_Sphere_d2_N32.gif` / `CEM_Sphere_d2_N32.gif`
- `DE_Rastrigin_d2_N32.gif` / `CEM_Rastrigin_d2_N32.gif`
- `DE_Rosenbrock_d2_N32.gif` / `CEM_Rosenbrock_d2_N32.gif`
- `DE_Ackley_d2_N32.gif` / `CEM_Ackley_d2_N32.gif`
- `DE_Griewank_d2_N32.gif` / `CEM_Griewank_d2_N32.gif`

---

## Kết luận

### Những Phát hiện Chính

1. **Sự Cân bằng Thuật toán:**
   - DE xuất sắc trên các hàm đơn giản, unimodal với hội tụ nhanh
   - CEM cung cấp hiệu suất vững chắc trên các bài toán đa dạng
   - Không có thuật toán nào vượt trội trên tất cả các benchmark

2. **Tác động Độ khó Bài toán:**
   - Cả hai thuật toán xử lý tốt các bài toán 2D
   - Các bài toán 10D hiển thị sự khác biệt hiệu suất đáng kể hơn
   - Các hàm multimodal (Rastrigin, Ackley) ưu tiên CEM

3. **Hiệu ứng Kích thước Quần thể:**
   - Quần thể lớn hơn nói chung cải thiện cả hai thuật toán
   - CEM mở rộng tốt hơn với độ khó bài toán
   - DE hưởi lợi nhiều hơn từ quần thể lớn hơn trên các bài toán đơn giản

4. **Đặc điểm Hội tụ:**
   - DE: Nhanh ban đầu, tiềm ẩn trì trệ
   - CEM: Cải tiến ổn định, nhất quán, đường cong mịn hơn

### Khuyến nghị

**Sử dụng Differential Evolution khi:**
- Hội tụ nhanh là quan trọng
- Bài toán được biết là unimodal hoặc đơn giản
- Tài nguyên tính toán cực kỳ hạn chế

**Sử dụng Cross Entropy Method khi:**
- Độ tin cậy trên các bài toán khác nhau là quan trọng
- Bài toán là multimodal hoặc cảnh quan không xác định
- Bạn cần kết quả ổn định, có thể tái tạo
- Bạn có thể chịu được lặp lại thêm một chút

### Cải tiến Tương lai

Các cải tiến tiềm ẩn cho nghiên cứu này:
1. Cài đặt kiểm soát tham số thích ứng (F và CR tự thích ứng cho DE)
2. Kiểm tra trên các bộ benchmark thử thách hơn (cuộc thi CEC)
3. Cài đặt các cách tiếp cận hybrid (DE + CEM)
4. Thực thi song song cho các hệ thống đa lõi
5. Phân tích độ nhạy trên các tham số thuật toán
6. So sánh thời gian chạy và hiệu quả tính toán

---

## Tham khảo

**Differential Evolution:**
- Storn, R., & Price, K. (1997). "Differential Evolution - A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces." Journal of Global Optimization, 11(4), 341-359.

**Cross Entropy Method:**
- Rubinstein, R. Y., & Kroese, D. P. (2004). "The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization, Monte Carlo Simulation and Machine Learning." Springer-Verlag.

**Các Hàm Benchmark:**
- Molina, D., Lozano, M., & García-Martínez, C. (2020). "Comprehensive taxonomies of nature-and bio-inspired optimization algorithms." Journal of Computational Design and Engineering, 7(4), 541-553.

---

## Cấu trúc Mã

```
BT2_23520108.ipynb
├── Mục 1: Nhập và Tiện ích
├── Mục 2: Định nghĩa Hàm Mục tiêu
├── Mục 3: Cài đặt Differential Evolution
├── Mục 4: Cài đặt Cross Entropy Method
├── Mục 5: Thiết kế Thử nghiệm và Thực thi
├── Mục 6: Tóm tắt Kết quả (Bảng Markdown)
├── Mục 7: Phân tích Hội tụ (Biểu đồ)
├── Mục 8: Tạo Animation (Tạo GIF)
└── Mục 9: Hiển thị Animation
```

---

## Các Tệp Được Tạo

- `BT2_23520108.ipynb` - Notebook chính với các cài đặt
- `GIF_Results/` - Thư mục chứa 10 animation GIF
  - 5 GIF cho Differential Evolution
  - 5 GIF cho Cross Entropy Method
- `README.md` - Tệp tài liệu này

---

## Cách Chạy

1. **Cài đặt Phụ thuộc:**
   ```bash
   pip install numpy matplotlib scipy pandas
   ```

2. **Chạy Notebook:**
   - Mở `BT2_23520108.ipynb` trong Jupyter Notebook
   - Thực thi các ô theo thứ tự từ trên xuống dưới
   - Thời gian thực thi: ~5-10 phút (tùy thuộc vào máy)

3. **Xem Kết quả:**
   - Bảng Markdown xuất hiện ở Mục 6
   - Biểu đồ hội tụ xuất hiện ở Mục 7
   - Các tệp GIF lưu trong thư mục `GIF_Results/`

---

**Lần Cập nhập Cuối cùng:** 11 tháng 4 năm 2026  
**Sinh viên:** 23520108  
**Môn học:** CS410.Q2 - Mạng Neural và Thuật giải Di truyền
