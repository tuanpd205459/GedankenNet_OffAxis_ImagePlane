# GedankenNet-Phase: Direct Off-Axis Holograms (.bmp)

Mã nguồn được đơn giản hóa hoàn toàn cho bài toán **Giao thoa lệch trục (Direct Off-Axis Holography)** tại **Mặt phẳng ảnh**:

- **BỎ HOÀN TOÀN** lan truyền sóng không gian tự do (không dùng Fresnel / Angular Spectrum FFT).
- **BỎ HOÀN TOÀN** bộ lọc thông thấp khẩu độ thấu kính (Pupil NA filter).
- **ĐẦU VÀO:** Đọc trực tiếp các cặp ảnh thực tế định dạng `.bmp` từ thư mục `data_raw/`.

---

## 📁 Cấu trúc thư mục

```
GedankenNet_OffAxis_ImagePlane/
├── data_raw/                    # Thư mục chứa các cặp ảnh thực nghiệm .bmp
├── data/
│   ├── train/                   # Ảnh dùng để huấn luyện giả lập tự giám sát
│   └── valid/                   # Ảnh validation
├── networks/
│   ├── fno.py                   # Mô hình Fourier Neural Operator 2D (FNO2d)
│   └── unet_parts.py            # Cấu trúc khối phụ U-Net
├── my_tools_offaxis.py          # Toán tử Giao thoa Trực tiếp & Dataset đọc file .bmp
├── train_GedankenOffAxis.py     # Script huấn luyện tự giám sát (Physics-Consistency)
├── test_GedankenOffAxis.py      # Script chạy dữ liệu thực nghiệm .bmp trong data_raw/
└── README.md
```

---

## 🔬 Mô hình Giao thoa Trực tiếp (Direct Interference Model)

Tại mặt phẳng ảnh, ảnh Hologram thu được từ giao thoa giữa trường vật thể $U_{obj} = e^{i \pi \phi}$ và chùm tia tham chiếu lệch góc $R_c = e^{i (k_{xc} x + k_{yc} y)}$:

$$I_c(x,y) = | U_{obj}(x,y) + R_c(x,y) |^2$$

Mô hình AI sẽ học cách khôi phục bản đồ pha $\phi(x,y)$ từ 2 kênh ảnh cường độ $I_1, I_2$ thu được ở 2 góc chụp khác nhau.

---

## 🚀 Hướng dẫn chạy Dữ liệu Thực (`.bmp`)

1. **Chuẩn bị dữ liệu thực nghiệm:**
   - Copy các bức ảnh `.bmp` của bạn vào thư mục `data_raw/`.
   - Mỗi mẫu (sample) gồm 2 bức ảnh ứng với 2 góc chụp (ví dụ: `sample01_angle0.bmp` và `sample01_angle1.bmp`).

2. **Chạy suy luận (Inference):**
   ```bash
   python test_GedankenOffAxis.py
   ```
   Kết quả ảnh pha khôi phục sẽ được xuất ra thư mục `outputs/Gedanken_DirectOffAxis_BMP/` dưới dạng cả file `.bmp` và ảnh màu `.jpg`.
