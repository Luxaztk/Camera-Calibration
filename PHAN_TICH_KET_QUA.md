# Phần 3–4: Camera Calibration, khử méo và phân tích kết quả

## Cách chạy

```bash
pip install -r requirements.txt
python calibrate.py          # Phần 2 + 3  (thêm --show để xem ảnh góc)
python undistort_image.py    # Phần 4      (thêm --show để xem ảnh so sánh)
```

| Phần | Sản phẩm bàn giao | File |
|---|---|---|
| 3 | Dữ liệu hiệu chỉnh | `calibration_data.npz` (mtx, dist, rvecs, tvecs, rms, sai số từng ảnh) |
| 3 | Thông số và kết quả sai số | `calibration_results.txt`, `calibration_results.json` |
| 3 | Biểu đồ sai số theo từng ảnh | `reprojection_errors.png` |
| 4 | Ảnh sau khi khử méo | `output_undistorted/` |
| 4 | Ảnh so sánh trước/sau (có lưới tham chiếu) | `output_comparison/` |
| 4 | Bản đồ méo và vùng phủ của bàn cờ | `distortion_map.png` |
| 4 | Đánh giá định lượng | `undistortion_results.txt` |

---

## Phần 3: Camera Calibration

**Thiết lập:** ảnh 1280×720, bàn cờ 9×6 góc trong, 26/26 ảnh phát hiện góc thành công.

### Quy trình
1. `findChessboardCorners` rồi `cornerSubPix` (tinh chỉnh góc đến mức sub-pixel).
2. `calibrateCamera` **lần 1** với cả 26 ảnh.
3. Tính **reprojection error** của từng ảnh: chiếu lại các điểm 3D bằng `projectPoints` rồi so với góc đã phát hiện.
4. Loại các ảnh có RMS > **1.5 px** và hiệu chỉnh **lần 2**.

### Kết quả

| | Lần 1 (26 ảnh) | Lần 2 (23 ảnh) – **cuối cùng** |
|---|---|---|
| RMS | 1.053 px | **0.780 px** |
| Mean reprojection error | 0.788 px | **0.636 px** |

Ba ảnh bị loại: `15_59_19` (2.70 px), `16_02_36` (2.06 px), `16_02_57` (1.55 px).

**Camera Matrix**

```
fx = 1203.57   fy = 1190.07
cx =  529.46   cy =  365.08
```

FOV ≈ 56.0° (ngang) × 33.7° (dọc), phù hợp với một webcam thông thường.

**Distortion Coefficients** `[k1, k2, p1, p2, k3] = [-0.0212, 0.5327, 0.0024, -0.0363, -0.4151]`

### Nhận xét
- RMS **0.78 px < 1 px**, nên kết quả hiệu chỉnh ở mức **tốt, chấp nhận được**. Mức "rất tốt" là < 0.5 px.
- Loại ảnh ngoại lai giảm RMS khoảng **26%**. Xem ảnh `output_corners/WIN_20261111_15_59_19_Pro_corners.jpg`: tờ giấy bị **cong** (cầm tay, mép trên bị gập), nên bàn cờ không còn phẳng. Điều này vi phạm giả thiết Z = 0 của mô hình.
- `cx` lệch khoảng 110 px so với tâm ảnh (640) và `p2` khá lớn. Một webcam thật hiếm khi lệch nhiều như vậy. Đây là dấu hiệu dữ liệu chưa phủ đều khung hình (xem Phần 4), khiến tâm quang học và méo tiếp tuyến bị "bù trừ" lẫn nhau.

---

## Phần 4: Khử méo ảnh và phân tích kết quả

### Quy trình
- `getOptimalNewCameraMatrix` (alpha = 1) → `initUndistortRectifyMap` → `remap` → cắt theo ROI.
- Ảnh so sánh: ghép ảnh gốc và ảnh đã khử méo cạnh nhau, kèm lưới thẳng để quan sát.
- **Đánh giá định lượng (độ thẳng):** góc của mỗi hàng/cột bàn cờ phải nằm trên một đường thẳng. Với mỗi hàng/cột, khớp đường thẳng (`fitLine`) và đo RMS khoảng cách từ các góc tới đường đó, trước và sau khi khử méo (`undistortPoints`).
- **Bản đồ méo:** độ dịch chuyển của từng điểm ảnh khi khử méo, kèm đường bao vùng mà bàn cờ đã đi qua.

### Kết quả

| Chỉ số | Giá trị |
|---|---|
| Độ dịch chuyển tại tâm ảnh | 1.15 px |
| Độ dịch chuyển lớn nhất **trong** vùng phủ bàn cờ | 19.7 px |
| Độ dịch chuyển lớn nhất **ngoài** vùng phủ (biên trái) | 40.6 px |
| Tỉ lệ khung hình có bàn cờ đi qua | **42%** |
| Độ thẳng trung bình trước → sau | 0.346 → 0.337 px (cải thiện 2.4%) |
| Số ảnh có độ thẳng cải thiện | 16/26 |

### Nhận xét
1. **Webcam có độ méo thấp.** Ngay trên ảnh gốc, các hàng góc đã thẳng với sai lệch khoảng 0.35 px, tương đương mức nhiễu khi phát hiện góc. Vì vậy, nhìn ảnh so sánh bằng mắt gần như không thấy khác biệt, và mức cải thiện độ thẳng chỉ khoảng 2.4%.
2. **Bàn cờ chỉ phủ 42% khung hình**, tập trung ở giữa (x ≈ 245–1095 px) và **không chạm biên trái/phải**. Trong khi đó, méo xuyên tâm tăng mạnh theo khoảng cách tới tâm, nên vùng biên là nơi cần dữ liệu nhất.
3. Độ dịch chuyển 40 px ở biên trái nằm **ngoài vùng có dữ liệu**. Đây là giá trị **ngoại suy** của đa thức `k1 r² + k2 r⁴ + k3 r⁶`. `k2` và `k3` lớn và trái dấu nhau (+0.53 và −0.42), cho thấy mô hình đang "khớp quá" (overfitting), nên kết quả khử méo ở vùng biên **kém tin cậy**.
4. Một số ảnh có độ thẳng tệ hơn sau khi khử méo (ví dụ `16_02_57`, `16_01_43`). Nguyên nhân là bàn cờ bị cong hoặc nghiêng mạnh: sai số do giấy không phẳng lớn hơn phần méo ống kính được bù.

### Nguyên nhân sai số
| Nguyên nhân | Biểu hiện | Ảnh hưởng |
|---|---|---|
| Bàn cờ in giấy, cầm tay, bị cong | `15_59_19` có RMS 2.7 px | Vi phạm giả thiết mặt phẳng, làm tăng RMS |
| Bàn cờ ở xa / nghiêng mạnh | `16_02_36`, `16_02_57` | Góc nhỏ, bị mờ nên phát hiện góc kém chính xác |
| Phân bố ảnh không đều, thiếu vùng biên | Vùng phủ chỉ 42% | k2, k3, cx, p2 ước lượng kém; ngoại suy sai ở biên |
| Ánh sáng yếu, nhiễu webcam, nén JPEG | Ảnh trong phòng tối | Tăng nhiễu vị trí góc (khoảng 0.3–0.5 px) |
| Rung tay / mờ chuyển động | Laplacian variance thấp (ví dụ `15_59_19` ≈ 109) | Góc kém sắc nét |

### Đề xuất cải thiện
- Dán bàn cờ lên **bìa cứng hoặc mặt phẳng cứng**, hoặc dùng màn hình hiển thị bàn cờ.
- Chụp thêm ảnh có bàn cờ ở **các góc và các cạnh** của khung hình, với nhiều độ nghiêng (±30–45°).
- Đặt camera trên giá cố định, đủ sáng, tránh rung.
- Nếu méo nhỏ, có thể dùng mô hình đơn giản hơn (cờ `cv2.CALIB_FIX_K3`) để tránh overfitting.
- Đặt `SQUARE_SIZE` đúng kích thước ô thực tế (mm) để `tvecs` có đơn vị thật.
