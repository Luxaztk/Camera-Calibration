"""
PHẦN 4: Khử méo ảnh & phân tích kết quả
- Đọc calibration_data.npz (chạy calibrate.py trước)
- Khử méo tất cả ảnh trong images/ -> output_undistorted/
- Tạo ảnh so sánh trước/sau (ghép cạnh nhau, có lưới tham chiếu) -> output_comparison/
- Tạo bản đồ độ dịch chuyển do méo + vùng phủ bàn cờ -> distortion_map.png
- Đánh giá định lượng: độ thẳng của các hàng/cột góc bàn cờ trước và sau khi khử méo
  -> undistortion_results.txt

Chạy:  python undistort_image.py          (không hiện cửa sổ)
       python undistort_image.py --show   (hiện ảnh so sánh cuối cùng)
"""
import cv2
import numpy as np
import os
import glob
import sys

CHECKERBOARD = (9, 6)
SHOW = '--show' in sys.argv
# alpha = 0: cắt bỏ toàn bộ vùng đen (mất ít điểm ảnh ở biên)
# alpha = 1: giữ toàn bộ điểm ảnh gốc (có vùng đen ở biên)
ALPHA = 1

calibration_file = 'calibration_data.npz'
if not os.path.exists(calibration_file):
    print(f"LỖI: Không tìm thấy file dữ liệu '{calibration_file}'. Vui lòng chạy 'calibrate.py' trước.")
    sys.exit(1)

print("Đang đọc dữ liệu hiệu chỉnh...")
data = np.load(calibration_file)
mtx = data['mtx']
dist = data['dist']
print("Đọc dữ liệu thành công.")
print("Camera Matrix:\n", mtx)
print("Distortion:", dist.ravel())

output_dir_undistorted = 'output_undistorted'
output_dir_compare = 'output_comparison'
os.makedirs(output_dir_undistorted, exist_ok=True)
os.makedirs(output_dir_compare, exist_ok=True)

image_paths = sorted(glob.glob('./images/*.jpg'))
if not image_paths:
    print("LỖI: Không tìm thấy ảnh nào trong thư mục 'images'.")
    sys.exit(1)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def straightness_error(pts):
    """RMS khoảng cách (px) từ các góc tới đường thẳng khớp nhất,
    tính trên mọi hàng và mọi cột của bàn cờ. Càng nhỏ -> đường càng thẳng."""
    pts = pts.reshape(CHECKERBOARD[1], CHECKERBOARD[0], 2)
    lines = [pts[r, :, :] for r in range(pts.shape[0])] + \
            [pts[:, c, :] for c in range(pts.shape[1])]
    d_all = []
    for ln in lines:
        vx, vy, x0, y0 = cv2.fitLine(ln.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        d = np.abs((ln[:, 0] - x0) * vy - (ln[:, 1] - y0) * vx)
        d_all.extend(d.tolist())
    d_all = np.array(d_all)
    return float(np.sqrt(np.mean(d_all ** 2))), float(np.max(d_all))


def draw_grid(img, step=80, color=(0, 255, 0)):
    """Vẽ lưới thẳng tham chiếu để dễ quan sát đường cong/thẳng."""
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), color, 1)
    return out


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


print(f"\nTìm thấy {len(image_paths)} ảnh. Bắt đầu xử lý...")

results = []  # (tên, before_rms, before_max, after_rms, after_max)
img = undistorted_img = compare = None
coverage = None  # mặt nạ vùng ảnh có bàn cờ đi qua (dữ liệu hiệu chỉnh)

for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Bỏ qua: Không thể đọc file {image_path}")
        continue

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), ALPHA, (w, h))

    # Khử méo bằng initUndistortRectifyMap + remap (tương đương cv2.undistort,
    # nhưng có thể tái sử dụng map cho nhiều ảnh/video)
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1)
    undistorted_full = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # Cắt ảnh theo vùng có ích (ROI)
    x, y, rw, rh = roi
    undistorted_img = undistorted_full[y:y + rh, x:x + rw] if rw > 0 and rh > 0 else undistorted_full

    base = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_dir_undistorted, f"{base}_undistorted.jpg"), undistorted_img)

    # --- Đánh giá định lượng: độ thẳng của các hàng/cột góc bàn cờ ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if coverage is None:
        coverage = np.zeros((h, w), np.uint8)
    if ok:
        hull = cv2.convexHull(corners.reshape(-1, 2).astype(np.int32))
        cv2.fillConvexPoly(coverage, hull, 255)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        und_corners = cv2.undistortPoints(corners, mtx, dist, P=newcameramtx)
        b_rms, b_max = straightness_error(corners.reshape(-1, 2))
        a_rms, a_max = straightness_error(und_corners.reshape(-1, 2))
        results.append((base, b_rms, b_max, a_rms, a_max))
        txt = f" | thang hang: {b_rms:.3f} -> {a_rms:.3f} px"
    else:
        txt = ""

    # --- Ảnh so sánh trước/sau (cùng kích thước, có lưới tham chiếu) ---
    left = label(draw_grid(img), "Truoc hieu chinh (Original)")
    right = label(draw_grid(cv2.resize(undistorted_img, (w, h))), "Sau hieu chinh (Undistorted)")
    compare = np.hstack([left, np.full((h, 6, 3), 255, np.uint8), right])
    cv2.imwrite(os.path.join(output_dir_compare, f"{base}_compare.jpg"), compare)

    print(f"Đã xử lý: {base}{txt}")

# ===================== BẢN ĐỒ MÉO =====================
# Độ dịch chuyển (pixel) của mỗi điểm ảnh khi khử méo, so với mô hình pinhole lý tưởng
h, w = img.shape[:2]
xs, ys = np.meshgrid(np.arange(0, w, 4), np.arange(0, h, 4))
pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
und = cv2.undistortPoints(pts, mtx, dist, P=mtx).reshape(-1, 2)
disp = np.linalg.norm(und - pts.reshape(-1, 2), axis=1).reshape(xs.shape)
max_disp = float(disp.max())
corner_disp = float(max(disp[0, 0], disp[0, -1], disp[-1, 0], disp[-1, -1]))
center_disp = float(disp[disp.shape[0] // 2, disp.shape[1] // 2])
norm = np.clip(disp / max(max_disp, 1e-6) * 255, 0, 255).astype(np.uint8)
heat = cv2.resize(cv2.applyColorMap(norm, cv2.COLORMAP_JET), (w, h), interpolation=cv2.INTER_NEAREST)
# Vẽ đường bao vùng ảnh mà bàn cờ đã phủ tới (vùng có dữ liệu hiệu chỉnh)
coverage_ratio = float((coverage > 0).mean())
cnts, _ = cv2.findContours(coverage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(heat, cnts, -1, (255, 255, 255), 2)
# Độ dịch chuyển lớn nhất bên trong / bên ngoài vùng phủ
cov_small = cv2.resize(coverage, (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
disp_in = float(disp[cov_small].max()) if cov_small.any() else 0.0
disp_out = float(disp[~cov_small].max()) if (~cov_small).any() else 0.0
heat = label(heat, f"Distortion map: max {max_disp:.1f} px (blue=0, red=max) | white = chessboard coverage")
cv2.imwrite('distortion_map.png', heat)

# ===================== BÁO CÁO =====================
lines = ["KẾT QUẢ KHỬ MÉO ẢNH", "=" * 60,
         f"Số ảnh xử lý          : {len(image_paths)}",
         f"alpha (getOptimalNewCameraMatrix): {ALPHA}",
         "",
         "BẢN ĐỒ MÉO (độ dịch chuyển điểm ảnh khi khử méo):",
         f"  Tại tâm ảnh         : {center_disp:.2f} px",
         f"  Tại 4 góc ảnh (max) : {corner_disp:.2f} px",
         f"  Lớn nhất toàn ảnh   : {max_disp:.2f} px",
         "",
         "VÙNG PHỦ CỦA BÀN CỜ (dữ liệu dùng để hiệu chỉnh):",
         f"  Tỉ lệ diện tích ảnh có bàn cờ đi qua: {coverage_ratio * 100:.1f}%",
         f"  Độ dịch chuyển max TRONG vùng phủ   : {disp_in:.2f} px",
         f"  Độ dịch chuyển max NGOÀI vùng phủ   : {disp_out:.2f} px (ngoại suy, kém tin cậy)",
         ""]
if results:
    arr = np.array([r[1:] for r in results])
    improved = int(np.sum(arr[:, 2] < arr[:, 0]))
    lines += ["ĐỘ THẲNG CỦA HÀNG/CỘT GÓC BÀN CỜ (RMS khoảng cách tới đường thẳng, px)",
              f"  {'Ảnh':32s} {'Trước':>8s} {'Sau':>8s} {'Giảm':>8s}"]
    for name, b, bm, a, am in results:
        red = (b - a) / b * 100 if b > 0 else 0
        lines.append(f"  {name:32s} {b:8.3f} {a:8.3f} {red:7.1f}%")
    mb, ma = arr[:, 0].mean(), arr[:, 2].mean()
    lines += ["",
              f"  Trung bình trước    : {mb:.3f} px (max {arr[:, 1].max():.3f} px)",
              f"  Trung bình sau      : {ma:.3f} px (max {arr[:, 3].max():.3f} px)",
              f"  Mức cải thiện       : {(mb - ma) / mb * 100:.1f}%",
              f"  Số ảnh cải thiện    : {improved}/{len(results)}"]
report = "\n".join(lines)
print("\n" + report)
with open('undistortion_results.txt', 'w', encoding='utf-8') as f:
    f.write(report + "\n")

print(f"\nHoàn tất! Ảnh đã khử méo: '{output_dir_undistorted}/', ảnh so sánh: '{output_dir_compare}/',")
print("bản đồ méo: 'distortion_map.png', báo cáo: 'undistortion_results.txt'.")

if SHOW and compare is not None:
    cv2.namedWindow('So sanh Truoc - Sau', cv2.WINDOW_NORMAL)
    cv2.imshow('So sanh Truoc - Sau', compare)
    print("Nhấn phím bất kỳ trên cửa sổ ảnh để đóng.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
