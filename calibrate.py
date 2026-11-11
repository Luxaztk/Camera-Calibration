"""
PHẦN 2 + 3: Phát hiện góc bàn cờ & Camera Calibration
- Tìm góc bàn cờ trên tất cả ảnh trong images/, lưu ảnh minh họa vào output_corners/
- Hiệu chỉnh camera: Camera Matrix, Distortion Coefficients
- Đánh giá: RMS (do OpenCV trả về) + reprojection error trung bình và theo từng ảnh
- Loại ảnh có sai số > ERROR_THRESHOLD và hiệu chỉnh lại
- Lưu kết quả: calibration_data.npz, calibration_results.txt, calibration_results.json,
  reprojection_errors.png (biểu đồ sai số theo từng ảnh)

Chạy:  python calibrate.py           (không hiện cửa sổ)
       python calibrate.py --show    (hiện ảnh góc trong lúc chạy)
"""
import cv2
import numpy as np
import os
import glob
import json
import sys

# ĐỊNH NGHĨA KÍCH THƯỚC BÀN CỜ (số góc trong: cột x hàng)
CHECKERBOARD = (9, 6)
# Kích thước 1 ô vuông (mm). Sửa lại theo bàn cờ thực tế để tvecs có đơn vị mm.
SQUARE_SIZE = 25.0

SHOW = '--show' in sys.argv

# Tiêu chí dừng cho thuật toán tinh chỉnh góc
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Tọa độ 3D của các góc trên mặt phẳng bàn cờ (Z = 0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []   # Tọa độ 3D
imgpoints = []   # Tọa độ 2D
valid_names = [] # Tên ảnh tìm thấy bàn cờ
failed_names = []

output_dir_corners = 'output_corners'
os.makedirs(output_dir_corners, exist_ok=True)

images = sorted(glob.glob('./images/*.jpg'))
gray_shape = None

print("Bắt đầu tìm góc bàn cờ...")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Bỏ qua: không đọc được {fname}")
        failed_names.append(os.path.basename(fname))
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        print(f"Đã tìm thấy bàn cờ trong ảnh: {os.path.basename(fname)}")
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        valid_names.append(os.path.basename(fname))

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        base_name, extension = os.path.splitext(os.path.basename(fname))
        cv2.imwrite(os.path.join(output_dir_corners, f"{base_name}_corners{extension}"), img)

        if SHOW:
            cv2.imshow('Cac goc duoc phat hien', img)
            cv2.waitKey(500)
    else:
        print(f"KHÔNG tìm thấy bàn cờ trong ảnh: {os.path.basename(fname)}")
        failed_names.append(os.path.basename(fname))

if SHOW:
    cv2.destroyAllWindows()

print(f"\nThống kê phát hiện góc: {len(valid_names)}/{len(images)} ảnh thành công, "
      f"{len(failed_names)} ảnh thất bại.")

if len(objpoints) == 0:
    print("\nLỖI: Không tìm thấy đủ điểm bàn cờ trong bất kỳ ảnh nào.")
    print("Vui lòng kiểm tra lại kích thước CHECKERBOARD và đường dẫn ảnh.")
    sys.exit(1)

# ===================== PHẦN 3: CAMERA CALIBRATION =====================
# Ngưỡng loại ảnh: ảnh có reprojection RMS > ngưỡng sẽ bị loại và hiệu chỉnh lại
ERROR_THRESHOLD = 1.5  # pixel


def calibrate(objpts, imgpts):
    """Hiệu chỉnh và tính reprojection error theo từng ảnh."""
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpts, imgpts, gray_shape, None, None)
    errs, means = [], []
    for i in range(len(objpts)):
        proj, _ = cv2.projectPoints(objpts[i], rvecs[i], tvecs[i], mtx, dist)
        d = np.linalg.norm(imgpts[i].reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
        errs.append(float(np.sqrt(np.mean(d ** 2))))
        means.append(float(np.mean(d)))
    return rms, mtx, dist, rvecs, tvecs, errs, means


print("\nBắt đầu hiệu chỉnh máy ảnh (lần 1 - tất cả ảnh hợp lệ)...")
rms0, mtx0, dist0, _, _, errs0, means0 = calibrate(objpoints, imgpoints)
names0 = list(valid_names)
print(f"  RMS lần 1 = {rms0:.4f} px với {len(names0)} ảnh")

# Loại ảnh có sai số lớn (ngoại lai) rồi hiệu chỉnh lại
keep = [i for i, e in enumerate(errs0) if e <= ERROR_THRESHOLD]
removed = [(names0[i], errs0[i]) for i, e in enumerate(errs0) if e > ERROR_THRESHOLD]
if removed and len(keep) >= 10:
    print(f"Loại {len(removed)} ảnh có sai số > {ERROR_THRESHOLD} px: "
          + ", ".join(n for n, _ in removed))
    objpoints = [objpoints[i] for i in keep]
    imgpoints = [imgpoints[i] for i in keep]
    valid_names = [valid_names[i] for i in keep]
    print("Hiệu chỉnh lại (lần 2)...")
    rms, mtx, dist, rvecs, tvecs, per_image_err, per_image_mean = calibrate(objpoints, imgpoints)
    print(f"  RMS lần 2 = {rms:.4f} px với {len(valid_names)} ảnh")
else:
    removed = []
    rms, mtx, dist, rvecs, tvecs, per_image_err, per_image_mean = \
        rms0, mtx0, dist0, *calibrate(objpoints, imgpoints)[3:5], errs0, means0

total_rms = float(np.sqrt(np.mean(np.square(per_image_err))))
mean_error = float(np.mean(per_image_mean))

# Thông số vật lý suy ra
w, h = gray_shape
fx, fy, cx, cy = mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]
fov_x = 2 * np.degrees(np.arctan(w / (2 * fx)))
fov_y = 2 * np.degrees(np.arctan(h / (2 * fy)))

# --- Lưu kết quả ra file .npz (giữ khóa mtx, dist cho undistort_image.py) ---
np.savez('calibration_data.npz', mtx=mtx, dist=dist, rms=rms,
         rvecs=np.array(rvecs), tvecs=np.array(tvecs),
         image_size=np.array(gray_shape),
         per_image_error=np.array(per_image_err),
         image_names=np.array(valid_names))
print("Đã lưu dữ liệu hiệu chỉnh vào file: calibration_data.npz")

# --- In & lưu báo cáo ---
k = dist.ravel()
lines = []
lines.append("KẾT QUẢ HIỆU CHỈNH CAMERA")
lines.append("=" * 50)
lines.append(f"Kích thước ảnh       : {w} x {h}")
lines.append(f"Bàn cờ (góc trong)   : {CHECKERBOARD[0]} x {CHECKERBOARD[1]}, ô = {SQUARE_SIZE} mm")
lines.append(f"Số ảnh hợp lệ        : {len(valid_names)}/{len(images)}")
if failed_names:
    lines.append(f"Ảnh thất bại         : {', '.join(failed_names)}")
lines.append("")
lines.append("MA TRẬN MÁY ẢNH (Camera Matrix):")
lines.append(np.array2string(mtx, precision=4, suppress_small=True))
lines.append(f"  fx = {fx:.3f} px, fy = {fy:.3f} px")
lines.append(f"  cx = {cx:.3f} px, cy = {cy:.3f} px  (tâm ảnh lý tưởng: {w/2:.1f}, {h/2:.1f})")
lines.append(f"  Góc nhìn FOV        : ngang {fov_x:.2f}°, dọc {fov_y:.2f}°")
lines.append("")
lines.append("HỆ SỐ MÉO (Distortion Coefficients) [k1, k2, p1, p2, k3]:")
lines.append("  " + ", ".join(f"{v:.6f}" for v in k))
lines.append("  k1, k2, k3: méo xuyên tâm (radial);  p1, p2: méo tiếp tuyến (tangential)")
lines.append("")
lines.append("ĐÁNH GIÁ SAI SỐ")
lines.append(f"  Lần 1 (tất cả {len(names0)} ảnh)          : RMS = {rms0:.4f} px, "
             f"mean = {np.mean(means0):.4f} px")
if removed:
    lines.append(f"  Ảnh bị loại (> {ERROR_THRESHOLD} px)          : "
                 + ", ".join(f"{n} ({e:.3f})" for n, e in removed))
    lines.append(f"  Lần 2 (sau khi loại, {len(valid_names)} ảnh)    : RMS = {rms:.4f} px")
lines.append("  --- Kết quả cuối cùng ---")
lines.append(f"  RMS (cv2.calibrateCamera)          : {rms:.4f} px")
lines.append(f"  Mean reprojection error            : {mean_error:.4f} px")
lines.append(f"  Ảnh sai số nhỏ nhất                : {valid_names[int(np.argmin(per_image_err))]} "
             f"({min(per_image_err):.4f} px)")
lines.append(f"  Ảnh sai số lớn nhất                : {valid_names[int(np.argmax(per_image_err))]} "
             f"({max(per_image_err):.4f} px)")
lines.append("")
lines.append("Sai số RMS theo từng ảnh (px) - lần 1 / cuối cùng:")
final_err = dict(zip(valid_names, per_image_err))
for n, e in zip(names0, errs0):
    fe = f"{final_err[n]:.4f}" if n in final_err else "  (loại)"
    lines.append(f"  {n:35s} {e:.4f} / {fe}")
lines.append("")
if rms < 0.5:
    lines.append("Nhận xét: RMS < 0.5 px -> hiệu chỉnh RẤT TỐT.")
elif rms < 1.0:
    lines.append("Nhận xét: RMS < 1 px -> hiệu chỉnh TỐT, chấp nhận được.")
else:
    lines.append("Nhận xét: RMS >= 1 px -> nên loại bỏ ảnh sai số cao / chụp lại ảnh.")

report = "\n".join(lines)
print("\n" + report)
with open('calibration_results.txt', 'w', encoding='utf-8') as f:
    f.write(report + "\n")

with open('calibration_results.json', 'w', encoding='utf-8') as f:
    json.dump({
        "image_size": [w, h],
        "checkerboard": list(CHECKERBOARD),
        "square_size_mm": SQUARE_SIZE,
        "num_images": len(images),
        "num_valid": len(valid_names),
        "failed_images": failed_names,
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": k.tolist(),
        "fov_deg": [fov_x, fov_y],
        "rms_initial": rms0,
        "removed_images": dict(removed),
        "rms": rms,
        "mean_reprojection_error": mean_error,
        "per_image_rms_initial": dict(zip(names0, errs0)),
        "per_image_rms": dict(zip(valid_names, per_image_err)),
    }, f, indent=2, ensure_ascii=False)
print("\nĐã lưu báo cáo: calibration_results.txt, calibration_results.json")

# --- Biểu đồ sai số theo từng ảnh (vẽ bằng OpenCV, không cần matplotlib) ---
per_image_err_plot = errs0
n = len(per_image_err_plot)
W, H, pad = 1200, 500, 60
chart = np.full((H, W, 3), 255, np.uint8)
ymax = max(per_image_err_plot) * 1.2
bw = (W - 2 * pad) / n
for i, e in enumerate(per_image_err_plot):
    x0 = int(pad + i * bw + 2)
    x1 = int(pad + (i + 1) * bw - 2)
    y0 = int(H - pad - (e / ymax) * (H - 2 * pad))
    color = (60, 60, 220) if e > ERROR_THRESHOLD else (200, 130, 40)
    cv2.rectangle(chart, (x0, y0), (x1, H - pad), color, -1)
    cv2.putText(chart, str(i + 1), (x0, H - pad + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
for val, label, col in [(ERROR_THRESHOLD, f"threshold = {ERROR_THRESHOLD} px", (0, 0, 200)),
                        (rms0, f"RMS before = {rms0:.3f} px", (0, 140, 255)),
                        (rms, f"RMS after = {rms:.3f} px", (0, 150, 0))]:
    ym = int(H - pad - (val / ymax) * (H - 2 * pad))
    cv2.line(chart, (pad, ym), (W - pad, ym), col, 2)
    cv2.putText(chart, label, (W - pad - 260, ym - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
cv2.line(chart, (pad, pad), (pad, H - pad), (0, 0, 0), 1)
cv2.line(chart, (pad, H - pad), (W - pad, H - pad), (0, 0, 0), 1)
cv2.putText(chart, "Reprojection error per image (px) - red: removed", (pad, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
cv2.putText(chart, f"{ymax:.2f}", (5, pad + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
cv2.putText(chart, "image #", (W // 2 - 30, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.imwrite('reprojection_errors.png', chart)
print("Đã lưu biểu đồ: reprojection_errors.png")
