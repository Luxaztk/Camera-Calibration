import cv2
import numpy as np
import os
import glob

# ĐỊNH NGHĨA KÍCH THƯỚC BÀN CỜ (sửa lại cho đúng với bộ ảnh của bạn)
CHECKERBOARD = (9, 6) # Ví dụ: (6,9) cho OCamCalib, (9,6) cho OpenCV

# Tạo tiêu chí dừng cho thuật toán tìm góc
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Chuẩn bị "object points" (tọa độ 3D của các góc)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Mảng để lưu trữ object points và image points từ tất cả các ảnh
objpoints = [] # Tọa độ 3D
imgpoints = [] # Tọa độ 2D

# --- PHẦN THÊM VÀO: TẠO THƯ MỤC LƯU ẢNH MINH HỌA ---
output_dir_corners = 'output_corners'
os.makedirs(output_dir_corners, exist_ok=True)
# --------------------------------------------------

# Lấy đường dẫn của tất cả các ảnh trong thư mục images
images = glob.glob('./images/*.jpg') # Sửa đuôi file nếu cần

# Biến để lưu kích thước ảnh
gray_shape = None

print("Bắt đầu tìm góc bàn cờ...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1] # Cập nhật kích thước ảnh

    # Tìm các góc của bàn cờ
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret == True:
        print(f"Đã tìm thấy bàn cờ trong ảnh: {os.path.basename(fname)}")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Vẽ các góc lên ảnh
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        # --- PHẦN THÊM VÀO: LƯU ẢNH ĐÃ VẼ GÓC RA FILE ---
        base_name, extension = os.path.splitext(os.path.basename(fname))
        new_filename = os.path.join(output_dir_corners, f"{base_name}_corners{extension}")
        cv2.imwrite(new_filename, img)
        print(f" -> Đã lưu ảnh minh họa vào: {new_filename}")
        # -----------------------------------------------

        # Hiển thị các góc để kiểm tra
        cv2.imshow('Cac goc duoc phat hien', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()
print("\nĐã xử lý xong tất cả các ảnh.")

# Kiểm tra xem có tìm thấy đủ điểm để hiệu chỉnh hay không
if len(objpoints) > 0:
    print("Bắt đầu hiệu chỉnh máy ảnh...")
    
    # Bắt đầu hiệu chỉnh
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    # Lưu kết quả ra file .npz
    output_file = 'calibration_data.npz'
    np.savez(output_file, mtx=mtx, dist=dist)
    print(f"\nĐã lưu dữ liệu hiệu chỉnh vào file: {output_file}")
    
    # In kết quả ra màn hình
    print("------------------------------------")
    print("MA TRẬN MÁY ẢNH (Camera Matrix):\n")
    print(mtx)
    print("\nHỆ SỐ MÉO (Distortion Coefficients):\n")
    print(dist)
    print("------------------------------------")
else:
    print("\nLỖI: Không tìm thấy đủ điểm bàn cờ trong bất kỳ ảnh nào.")
    print("Vui lòng kiểm tra lại kích thước CHECKERBOARD và đường dẫn ảnh.")