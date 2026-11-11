import cv2
import numpy as np
import os
import glob

# Tự động đọc dữ liệu hiệu chỉnh từ file
calibration_file = 'calibration_data.npz'
if not os.path.exists(calibration_file):
    print(f"LỖI: Không tìm thấy file dữ liệu '{calibration_file}'. Vui lòng chạy 'calibrate.py' trước.")
    exit()

print("Đang đọc dữ liệu hiệu chỉnh...")
data = np.load(calibration_file)
mtx = data['mtx']
dist = data['dist']
print("Đọc dữ liệu thành công.")

# Tạo thư mục để lưu kết quả nếu chưa có
output_dir_undistorted = 'output_undistorted'
os.makedirs(output_dir_undistorted, exist_ok=True)

# Lấy danh sách tất cả các ảnh trong thư mục images
# !!! LƯU Ý: Sửa lại đuôi file '.bmp' nếu bạn dùng .jpg hoặc .png
image_paths = glob.glob('./images/*.jpg')

if not image_paths:
    print("LỖI: Không tìm thấy ảnh nào trong thư mục 'images'.")
    exit()

print(f"\nTìm thấy {len(image_paths)} ảnh. Bắt đầu xử lý...")

# Vòng lặp để xử lý tất cả ảnh
for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Bỏ qua: Không thể đọc file {image_path}")
        continue

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Chỉnh thẳng ảnh
    undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Cắt ảnh theo vùng có ích
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    # Tạo tên file mới và lưu lại
    base_name = os.path.basename(image_path)
    output_filename = os.path.join(output_dir_undistorted, f"{os.path.splitext(base_name)[0]}_undistorted.jpg")
    cv2.imwrite(output_filename, undistorted_img)
    print(f"Đã xử lý và lưu ảnh: {output_filename}")


print("\nHoàn tất! Tất cả các ảnh đã được chỉnh thẳng và lưu vào thư mục 'output_undistorted'.")

# Kiểm tra xem biến 'img' và 'undistorted_img' có tồn tại không (phòng trường hợp không xử lý được ảnh nào)
if 'img' in locals() and 'undistorted_img' in locals():
    print("Đang mở ảnh cuối cùng để so sánh...")
    # Thay đổi kích thước cửa sổ để dễ xem hơn (tùy chọn)
    cv2.namedWindow('Anh Goc', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Anh Da Chinh Thang', cv2.WINDOW_NORMAL)
    
    # Hiển thị ảnh gốc và ảnh sau khi chỉnh của tấm ảnh cuối cùng
    cv2.imshow('Anh Goc', img)
    cv2.imshow('Anh Da Chinh Thang', undistorted_img)

    print("Nhấn phím bất kỳ trên cửa sổ ảnh để đóng.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# --------------------------------------------------