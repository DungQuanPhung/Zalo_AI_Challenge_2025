import cv2
import json
import os
import shutil
from pathlib import Path

DATA_SOURCES = [
    # NGUỒN 1: Video 'Backpack_0' (giống như file cũ)
    {
        'type': 'video',
        'video_file': 'observing/train/samples/Backpack_0/drone_video.mp4',
        'json_file': 'observing/train/annotations/annotations.json',
        'video_id': 'Backpack_0',
        'class_id_to_assign': 0,  # Gán tất cả box trong video này là lớp 0 (backpack)
        'split': 'train'           # Đưa vào tập 'train'
    },
    
    # NGUỒN 2: Ảnh tĩnh 'val' cũ (giống như file cũ)
    {
        'type': 'static',
        'image_folder': 'observing/train/samples/Backpack_0/', # Thư mục chứa 3 ảnh
        'image_files': ['backpack0_1.jpg', 'backpack0_2.jpg', 'backpack0_3.jpg'],
        'split': 'val'             # Đưa vào tập 'val'
    },
    
    # === VÍ DỤ THÊM DỮ LIỆU MỚI ===  
    # Backpack_1 video và ảnh tĩnh
    {
        'type': 'video',
        'video_file': 'observing/train/samples/Backpack_1/drone_video.mp4',
        'json_file': 'observing/train/annotations/annotations.json', # Tệp JSON cho video vali
        'video_id': 'Backpack_1', # ID trong tệp JSON đó
        'class_id_to_assign': 0,
        'split': 'train'
    },
    {
        'type': 'static',
        'image_folder': 'observing/train/samples/Backpack_1/', # Thư mục chứa 3 ảnh
        'image_files': ['backpack1_1.jpg', 'backpack1_2.jpg', 'backpack1_3.jpg'],
        'split': 'val'             # Đưa vào tập 'val'
    },

    # Jacket_0 video và ảnh tĩnh
    {
        'type': 'video',
        'video_file': 'observing/train/samples/Jacket_0/drone_video.mp4',
        'json_file': 'observing/train/annotations/annotations.json', # Tệp JSON cho video vali
        'video_id': 'Jacket_0', # ID trong tệp JSON đó
        'class_id_to_assign': 1,
        'split': 'train'
    },
    {
        'type': 'static',
        'image_folder': 'observing/train/samples/Jacket_0/', # Thư mục chứa 3 ảnh
        'image_files': ['jacket0_1.jpg', 'jacket0_2.jpg', 'jacket0_3.jpg'],
        'split': 'val'             # Đưa vào tập 'val'
    },

    # Jacket_1 video và ảnh tĩnh
    {
        'type': 'video',
        'video_file': 'observing/train/samples/Jacket_1/drone_video.mp4',
        'json_file': 'observing/train/annotations/annotations.json', # Tệp JSON cho video vali
        'video_id': 'Jacket_1', # ID trong tệp JSON đó
        'class_id_to_assign': 1,
        'split': 'train'
    },
    {
        'type': 'static',
        'image_folder': 'observing/train/samples/Jacket_1/', # Thư mục chứa 3 ảnh
        'image_files': ['jacket1_1.jpg', 'jacket1_2.jpg', 'jacket1_3.jpg'],
        'split': 'val'             # Đưa vào tập 'val'
    },

    # Laptop_0 video và ảnh tĩnh
    {
        'type': 'video',
        'video_file': 'observing/train/samples/Laptop_0/drone_video.mp4',
        'json_file': 'observing/train/annotations/annotations.json', # Tệp JSON cho video vali
        'video_id': 'Laptop_0', # ID trong tệp JSON đó
        'class_id_to_assign': 2,
        'split': 'train'
    },
    {
        'type': 'static',
        'image_folder': 'observing/train/samples/Laptop_0/', # Thư mục chứa 3 ảnh
        'image_files': ['laptop0_1.jpg', 'laptop0_2.jpg', 'laptop0_3.jpg'],
        'split': 'val'             # Đưa vào tập 'val'
    },

    # Laptop_1 video và ảnh tĩnh
    {
        'type': 'video',
        'video_file': 'observing/train/samples/Laptop_1/drone_video.mp4',
        'json_file': 'observing/train/annotations/annotations.json', # Tệp JSON cho video vali
        'video_id': 'Laptop_1', # ID trong tệp JSON đó
        'class_id_to_assign': 2,
        'split': 'train'
    },
    {
        'type': 'static',
        'image_folder': 'observing/train/samples/Laptop_1/', # Thư mục chứa 3 ảnh
        'image_files': ['laptop1_1.jpg', 'laptop1_2.jpg', 'laptop1_3.jpg'],
        'split': 'val'             # Đưa vào tập 'val'
    },

    # Lifering_0 video và ảnh tĩnh
    {
        'type': 'video',
        'video_file': 'observing/train/samples/Lifering_0/drone_video.mp4',
        'json_file': 'observing/train/annotations/annotations.json', # Tệp JSON cho video vali
        'video_id': 'Lifering_0', # ID trong tệp JSON đó
        'class_id_to_assign': 3,
        'split': 'train'
    },
    {
        'type': 'static',
        'image_folder': 'observing/train/samples/Lifering_0/', # Thư mục chứa 3 ảnh
        'image_files': ['lifering0_1.jpg', 'lifering0_2.jpg', 'lifering0_3.jpg'],
        'split': 'val'             # Đưa vào tập 'val'
    },
    
    # Lifering_1 video và ảnh tĩnh
    {
        'type': 'video',
        'video_file': 'observing/train/samples/Lifering_1/drone_video.mp4',
        'json_file': 'observing/train/annotations/annotations.json', # Tệp JSON cho video vali
        'video_id': 'Lifering_1', # ID trong tệp JSON đó
        'class_id_to_assign': 3,
        'split': 'train'
    },
    {
        'type': 'static',
        'image_folder': 'observing/train/samples/Lifering_1/', # Thư mục chứa 3 ảnh
        'image_files': ['lifering1_1.jpg', 'lifering1_2.jpg', 'lifering1_3.jpg'],
        'split': 'val'             # Đưa vào tập 'val'
    },

    # MobilePhone_0 video và ảnh tĩnh
    {
        'type': 'video',
        'video_file': 'observing/train/samples/MobilePhone_0/drone_video.mp4',
        'json_file': 'observing/train/annotations/annotations.json', # Tệp JSON cho video vali
        'video_id': 'MobilePhone_0', # ID trong tệp JSON đó
        'class_id_to_assign': 4,
        'split': 'train'
    },
    {
        'type': 'static',
        'image_folder': 'observing/train/samples/MobilePhone_0/', # Thư mục chứa 3 ảnh
        'image_files': ['mobilephone0_1.jpg', 'mobilephone0_2.jpg', 'mobilephone0_3.jpg'],
        'split': 'val'             # Đưa vào tập 'val'
    },

    # MobilePhone_1 video và ảnh tĩnh
    {
        'type': 'video',
        'video_file': 'observing/train/samples/MobilePhone_1/drone_video.mp4',
        'json_file': 'observing/train/annotations/annotations.json', # Tệp JSON cho video vali
        'video_id': 'MobilePhone_1', # ID trong tệp JSON đó
        'class_id_to_assign': 4,
        'split': 'train'
    },
    {
        'type': 'static',
        'image_folder': 'observing/train/samples/MobilePhone_1/', # Thư mục chứa 3 ảnh
        'image_files': ['mobilephone1_1.jpg', 'mobilephone1_2.jpg', 'mobilephone1_3.jpg'],
        'split': 'val'             # Đưa vào tập 'val'
    },

    # Person1_0 video và ảnh tĩnh
    {
        'type': 'video',
        'video_file': 'observing/train/samples/Person1_0/drone_video.mp4',
        'json_file': 'observing/train/annotations/annotations.json', # Tệp JSON cho video vali
        'video_id': 'Person1_0', # ID trong tệp JSON đó
        'class_id_to_assign': 5,
        'split': 'train'
    },
    {
        'type': 'static',
        'image_folder': 'observing/train/samples/Person1_0/', # Thư mục chứa 3 ảnh
        'image_files': ['person10_1.jpg', 'person10_2.jpg', 'person10_3.jpg'],
        'split': 'val'             # Đưa vào tập 'val'
    },

    # Person1_1 video và ảnh tĩnh
    {
        'type': 'video',
        'video_file': 'observing/train/samples/Person1_1/drone_video.mp4',
        'json_file': 'observing/train/annotations/annotations.json', # Tệp JSON cho video vali
        'video_id': 'Person1_1', # ID trong tệp JSON đó
        'class_id_to_assign': 5,
        'split': 'train'
    },
    {
        'type': 'static',
        'image_folder': 'observing/train/samples/Person1_1/', # Thư mục chứa 3 ảnh
        'image_files': ['person11_1.jpg', 'person11_2.jpg', 'person11_3.jpg'],
        'split': 'val'             # Đưa vào tập 'val'
    },

    # WaterBottle_0 video và ảnh tĩnh
    {
        'type': 'video',
        'video_file': 'observing/train/samples/WaterBottle_0/drone_video.mp4',
        'json_file': 'observing/train/annotations/annotations.json', # Tệp JSON cho video vali
        'video_id': 'WaterBottle_0', # ID trong tệp JSON đó
        'class_id_to_assign': 6,
        'split': 'train'
    },
    {
        'type': 'static',
        'image_folder': 'observing/train/samples/WaterBottle_0/', # Thư mục chứa 3 ảnh
        'image_files': ['waterbottle0_1.jpg', 'waterbottle0_2.jpg', 'waterbottle0_3.jpg'],
        'split': 'val'             # Đưa vào tập 'val'
    },

    # WaterBottle_1 video và ảnh tĩnh
    {
        'type': 'video',
        'video_file': 'observing/train/samples/WaterBottle_1/drone_video.mp4',
        'json_file': 'observing/train/annotations/annotations.json', # Tệp JSON cho video vali
        'video_id': 'WaterBottle_1', # ID trong tệp JSON đó
        'class_id_to_assign': 6,
        'split': 'train'
    },
    {
        'type': 'static',
        'image_folder': 'observing/train/samples/WaterBottle_1/', # Thư mục chứa 3 ảnh
        'image_files': ['waterbottle1_1.jpg', 'waterbottle1_2.jpg', 'waterbottle1_3.jpg'],
        'split': 'val'             # Đưa vào tập 'val'
    }
]

def convert_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Chuyển đổi tọa độ [x1, y1, x2, y2] sang định dạng YOLO."""
    dw = 1. / img_w
    dh = 1. / img_h
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    x_center_norm = x_center * dw
    y_center_norm = y_center * dh
    w_norm = w * dw
    h_norm = h * dh
    return (x_center_norm, y_center_norm, w_norm, h_norm)

def setup_directories(base_dir='dataset'):
    """Xóa và tạo lại cấu trúc thư mục dataset."""
    output_base = Path(base_dir)
    if output_base.exists():
        print(f"Cảnh báo: Đang xóa thư mục {output_base} cũ...")
        shutil.rmtree(output_base)

    global paths
    paths = {
        'img_train': output_base / 'images' / 'train',
        'lbl_train': output_base / 'labels' / 'train',
        'img_val': output_base / 'images' / 'val',
        'lbl_val': output_base / 'labels' / 'val',
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
        
    print(f"Đã tạo cấu trúc thư mục tại: {output_base}")

def process_video_source(source):
    """Xử lý nguồn dữ liệu 'video'."""
    print(f"\n--- Đang xử lý [VIDEO]: {source['video_file']} ---")
    
    video_file = Path(source['video_file'])
    json_file = Path(source['json_file'])
    video_id = source['video_id']
    class_id = source['class_id_to_assign']
    split = source['split'] # 'train' or 'val'
    
    img_dir = paths[f'img_{split}']
    lbl_dir = paths[f'lbl_{split}']

    if not video_file.exists():
        print(f"Lỗi: Không tìm thấy tệp video {video_file}")
        return
    if not json_file.exists():
        print(f"Lỗi: Không tìm thấy tệp JSON {json_file}")
        return

    # Mở video
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_file}")
        return
    IMG_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    IMG_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Kích thước video: {IMG_WIDTH}x{IMG_HEIGHT}")

    # Mở JSON
    with open(json_file, 'r') as f:
        all_data = json.load(f)
    video_data = next((item for item in all_data if item['video_id'] == video_id), None)

    if not video_data:
        print(f"Lỗi: Không tìm thấy video_id '{video_id}' trong {json_file}")
        cap.release()
        return

    # Nhóm các box theo khung hình (logic từ prepare_data_FIXED.py)
    frame_to_bboxes = {}
    total_bboxes_found = 0
    for annotation_set in video_data['annotations']:
        for bbox in annotation_set.get('bboxes', []):
            frame_num = bbox['frame']
            if frame_num not in frame_to_bboxes:
                frame_to_bboxes[frame_num] = []
            frame_to_bboxes[frame_num].append(bbox)
            total_bboxes_found += 1
    print(f"Đã tìm thấy {total_bboxes_found} box, nhóm vào {len(frame_to_bboxes)} khung hình duy nhất.")

    # Xử lý và trích xuất
    frames_processed = 0
    for frame_num, bboxes_in_frame in frame_to_bboxes.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Cảnh báo: Không thể đọc khung hình số {frame_num}")
            continue

        # Tạo tên tệp duy nhất bằng cách thêm video_id (ví dụ: Backpack_0_frame_3483.jpg)
        base_filename = f"{video_id}_frame_{frame_num}"
        img_path = img_dir / f"{base_filename}.jpg"
        lbl_path = lbl_dir / f"{base_filename}.txt"
        
        cv2.imwrite(str(img_path), frame)
        
        yolo_labels = []
        for bbox in bboxes_in_frame:
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            x_c_norm, y_c_norm, w_norm, h_norm = convert_to_yolo(
                x1, y1, x2, y2, IMG_WIDTH, IMG_HEIGHT
            )
            # SỬ DỤNG CLASS ID ĐÃ ĐỊNH NGHĨA
            yolo_labels.append(f"{class_id} {x_c_norm} {y_c_norm} {w_norm} {h_norm}")
        
        with open(lbl_path, 'w') as f:
            f.write("\n".join(yolo_labels))
            
        frames_processed += 1

    cap.release()
    print(f"Đã trích xuất {frames_processed} khung hình vào 'dataset/{split}'.")

def process_static_source(source):
    """Xử lý nguồn dữ liệu 'static'."""
    print(f"\n--- Đang xử lý [STATIC]: {source['image_folder']} ---")
    
    img_folder = Path(source['image_folder'])
    split = source['split']
    img_dir = paths[f'img_{split}']
    
    if not img_folder.exists():
        print(f"Lỗi: Không tìm thấy thư mục ảnh {img_folder}")
        return

    # Xác định danh sách tệp ảnh cần sao chép
    if source['image_files'] is not None:
        # Nếu danh sách được chỉ định
        image_files_to_copy = [img_folder / f for f in source['image_files']]
    else:
        # Nếu không, lấy tất cả các tệp ảnh phổ biến
        image_files_to_copy = list(img_folder.glob('*.jpg')) + \
                              list(img_folder.glob('*.jpeg')) + \
                              list(img_folder.glob('*.png'))
                              
    if not image_files_to_copy:
        print("Không tìm thấy ảnh nào để sao chép.")
        return

    print(f"Đang sao chép {len(image_files_to_copy)} ảnh tĩnh vào 'dataset/{split}'...")
    for img_path in image_files_to_copy:
        if img_path.exists():
            shutil.copy(str(img_path), img_dir)
        else:
            print(f"Cảnh báo: Không tìm thấy tệp ảnh {img_path}")

    print("QUAN TRỌNG: Đã sao chép ảnh tĩnh.")
    print(f"BẠN PHẢI TỰ GÁN NHÃN cho các ảnh này bằng cách tạo tệp .txt trong:")
    print(f"{paths[f'lbl_{split}']}\n")

# --- HÀM CHÍNH ---
def main():
    setup_directories()
    
    for source in DATA_SOURCES:
        if source['type'] == 'video':
            process_video_source(source)
        elif source['type'] == 'static':
            process_static_source(source)
        else:
            print(f"Cảnh báo: Bỏ qua nguồn không xác định loại '{source['type']}'")
            
    print("\n--- HOÀN TẤT CHUẨN BỊ DỮ LIỆU ---")
    print("Kiểm tra thư mục 'dataset/' và đảm bảo bạn đã TỰ GÁN NHÃN cho tất cả ảnh tĩnh.")

if __name__ == "__main__":
    main()