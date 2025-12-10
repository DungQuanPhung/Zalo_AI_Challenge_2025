from ultralytics import YOLO
import cv2
import os
import json
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path

class Predictor:
    """
    Predictor class for drone object detection with streaming support
    """
    
    def __init__(self, model_path: str = 'yolo11s_best.engine', conf_threshold: float = 0.5):
        """
        Initialize predictor with model
        
        Args:
            model_path: Path to YOLO model (.pt, .engine, etc.)
            conf_threshold: Confidence threshold for detection
        """
        if not os.path.exists(model_path):
            print(f"Không tìm thấy mô hình tại: {model_path}")
            print("Vui lòng kiểm tra lại đường dẫn.")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Storage for results
        self.results_storage = {}
        
        print(f"✓ Đã tải mô hình: {model_path}")
        print(f"✓ Confidence threshold: {conf_threshold}")
    
    def predict_streaming(self, frame_rgb_np: np.ndarray, frame_idx: int, video_id: str = None) -> Optional[List[int]]:
        """
        Predict bounding box for streaming mode (drone deployment)
        
        Args:
            frame_rgb_np: RGB frame as numpy array (H, W, 3)
            frame_idx: Frame index (incremental)
            video_id: Optional video ID for storing results
        
        Returns:
            [x1, y1, x2, y2] if object detected, None otherwise
        """
        # Run YOLO inference
        results = self.model(frame_rgb_np, conf=self.conf_threshold, verbose=False)
        
        # Get first result
        if len(results) == 0:
            return None
        
        result = results[0]
        
        # Check if any boxes detected
        if result.boxes is None or len(result.boxes) == 0:
            return None
        
        # Get the box with highest confidence
        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)
        
        # Get bounding box coordinates
        bbox = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = bbox.astype(int).tolist()
        
        # Store result if video_id provided
        if video_id is not None:
            if video_id not in self.results_storage:
                self.results_storage[video_id] = {}
            self.results_storage[video_id][frame_idx] = [{"x1": x1, "y1": y1, "x2": x2, "y2": y2}]
        
        return [x1, y1, x2, y2]
    
    def run_inference(self):
        model_path = 'yolo11s_best.engine'
    
        if not os.path.exists(model_path):
            print(f"Không tìm thấy mô hình tại: {model_path}")
            print("Vui lòng chạy train.py trước hoặc kiểm tra lại đường dẫn.")
            return

        # Tải mô hình đã huấn luyện
        model = YOLO(model_path)
        print("Đã tải mô hình tùy chỉnh.")

        # --- 1. Dự đoán trên các ảnh tĩnh ---
        print("Đang dự đoán trên ảnh tĩnh...")
        image_paths = [
            'dataset/images/val/blackbox_1.jpg',
            'dataset/images/val/blackbox_2.jpg',
            'dataset/images/val/blackbox_3.jpg'
        ]
        results_img = model(image_paths, conf=0.5) # conf: ngưỡng tự tin

        # Lưu và hiển thị kết quả ảnh
        output_dir = 'inference_results/images'
        os.makedirs(output_dir, exist_ok=True)
        
        for i, r in enumerate(results_img):
            output_path = os.path.join(output_dir, f'result_{os.path.basename(image_paths[i])}')
            r.save(filename=output_path) # Lưu ảnh kết quả
            # r.show() # Mở cửa sổ hiển thị (tùy chọn)
        
        print(f"Đã lưu kết quả ảnh vào: {output_dir}")

        # --- 2. Dự đoán trên video ---
        print("Đang xử lý video drone_video.mp4...")
        video_path = 'public_test/samples/BlackBox_0/drone_video.mp4'
        video_out_path = 'inference_results/output_drone_video.mp4'
        os.makedirs('inference_results', exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        
        # Lấy thông tin video để tạo tệp output
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Định nghĩa codec và tạo đối tượng VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec .mp4
        out = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Chạy dự đoán trên từng khung hình
            # stream=True để tối ưu hóa bộ nhớ khi xử lý video
            results_video = model(frame, conf=0.5, stream=True)
            
            for r in results_video:
                # Lấy khung hình đã được vẽ bounding box
                annotated_frame = r.plot()
                out.write(annotated_frame) # Ghi khung hình vào tệp video output
                
                # (Tùy chọn) Hiển thị trực tiếp
                cv2.imshow('Drone Video Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Xử lý video hoàn tất. Đã lưu tại: {video_out_path}")
    
    def save_submission(self, output_path: str = "/result/submission.json"):
        """
        Save results to submission.json (for Docker container)
        
        Format chuẩn theo Zalo AI Challenge:
        [
            {
                "video_id": "BlackBox_0",
                "detections": [
                    {
                        "bboxes": [
                            {"frame": 123, "x1": 100, "y1": 200, "x2": 150, "y2": 250}
                        ]
                    }
                ]
            }
        ]
        
        Args:
            output_path: Path to save submission file (default: /result/submission.json)
        """
        # Create /result folder if not exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results to submission format (theo chuẩn Zalo AI)
        submission = []
        
        for video_id, frames in self.results_storage.items():
            # Collect all bboxes from all frames
            all_bboxes = []
            
            for frame_idx, bboxes in frames.items():
                # Add frame info to each bbox
                for bbox in bboxes:
                    bbox_with_frame = {
                        'frame': int(frame_idx),
                        'x1': int(bbox['x1']),
                        'y1': int(bbox['y1']),
                        'x2': int(bbox['x2']),
                        'y2': int(bbox['y2'])
                    }
                    all_bboxes.append(bbox_with_frame)
            
            # Create video entry
            video_data = {
                'video_id': video_id,
                'detections': [
                    {
                        'bboxes': all_bboxes
                    }
                ]
            }
            
            submission.append(video_data)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(submission, f, indent=2)
        
        # Print summary
        total_bboxes = sum(len(v['detections'][0]['bboxes']) for v in submission)
        print(f"\n✓ Submission saved to: {output_path}")
        print(f"  Videos: {len(submission)}")
        print(f"  Total bboxes: {total_bboxes}")
        
        # Print per-video stats
        for video_data in submission:
            bbox_count = len(video_data['detections'][0]['bboxes'])
            print(f"  - {video_data['video_id']}: {bbox_count} bboxes")
    
    def test_streaming_mode(self, save_submission: bool = True):
        """
        Test streaming mode simulation (giống như trên drone)
        
        Args:
            save_submission: Whether to save results to /result/submission.json
        """
        print("\n" + "="*70)
        print("Testing Streaming Mode (Drone Simulation)")
        print("="*70)
        
        video_path = 'public_test/samples/BlackBox_0/drone_video.mp4'
        video_id = 'BlackBox_0'  # Extract from path
        
        if not os.path.exists(video_path):
            print(f"Video không tồn tại: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        detection_count = 0
        
        print(f"\nProcessing video: {video_path}")
        print(f"Video ID: {video_id}")
        print("Press 'q' to quit\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB (như trên drone)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Call streaming prediction with video_id to store results
            bbox = self.predict_streaming(frame_rgb, frame_idx, video_id=video_id)
            
            # Visualize result
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                detection_count += 1
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Object: [{x1}, {y1}, {x2}, {y2}]", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Frame: {frame_idx} - No detection", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Streaming Mode Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_idx += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Streaming test completed")
        print(f"  Total frames: {frame_idx}")
        print(f"  Detections: {detection_count} ({detection_count/frame_idx*100:.1f}%)")
        
        # Save submission file
        if save_submission:
            self.save_submission()


if __name__ == '__main__':
    # Example 1: Use streaming mode (recommended for drone deployment)
    print("Mode 1: Streaming Mode (Drone Simulation)")
    predictor = Predictor(model_path='.\save_models\yolo.engine', conf_threshold=0.5)
    predictor.test_streaming_mode()
    
    # Example 2: Traditional inference (optional)
    # print("\nMode 2: Traditional Inference")
    # predictor.run_inference()