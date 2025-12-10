"""
Main inference script for Zalo AI Challenge 2025 - AeroEyes
Chạy inference trên tất cả video trong /data và lưu kết quả vào /result/submission.json
"""

import os
import cv2
import json
import glob
from pathlib import Path
from ultralytics import YOLO
import numpy as np


class DroneInference:
    """
    Inference engine for drone object detection
    Compatible with Docker container deployment
    """
    
    def __init__(self, model_path: str = 'yolo.engine', conf_threshold: float = 0.5):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to YOLO model (.pt, .engine, .onnx)
            conf_threshold: Confidence threshold for detection
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load YOLO model
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Confidence threshold: {conf_threshold}")
    
    def process_video(self, video_path: str, video_id: str) -> dict:
        """
        Process single video and return detections
        
        Args:
            video_path: Path to video file
            video_id: Video identifier (e.g., "BlackBox_0")
        
        Returns:
            Dictionary mapping frame_idx -> list of bboxes
        """
        print(f"\nProcessing: {video_id}")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        detections = {}
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO inference
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            # Extract bounding boxes
            if len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    frame_bboxes = []
                    
                    for box in result.boxes:
                        bbox = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = bbox.astype(int).tolist()
                        
                        frame_bboxes.append({
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2
                        })
                    
                    if frame_bboxes:
                        detections[frame_idx] = frame_bboxes
            
            # Progress indicator
            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{total_frames} - Detections: {len(detections)}")
            
            frame_idx += 1
        
        cap.release()
        
        print(f"  ✓ Complete: {len(detections)} frames with detections")
        return detections
    
    def run_inference(self, data_dir: str = '/data', output_path: str = '/result/submission.json'):
        """
        Run inference on all videos in data directory
        
        Args:
            data_dir: Directory containing video folders (e.g., /data/BlackBox_0/drone_video.mp4)
            output_path: Path to save submission JSON (e.g., /result/submission.json)
        """
        print("="*70)
        print("Zalo AI Challenge 2025 - AeroEyes Inference")
        print("="*70)
        
        # Find all video files
        video_pattern = os.path.join(data_dir, '**', 'drone_video.mp4')
        video_paths = glob.glob(video_pattern, recursive=True)
        
        if not video_paths:
            print(f"\n❌ No videos found in {data_dir}")
            print(f"   Expected pattern: {video_pattern}")
            return
        
        print(f"\nFound {len(video_paths)} videos:")
        for vp in video_paths:
            print(f"  - {vp}")
        
        # Process each video
        submission = []
        
        for video_path in video_paths:
            # Extract video_id from path (e.g., "BlackBox_0" from "/data/BlackBox_0/drone_video.mp4")
            video_id = Path(video_path).parent.name
            
            # Process video
            frame_detections = self.process_video(video_path, video_id)
            
            # Convert to submission format
            all_bboxes = []
            for frame_idx, bboxes in frame_detections.items():
                for bbox in bboxes:
                    bbox_with_frame = {
                        'frame': int(frame_idx),
                        'x1': int(bbox['x1']),
                        'y1': int(bbox['y1']),
                        'x2': int(bbox['x2']),
                        'y2': int(bbox['y2'])
                    }
                    all_bboxes.append(bbox_with_frame)
            
            # Add video entry to submission
            video_data = {
                'video_id': video_id,
                'detections': [
                    {
                        'bboxes': all_bboxes
                    }
                ]
            }
            
            submission.append(video_data)
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save submission
        with open(output_path, 'w') as f:
            json.dump(submission, f, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print("✅ INFERENCE COMPLETE")
        print("="*70)
        
        total_bboxes = sum(len(v['detections'][0]['bboxes']) for v in submission)
        print(f"Videos processed: {len(submission)}")
        print(f"Total bboxes: {total_bboxes}")
        print(f"Submission saved: {output_path}")
        
        # Per-video stats
        print("\nPer-video statistics:")
        for video_data in submission:
            bbox_count = len(video_data['detections'][0]['bboxes'])
            print(f"  - {video_data['video_id']}: {bbox_count} bboxes")
        
        print("="*70 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Zalo AI Challenge 2025 - AeroEyes Inference')
    parser.add_argument('--model', type=str, default='yolo.engine',
                        help='Path to YOLO model (.pt, .engine, .onnx)')
    parser.add_argument('--data', type=str, default='/data',
                        help='Directory containing video folders')
    parser.add_argument('--output', type=str, default='/result/submission.json',
                        help='Path to save submission JSON')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for detection')
    
    args = parser.parse_args()
    
    # Run inference
    try:
        engine = DroneInference(model_path=args.model, conf_threshold=args.conf)
        engine.run_inference(data_dir=args.data, output_path=args.output)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
