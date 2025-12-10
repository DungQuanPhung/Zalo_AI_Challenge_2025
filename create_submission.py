"""
Tạo file submission theo format yêu cầu
"""

import json
from pathlib import Path

def create_submission(tracking_results, output_path='submission.json'):
    """
    Create submission file theo đúng format
    
    Format yêu cầu:
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
    
    Args:
        tracking_results: {
            "video_id": {
                frame_id: [{"x1": ..., "y1": ..., "x2": ..., "y2": ...}, ...]
            }
        }
    """
    submission = []
    
    for video_id, annotations in tracking_results.items():
        # Collect tất cả bboxes từ tất cả frames
        all_bboxes = []
        
        for frame_id, bboxes in annotations.items():
            # Thêm frame_id vào mỗi bbox
            for bbox in bboxes:
                bbox_with_frame = {
                    'frame': int(frame_id),
                    'x1': int(bbox['x1']),
                    'y1': int(bbox['y1']),
                    'x2': int(bbox['x2']),
                    'y2': int(bbox['y2'])
                }
                all_bboxes.append(bbox_with_frame)
        
        # Tạo entry cho video
        video_data = {
            'video_id': video_id,
            'detections': [
                {
                    'bboxes': all_bboxes
                }
            ]
        }
        
        submission.append(video_data)
    
    # Save file
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)
    
    # Print summary
    total_bboxes = sum(len(v['detections'][0]['bboxes']) for v in submission)
    print(f"\n✓ Submission saved: {output_path}")
    print(f"  Videos: {len(submission)}")
    print(f"  Total bboxes: {total_bboxes}")
    
    # Print per-video stats
    for video_data in submission:
        bbox_count = len(video_data['detections'][0]['bboxes'])
        print(f"  - {video_data['video_id']}: {bbox_count} bboxes")