"""
ReID + Tracking Pipeline - Clean & Modular Version
- Centralized configuration
- Folder selection support
- Optimized code structure
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from create_submission import create_submission
import onnxruntime as ort


# =====================================================================
# CORE COMPONENTS
# =====================================================================

class TensorRTReIDModel:
    """ReID model using TensorRT Engine"""
    
    def __init__(self, engine_path, device='cuda'):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        from torchvision import transforms
        
        self.device = device
        
        # Load TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def extract(self, crop):
        """Extract L2-normalized embedding from BGR image crop"""
        import pycuda.driver as cuda
        
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(crop_rgb).unsqueeze(0).numpy().astype(np.float32)
        
        # Copy input to device
        np.copyto(self.inputs[0]['host'], img_tensor.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Copy output to host
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        # Get embedding
        emb = self.outputs[0]['host'].flatten().astype(np.float32)
        
        # L2 normalize
        norm = np.linalg.norm(emb) + 1e-8
        emb = emb / norm
        
        return emb


class ONNXReIDModel:
    """ReID model using ONNX Runtime"""
    
    def __init__(self, model_path, device='cuda'):
        from torchvision import transforms
        
        self.device = device
        
        # Setup ONNX Runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def extract(self, crop):
        """Extract L2-normalized embedding from BGR image crop"""
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(crop_rgb).unsqueeze(0).numpy()
        
        # Run inference
        embedding = self.session.run([self.output_name], {self.input_name: img_tensor})[0]
        
        emb = embedding.flatten().astype(np.float32)
        
        # L2 normalize
        norm = np.linalg.norm(emb) + 1e-8
        emb = emb / norm
        
        return emb


class PyTorchReIDModel:
    """ReID model for feature extraction"""
    
    def __init__(self, model_path, device='cuda'):
        from train_reid import LightweightReIDModel
        from torchvision import transforms
        
        self.device = device
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Auto-detect model config
        embedding_dim = checkpoint.get('embedding.0.weight', torch.zeros(512, 1)).shape[0]
        num_classes = checkpoint.get('classifier.weight', torch.zeros(10, 1)).shape[0]
        
        # Initialize model
        self.model = LightweightReIDModel(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            pretrained=False,
            pooling='gem'
        ).to(device)
        
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def extract(self, crop):
        """Extract L2-normalized embedding from BGR image crop"""
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(img_tensor)

        emb = embedding.cpu().numpy().flatten().astype(np.float32)

        # L2 normalize để dùng cho cosine similarity
        norm = np.linalg.norm(emb) + 1e-8
        emb = emb / norm

        return emb



class ReIDTracker:
    """Hybrid ReID + IoU Tracker"""
    
    def __init__(self, yolo, reid, reference_embeddings, config):
        """
        Args:
            yolo: YOLO model
            reid: ReID model
            reference_embeddings: {video_id: embedding}
            config: Config dict with all parameters
        """
        self.yolo = yolo
        self.reid = reid
        self.reference_embeddings = reference_embeddings
        
        # Unpack config
        self.yolo_conf = config['yolo_conf']
        self.reid_threshold = config['reid_threshold']
        self.track_iou = config['track_iou']
        self.max_lost = config['max_lost']
        self.min_track_age = config['min_track_age']
        self.max_objects = config['max_objects_per_frame']
        
        # State
        self.tracks = {}
        self.next_track_id = 1
        self.frame_count = 0
        self.frame_shape = None
    
    def _cosine_sim(self, emb1, emb2):
        """Cosine similarity"""
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return np.dot(emb1, emb2)
    
    def _iou(self, box1, box2):
        """Calculate IoU"""
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])
        
        inter = max(0, x2-x1) * max(0, y2-y1)
        area1 = (box1['x2']-box1['x1']) * (box1['y2']-box1['y1'])
        area2 = (box2['x2']-box2['x1']) * (box2['y2']-box2['y1'])
        
        return inter / (area1 + area2 - inter + 1e-6)
    
    def _match_reference(self, embedding):
        """Match embedding with references"""
        best_id, best_score = None, 0.0
        
        for vid_id, ref_emb in self.reference_embeddings.items():
            score = self._cosine_sim(embedding, ref_emb)
            if score > best_score and score >= self.reid_threshold:
                best_score = score
                best_id = vid_id
        
        return best_id, best_score
    
    def process_frame(self, frame):
        """Process one frame"""
        self.frame_shape = frame.shape
        self.frame_count += 1
        
        # YOLO Detection
        results = self.yolo(frame, conf=self.yolo_conf, verbose=False, imgsz=800)
        
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                
                # Validate bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Clip to frame
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                
                # Extract ReID
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                try:
                    embedding = self.reid.extract(crop)
                    identity, reid_score = self._match_reference(embedding)
                    
                    # Only accept strong matches
                    if identity is None or reid_score < self.reid_threshold:
                        continue
                    
                    detections.append({
                        'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        'embedding': embedding,
                        'identity': identity,
                        'reid_score': reid_score,
                        'confidence': conf
                    })
                except:
                    continue
        
        # Match with existing tracks
        matched_tracks = set()
        matched_dets = set()
        
        for tid, track in list(self.tracks.items()):
            best_iou, best_idx = 0, -1
            
            for i, det in enumerate(detections):
                if i in matched_dets:
                    continue
                
                iou = self._iou(track['bbox'], det['bbox'])
                
                # Prioritize same identity
                if det['identity'] == track['identity'] and iou > 0.01:
                    best_iou, best_idx = iou, i
                    break
                
                if iou > best_iou and iou > self.track_iou:
                    best_iou, best_idx = iou, i
            
            # Update matched track
            if best_idx >= 0:
                det = detections[best_idx]
                self.tracks[tid].update({
                    'bbox': det['bbox'],
                    'embedding': det['embedding'],
                    'identity': det['identity'],
                    'reid_score': det['reid_score'],
                    'lost_count': 0,
                    'age': self.tracks[tid].get('age', 0) + 1
                })
                matched_tracks.add(tid)
                matched_dets.add(best_idx)
            else:
                # Lost track
                self.tracks[tid]['lost_count'] += 1
                self.tracks[tid]['age'] = self.tracks[tid].get('age', 0) + 1
                
                # Delete if lost too long
                if self.tracks[tid]['lost_count'] > self.max_lost:
                    del self.tracks[tid]
        
        # Create new tracks
        for i, det in enumerate(detections):
            if i not in matched_dets:
                self.tracks[self.next_track_id] = {
                    'bbox': det['bbox'],
                    'embedding': det['embedding'],
                    'identity': det['identity'],
                    'reid_score': det['reid_score'],
                    'lost_count': 0,
                    'age': 0
                }
                self.next_track_id += 1
        
        # Return valid tracks
        expected_id = list(self.reference_embeddings.keys())[0] if self.reference_embeddings else None
        
        results = []
        for tid, track in self.tracks.items():
            # Filters
            if track['identity'] != expected_id:
                continue
            if track['age'] < self.min_track_age:
                continue
            if track['lost_count'] != 0:
                continue
            
            results.append({
                'track_id': tid,
                'identity': track['identity'],
                'bbox': track['bbox'],
                'reid_score': track['reid_score'],
                'confidence': track['reid_score']
            })
        
        # Sort by confidence and limit
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:self.max_objects]
    
    def reset(self):
        """Reset tracker for new video"""
        self.tracks = {}
        self.next_track_id = 1
        self.frame_count = 0
        self.frame_shape = None


# =====================================================================
# PROCESSING FUNCTIONS
# =====================================================================

def load_reference_embeddings(video_id, reference_dir, reid_model):
    """Load reference embeddings for one video"""
    ref_path = reference_dir / video_id / 'object_images'
    
    if not ref_path.exists():
        return None
    
    embeddings = []
    for img_path in sorted(ref_path.glob('*.jpg')):
        img = cv2.imread(str(img_path))
        if img is not None:
            try:
                emb = reid_model.extract(img)  # đã L2-normalized
                embeddings.append(emb)
            except:
                continue

    if not embeddings:
        return None

    # Average all embeddings và normalize lại một lần nữa
    avg_emb = np.mean(embeddings, axis=0)
    avg_norm = np.linalg.norm(avg_emb) + 1e-8
    avg_emb = avg_emb / avg_norm

    return {video_id: avg_emb}



def process_video(video_path, video_id, tracker, output_path=None):
    """Process one video"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return {}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\n🎬 Processing: {video_id} ({total_frames} frames, {fps:.1f} FPS)")
    
    tracker.reset()
    annotations = {}
    
    # Optional video writer
    writer = None
    if output_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    frame_id = 0
    with tqdm(total=total_frames, desc=f"  {video_id}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track
            matched = tracker.process_frame(frame)
            
            # Store annotations
            if matched:
                annotations[frame_id] = [obj['bbox'] for obj in matched]
            
            # Visualize
            if writer:
                # Add frame number text
                cv2.putText(frame, f"Frame: {frame_id}", 
                          (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                          (255, 255, 255), 2)
                
                for obj in matched:
                    bbox = obj['bbox']
                    color = (0, 255, 0) if obj['identity'] == video_id else (0, 0, 255)
                    cv2.rectangle(frame, (bbox['x1'], bbox['y1']), 
                                (bbox['x2'], bbox['y2']), color, 2)
                    cv2.putText(frame, f"{obj['identity']} ({obj['confidence']:.2f})",
                              (bbox['x1'], bbox['y1']-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                writer.write(frame)
            
            frame_id += 1
            pbar.update(1)
    
    cap.release()
    if writer:
        writer.release()
    
    coverage = len(annotations) / total_frames * 100 if total_frames > 0 else 0
    print(f"  ✓ Detected in {len(annotations)}/{total_frames} frames ({coverage:.1f}%)")
    
    return annotations


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main(start_idx=0, end_idx=None, config_preset='balanced'):
    """
    Main pipeline with centralized configuration
    
    Args:
        start_idx: Start from folder index (0-based)
        end_idx: End at folder index (None = all)
        config_preset: 'strict', 'balanced', or 'lenient'
    """
    
    # ═══════════════════════════════════════════════════════════════
    # CENTRALIZED CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    
    CONFIG = {
        # Paths
        'test_dir': Path('public_test/samples'),
        'reference_dir': Path('public_test/samples'),
        'output_dir': Path('reid_tracking_output'),
        'submission_path': 'submission.json',
        
        # Models
        'yolo_model_path': 'yolo11s_best.engine',
        'reid_model_path': 'reid_model.onnx',  # ONNX: Best balance of speed & compatibility
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Processing
        'save_videos': True,
        'start_index': start_idx,
        'end_index': end_idx,
    }
    
    # Preset configurations
    PRESETS = {
        'strict': {
            'yolo_conf': 0.6,
            'reid_threshold': 0.6,
            'track_iou': 0.2,
            'max_lost': 50,
            'min_track_age': 5,
            'max_objects_per_frame': 1,
        },
        'balanced': {
            'yolo_conf': 0.4,
            'reid_threshold': 0.4,
            'track_iou': 0.2,
            'max_lost': 100,
            'min_track_age': 3,
            'max_objects_per_frame': 1,
        },
        'lenient': {
            'yolo_conf': 0.2,
            'reid_threshold': 0.2,
            'track_iou': 0.01,
            'max_lost': 150,
            'min_track_age': 1,
            'max_objects_per_frame': 1,
        }
    }
    
    # Merge preset into config
    CONFIG.update(PRESETS[config_preset])
    
    # ═══════════════════════════════════════════════════════════════
    # PRINT CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"ReID Tracking Pipeline - {config_preset.upper()} Config")
    print(f"{'='*70}")
    print(f"Device: {CONFIG['device']}")
    print(f"YOLO Confidence: {CONFIG['yolo_conf']}")
    print(f"ReID Threshold: {CONFIG['reid_threshold']}")
    print(f"Track IoU: {CONFIG['track_iou']}")
    print(f"Max Lost Frames: {CONFIG['max_lost']}")
    print(f"Min Track Age: {CONFIG['min_track_age']}")
    print(f"Process Range: [{CONFIG['start_index']}, {CONFIG['end_index'] or 'end'}]")
    print(f"{'='*70}\n")
    
    # ═══════════════════════════════════════════════════════════════
    # LOAD MODELS
    # ═══════════════════════════════════════════════════════════════
    
    print("Loading models...")
    yolo = YOLO(CONFIG['yolo_model_path'])
    
    # Auto-detect ReID model type
    reid_path = CONFIG['reid_model_path']
    if reid_path.endswith('.engine'):
        print(f"Using TensorRT ReID model: {reid_path}")
        reid = TensorRTReIDModel(reid_path, CONFIG['device'])
    elif reid_path.endswith('.onnx'):
        print(f"Using ONNX ReID model: {reid_path}")
        reid = ONNXReIDModel(reid_path, CONFIG['device'])
    else:
        print(f"Using PyTorch ReID model: {reid_path}")
        reid = PyTorchReIDModel(reid_path, CONFIG['device'])
    
    print("✓ Models loaded\n")
    
    # ═══════════════════════════════════════════════════════════════
    # GET VIDEO LIST
    # ═══════════════════════════════════════════════════════════════
    
    videos = sorted(CONFIG['test_dir'].glob('*/drone_video.mp4'))
    
    if not videos:
        print(f"❌ No videos found in {CONFIG['test_dir']}")
        return
    
    # Apply index filtering
    if CONFIG['end_index'] is None:
        selected = videos[CONFIG['start_index']:]
    else:
        selected = videos[CONFIG['start_index']:CONFIG['end_index']]
    
    print(f"Found {len(videos)} videos total")
    print(f"Processing {len(selected)} videos:\n")
    for i, v in enumerate(selected, start=CONFIG['start_index']):
        print(f"  [{i}] {v.parent.name}")
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # PROCESS VIDEOS
    # ═══════════════════════════════════════════════════════════════
    
    CONFIG['output_dir'].mkdir(exist_ok=True)
    all_results = {}
    
    for idx, video_path in enumerate(selected, start=CONFIG['start_index']):
        video_id = video_path.parent.name
        
        print(f"\n{'='*70}")
        print(f"[{idx}] {video_id}")
        print(f"{'='*70}")
        
        # Load references
        ref_embeddings = load_reference_embeddings(
            video_id, CONFIG['reference_dir'], reid
        )
        
        if ref_embeddings is None:
            print(f"⚠️  No reference embeddings for {video_id}")
            continue
        
        print(f"✓ Reference loaded")
        
        # Initialize tracker
        tracker = ReIDTracker(yolo, reid, ref_embeddings, CONFIG)
        
        # Process
        output_path = None
        if CONFIG['save_videos']:
            output_path = CONFIG['output_dir'] / f"{video_id}_tracked.mp4"
        
        annotations = process_video(video_path, video_id, tracker, output_path)
        all_results[video_id] = annotations
    
    # ═══════════════════════════════════════════════════════════════
    # CREATE SUBMISSION
    # ═══════════════════════════════════════════════════════════════
    
    if all_results:
        print(f"\n{'='*70}")
        print("Creating submission...")
        create_submission(all_results, CONFIG['submission_path'])
        
        total_dets = sum(sum(len(b) for b in a.values()) for a in all_results.values())
        
        print(f"\n{'='*70}")
        print("✅ PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Videos processed: {len(all_results)}")
        print(f"Total detections: {total_dets}")
        print(f"Submission: {CONFIG['submission_path']}")
        print(f"Output videos: {CONFIG['output_dir']}/")
        print(f"{'='*70}\n")
    else:
        print("\n⚠️  No videos processed!")


if __name__ == "__main__":
    import sys
    
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else None
    preset = sys.argv[3] if len(sys.argv) > 3 else 'balanced'
    
    # Validate preset
    if preset not in ['strict', 'balanced', 'lenient']:
        print(f"❌ Invalid preset: {preset}")
        print("   Valid options: strict, balanced, lenient")
        sys.exit(1)
    
    # Handle -1 as "all"
    if end == -1:
        end = None
    
    main(start_idx=1, end_idx=2, config_preset='lenient')