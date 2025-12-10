"""
Script tổng hợp để train YOLO và ReID models
- YOLO: Train từ dataset trong folder dataset/
- ReID: Train từ data trong observing/train/samples/

Usage:
    python train_models.py --train-yolo --train-reid
    python train_models.py --train-yolo  # Chỉ train YOLO
    python train_models.py --train-reid  # Chỉ train ReID
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def train_yolo(
    data_yaml='data.yaml',
    model_size='s',  # n, s, m, l, x
    epochs=20,
    img_size=640,
    batch_size=64,
    device='cuda',
    project='runs/detect',
    name='yolo_drone_model',
    resume=False,  # Thêm tham số resume
    resume_model=None  # Đường dẫn model để resume
):
    """
    Train YOLO model từ dataset trong folder dataset/
    
    Args:
        data_yaml: Đường dẫn đến file data.yaml
        model_size: Kích thước model (n, s, m, l, x)
        epochs: Số epochs
        img_size: Kích thước ảnh input
        batch_size: Batch size
        device: Device ('cuda' hoặc 'cpu')
        project: Thư mục project
        name: Tên model
        resume: Resume training từ checkpoint
        resume_model: Đường dẫn checkpoint để resume (ví dụ: 'yolo11s_best.pt')
    """
    print("\n" + "="*70)
    print("TRAINING YOLO MODEL")
    print("="*70)
    
    # Kiểm tra data.yaml
    if not Path(data_yaml).exists():
        print(f"❌ Không tìm thấy file {data_yaml}")
        print("   Vui lòng tạo file data.yaml với cấu trúc:")
        print("   train: ./dataset/images/train")
        print("   val: ./dataset/images/val")
        print("   nc: <số lớp>")
        print("   names: [<danh sách tên lớp>]")
        return False
    
    # Kiểm tra dataset
    dataset_dir = Path('dataset')
    if not dataset_dir.exists():
        print(f"❌ Không tìm thấy folder dataset/")
        print("   Vui lòng chạy prepare_multi_data.py để tạo dataset trước")
        return False
    
    # Kiểm tra images và labels
    train_images = dataset_dir / 'images' / 'train'
    train_labels = dataset_dir / 'labels' / 'train'
    val_images = dataset_dir / 'images' / 'val'
    val_labels = dataset_dir / 'labels' / 'val'
    
    if not train_images.exists() or not train_labels.exists():
        print(f"❌ Không tìm thấy train data trong dataset/")
        return False
    
    if not val_images.exists() or not val_labels.exists():
        print(f"⚠️  Không tìm thấy validation data, sẽ dùng train data")
    
    # Đếm số file
    num_train = len(list(train_images.glob('*.jpg'))) if train_images.exists() else 0
    num_val = len(list(val_images.glob('*.jpg'))) if val_images.exists() else 0
    
    print(f"\n📊 Dataset Information:")
    print(f"   Train images: {num_train}")
    print(f"   Val images: {num_val}")
    print(f"   Data config: {data_yaml}")
    
    # Chọn model
    if resume and resume_model:
        # Sử dụng pretrained model để train tiếp
        model_name = resume_model
        if not Path(model_name).exists():
            print(f"❌ Không tìm thấy model {model_name}")
            return False
        print(f"\n🔄 Continue training with pretrained model: {model_name}")
    else:
        # Train mới
        model_name = f'yolo11{model_size}.pt'
        if not Path(model_name).exists():
            # Fallback to yolov8
            model_name = f'yolov8{model_size}.pt'
            if not Path(model_name).exists():
                print(f"❌ Không tìm thấy model {model_name}")
                print("   Vui lòng download model trước")
                return False
    
    print(f"\n🤖 Model: {model_name}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {img_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    
    # Load model
    print(f"\n📦 Loading model...")
    model = YOLO(model_name)
    
    # Train
    print(f"\n🚀 Starting training...")
    try:
        # Nếu resume model, không dùng resume=True mà dùng model như pretrained
        if resume and resume_model:
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                project=project,
                name=name,
                workers=4,
                patience=10,  # Early stopping
                save=True,
                save_period=10,  # Save checkpoint every 10 epochs
                val=True,  # Validate during training
                plots=True,  # Generate plots
                verbose=True
                # Không dùng resume=True để tránh lỗi
            )
        else:
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                project=project,
                name=name,
                workers=4,
                patience=10,  # Early stopping
                save=True,
                save_period=10,  # Save checkpoint every 10 epochs
                val=True,  # Validate during training
                plots=True,  # Generate plots
                verbose=True
            )
        
        print(f"\n✅ Training completed!")
        print(f"   Best model: {results.save_dir}/weights/best.pt")
        print(f"   Last model: {results.save_dir}/weights/last.pt")
        
        # Copy best model to root
        import shutil
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        if best_model_path.exists():
            output_path = Path(f'yolo11{model_size}_best.pt')
            shutil.copy(best_model_path, output_path)
            print(f"   Copied to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_reid(
    train_dir='observing/train',
    classified_dir='classified_objects',
    use_classified=True,
    model_path='reid_model_best.pth',
    onnx_path='reid_model.onnx',
    embedding_dim=256,
    batch_size=64,
    epochs=20,
    lr=0.001,
    loss_type='crossentropy',
    input_height=256,
    input_width=128,
    pooling='gem',
    device='cuda',
    skip_dataset=False
):
    """
    Train ReID model từ data trong observing/train/samples/
    
    Args:
        train_dir: Thư mục chứa training data
        classified_dir: Thư mục chứa classified objects (nếu dùng)
        use_classified: Sử dụng classified data thay vì extract từ video
        model_path: Đường dẫn lưu model
        onnx_path: Đường dẫn lưu ONNX model
        embedding_dim: Kích thước embedding
        batch_size: Batch size
        epochs: Số epochs
        lr: Learning rate
        loss_type: Loại loss ('triplet' hoặc 'crossentropy')
        input_height: Chiều cao input
        input_width: Chiều rộng input
        pooling: Loại pooling ('gem', 'max', 'avg')
        device: Device
        skip_dataset: Bỏ qua bước tạo dataset (dùng data có sẵn)
    """
    print("\n" + "="*70)
    print("TRAINING REID MODEL")
    print("="*70)
    
    # Import train_reid functions
    try:
        from train_reid import (
            create_reid_dataset,
            create_reid_dataset_from_classified,
            train_reid_model,
            export_to_onnx
        )
    except ImportError as e:
        print(f"❌ Không thể import train_reid: {e}")
        return False
    
    # Kiểm tra train_dir
    if not Path(train_dir).exists():
        print(f"❌ Không tìm thấy folder {train_dir}")
        return False
    
    annotations_path = Path(train_dir) / 'annotations' / 'annotations.json'
    if not annotations_path.exists():
        print(f"❌ Không tìm thấy annotations.json tại {annotations_path}")
        return False
    
    samples_dir = Path(train_dir) / 'samples'
    if not samples_dir.exists():
        print(f"❌ Không tìm thấy samples folder tại {samples_dir}")
        return False
    
    print(f"\n📊 Dataset Information:")
    print(f"   Train dir: {train_dir}")
    print(f"   Annotations: {annotations_path}")
    print(f"   Samples: {samples_dir}")
    
    if use_classified:
        classified_path = Path(classified_dir)
        if not classified_path.exists():
            print(f"⚠️  Không tìm thấy {classified_dir}, sẽ extract từ video")
            use_classified = False
    
    # Step 1: Create dataset
    data_dict = None
    if not skip_dataset:
        print(f"\n📂 Creating dataset...")
        if use_classified:
            print(f"   Using classified data from {classified_dir}")
            try:
                data_dict = create_reid_dataset_from_classified(
                    train_dir=train_dir,
                    classified_dir=classified_dir,
                    output_dir='reid_dataset'
                )
            except Exception as e:
                print(f"⚠️  Failed to create from classified: {e}")
                print(f"   Falling back to video extraction...")
                use_classified = False
        
        if not use_classified:
            print(f"   Extracting from videos...")
            try:
                data_dict = create_reid_dataset(
                    train_dir=train_dir,
                    output_dir='reid_dataset'
                )
            except Exception as e:
                print(f"❌ Failed to create dataset: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Save data_dict
        import pickle
        with open('reid_data_dict.pkl', 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"   ✓ Dataset saved to reid_data_dict.pkl")
    else:
        # Load existing data_dict
        import pickle
        try:
            with open('reid_data_dict.pkl', 'rb') as f:
                data_dict = pickle.load(f)
            print(f"   ✓ Loaded existing dataset from reid_data_dict.pkl")
        except FileNotFoundError:
            print(f"❌ Không tìm thấy reid_data_dict.pkl")
            print(f"   Vui lòng chạy lại mà không có --skip-dataset")
            return False
    
    if data_dict is None or len(data_dict) == 0:
        print(f"❌ Dataset rỗng!")
        return False
    
    print(f"\n   Identities: {len(data_dict)}")
    total_images = sum(len(images) for images in data_dict.values())
    print(f"   Total images: {total_images}")
    print(f"   Avg images per identity: {total_images / len(data_dict):.1f}")
    
    # Step 2: Train model
    print(f"\n🚀 Starting training...")
    print(f"   Model: {model_path}")
    print(f"   Embedding dim: {embedding_dim}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Learning rate: {lr}")
    print(f"   Loss type: {loss_type}")
    print(f"   Input size: {input_height}x{input_width}")
    print(f"   Pooling: {pooling}")
    print(f"   Device: {device}")
    
    try:
        train_reid_model(
            data_dict=data_dict,
            output_model_path=model_path,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            num_epochs=epochs,
            lr=lr,
            device=device,
            loss_type=loss_type,
            input_size=(input_height, input_width),
            pooling=pooling
        )
        print(f"\n✅ Training completed!")
        print(f"   Model saved to: {model_path}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Export to ONNX
    print(f"\n📦 Exporting to ONNX...")
    try:
        num_classes = len(data_dict) if loss_type == 'crossentropy' else None
        export_to_onnx(
            model_path=model_path,
            onnx_path=onnx_path,
            embedding_dim=embedding_dim,
            input_size=(1, 3, input_height, input_width),
            pooling=pooling,
            num_classes=num_classes
        )
        print(f"   ✓ ONNX model saved to: {onnx_path}")
    except Exception as e:
        print(f"⚠️  ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO and/or ReID models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train cả YOLO và ReID
  python train_models.py --train-yolo --train-reid
  
  # Chỉ train YOLO
  python train_models.py --train-yolo --yolo-epochs 100
  
  # Chỉ train ReID
  python train_models.py --train-reid --reid-epochs 50
  
  # Train YOLO với model lớn hơn
  python train_models.py --train-yolo --yolo-model-size x
        """
    )
    
    # General arguments
    parser.add_argument('--train-yolo', action='store_true',
                       help='Train YOLO model')
    parser.add_argument('--train-reid', action='store_true',
                       help='Train ReID model')
    parser.add_argument('--device', default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    # YOLO arguments
    parser.add_argument('--yolo-data', default='data.yaml',
                       help='YOLO data.yaml path')
    parser.add_argument('--yolo-model-size', default='s',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size')
    parser.add_argument('--yolo-epochs', type=int, default=5,
                       help='YOLO training epochs')
    parser.add_argument('--yolo-img-size', type=int, default=640,
                       help='YOLO image size')
    parser.add_argument('--yolo-batch-size', type=int, default=20,
                       help='YOLO batch size')
    parser.add_argument('--yolo-project', default='runs/detect',
                       help='YOLO project directory')
    parser.add_argument('--yolo-name', default='yolo_drone_model',
                       help='YOLO model name')
    parser.add_argument('--yolo-resume', action='store_true',
                       help='Resume YOLO training from checkpoint')
    parser.add_argument('--yolo-resume-model', default=None,
                       help='YOLO checkpoint path to resume (e.g., yolo11s_best.pt)')
    
    # ReID arguments
    parser.add_argument('--reid-train-dir', default='observing/train',
                       help='ReID training data directory')
    parser.add_argument('--reid-classified-dir', default='classified_objects',
                       help='ReID classified objects directory')
    parser.add_argument('--reid-use-classified', action='store_true', default=True,
                       help='Use classified data for ReID')
    parser.add_argument('--reid-model-path', default='reid_model_best.pth',
                       help='ReID model output path')
    parser.add_argument('--reid-onnx-path', default='reid_model.onnx',
                       help='ReID ONNX output path')
    parser.add_argument('--reid-embedding-dim', type=int, default=256,
                       help='ReID embedding dimension')
    parser.add_argument('--reid-batch-size', type=int, default=64,
                       help='ReID batch size')
    parser.add_argument('--reid-epochs', type=int, default=20,
                       help='ReID training epochs')
    parser.add_argument('--reid-lr', type=float, default=0.001,
                       help='ReID learning rate')
    parser.add_argument('--reid-loss-type', default='crossentropy',
                       choices=['triplet', 'crossentropy'],
                       help='ReID loss type')
    parser.add_argument('--reid-input-height', type=int, default=256,
                       help='ReID input height')
    parser.add_argument('--reid-input-width', type=int, default=128,
                       help='ReID input width')
    parser.add_argument('--reid-pooling', default='gem',
                       choices=['gem', 'max', 'avg'],
                       help='ReID pooling type')
    parser.add_argument('--reid-skip-dataset', action='store_true',
                       help='Skip ReID dataset creation')
    
    args = parser.parse_args()
    
    # Kiểm tra arguments
    if not args.train_yolo and not args.train_reid:
        parser.print_help()
        print("\n❌ Vui lòng chọn ít nhất một model để train:")
        print("   --train-yolo  # Train YOLO")
        print("   --train-reid  # Train ReID")
        sys.exit(1)
    
    # Kiểm tra device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA không khả dụng, chuyển sang CPU")
        args.device = 'cpu'
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE")
    print("="*70)
    print(f"Device: {args.device}")
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    # Train YOLO
    yolo_success = True
    if args.train_yolo:
        yolo_success = train_yolo(
            data_yaml=args.yolo_data,
            model_size=args.yolo_model_size,
            epochs=args.yolo_epochs,
            img_size=args.yolo_img_size,
            batch_size=args.yolo_batch_size,
            device=args.device,
            project=args.yolo_project,
            name=args.yolo_name,
            resume=args.yolo_resume,
            resume_model=args.yolo_resume_model
        )
    
    # Train ReID
    reid_success = True
    if args.train_reid:
        reid_success = train_reid(
            train_dir=args.reid_train_dir,
            classified_dir=args.reid_classified_dir,
            use_classified=args.reid_use_classified,
            model_path=args.reid_model_path,
            onnx_path=args.reid_onnx_path,
            embedding_dim=args.reid_embedding_dim,
            batch_size=args.reid_batch_size,
            epochs=args.reid_epochs,
            lr=args.reid_lr,
            loss_type=args.reid_loss_type,
            input_height=args.reid_input_height,
            input_width=args.reid_input_width,
            pooling=args.reid_pooling,
            device=args.device,
            skip_dataset=args.reid_skip_dataset
        )
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    if args.train_yolo:
        status = "✅ SUCCESS" if yolo_success else "❌ FAILED"
        print(f"YOLO: {status}")
    if args.train_reid:
        status = "✅ SUCCESS" if reid_success else "❌ FAILED"
        print(f"ReID: {status}")
    print("="*70)
    
    if (args.train_yolo and not yolo_success) or (args.train_reid and not reid_success):
        sys.exit(1)


if __name__ == "__main__":
    main()