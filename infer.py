import argparse
import json
from pathlib import Path
import time  # Import the time module

import torch
import numpy as np
from PIL import Image
import cv2

import datasets.transforms as T
from models.registry import MODULE_BUILD_FUNCS
from util.slconfig import SLConfig
from util.utils import to_device
from util import box_ops

def get_args_parser():
    """Parses command-line arguments for inference."""
    parser = argparse.ArgumentParser('DINO Inference Script', add_help=False)
    
    # Model and config
    parser.add_argument('--config_file', '-c', type=str, required=True,
                        help="Path to the model configuration file.")
    parser.add_argument('--resume', '-r', type=str, required=True,
                        help="Path to the model checkpoint for inference.")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_path', type=str, help="Path to a single image.")
    group.add_argument('--image_dir', type=str, help="Path to a directory of images.")
    group.add_argument('--coco_path', type=str, help="Path to COCO-style dataset root.")
    
    # --- NEW: Argument to select the dataset split ---
    parser.add_argument('--split', default='val', choices=['train', 'val', 'test'],
                        help="Dataset split to use when --coco_path is provided.")
    
    # Output options
    parser.add_argument('--output_dir', default='outputs',
                        help="Path to save the output visualizations.")
    parser.add_argument('--threshold', type=float, default=0.3,
                        help="Confidence threshold to filter predictions.")
    
    # Other
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    return parser

def main(args):
    """Main function to run the inference."""
    # --- Setup ---
    print("Loading configuration...")
    cfg = SLConfig.fromfile(args.config_file)
    
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    
    device = torch.device(cfg.device)
    
    # --- Build Model ---
    print("Building model...")
    build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
    model, _, postprocessors = build_func(cfg)
    model.to(device)
    
    # --- Load Checkpoint ---
    print(f"Loading checkpoint from {args.resume}...")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # --- Prepare Output Directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Define Class Labels ---
    CLASS_LABELS = {
        0: 'elektrik-diregi-bbox',
        1: 'trafo-bbox'
    }
    PRED_COLOR = (0, 255, 0)  # Green for predictions
    GT_COLOR = (255, 0, 0)   # Blue for ground truth

    # --- Prepare Image Paths and Ground Truths ---
    image_paths = []
    targets_by_filename = {}
    
    if args.image_path:
        image_paths.append(Path(args.image_path))
    elif args.image_dir:
        image_paths = list(Path(args.image_dir).glob('*.png')) + \
                      list(Path(args.image_dir).glob('*.jpg'))
    elif args.coco_path:
        coco_root = Path(args.coco_path)
        split_dir_name = f"{args.split}2017"
        ann_file = coco_root / "annotations" / f"instances_{args.split}2017.json"
        
        print(f"Loading annotations from: {ann_file}")
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        img_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            filename = img_id_to_filename.get(img_id)
            if not filename:
                continue

            if filename not in targets_by_filename:
                targets_by_filename[filename] = {'boxes': [], 'labels': []}
            
            box = ann['bbox']
            xtl, ytl, w, h = box
            targets_by_filename[filename]['boxes'].append([xtl, ytl, xtl + w, ytl + h])
            targets_by_filename[filename]['labels'].append(ann['category_id'])

        for img_info in coco_data['images']:
            image_paths.append(coco_root / split_dir_name / img_info['file_name'])
            
    # ==============================================================================
    # --- WARM-UP RUN (Important for accurate GPU timing) ---
    # The first inference can have overhead. We run a dummy inference to warm up.
    # ==============================================================================
    if device.type == 'cuda':
        print("Performing a warm-up run on the GPU...")
        dummy_input = torch.rand(1, 3, 800, 800).to(device)
        with torch.no_grad():
            model(dummy_input)
        torch.cuda.synchronize()
        print("Warm-up complete.")

    # ==============================================================================
    # --- TIMING SETUP ---
    # ==============================================================================
    # For pure model inference time
    if device.type == 'cuda':
        # Use torch.cuda.Event for precise GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    total_pure_inference_time_ms = 0
    total_processing_time_s = 0
    processed_image_count = 0

    # --- Inference Loop ---
    print(f"Found {len(image_paths)} images to process.")
    for img_path in image_paths:
        if not img_path.exists():
            print(f"  Image not found at {img_path}, skipping.")
            continue
            
        print(f"Processing: {img_path.name}")
        

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Could not open image {img_path}, skipping. Error: {e}")
            continue

        # --- START: Total Processing Time Measurement ---
        start_total_time = time.time()

        # Pre-processing
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(img, None)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            # --- START: Pure Model Inference Time Measurement ---
            if device.type == 'cuda':
                start_event.record()
            else: # For CPU
                start_inference_time = time.time()

            outputs = model(image_tensor)

            if device.type == 'cuda':
                end_event.record()
                # Crucially, wait for the GPU to finish the work
                torch.cuda.synchronize()
                # Calculate elapsed time in milliseconds
                pure_inference_time_ms = start_event.elapsed_time(end_event)
            else: # For CPU
                end_inference_time = time.time()
                pure_inference_time_ms = (end_inference_time - start_inference_time) * 1000
            # --- END: Pure Model Inference Time Measurement ---
        
        # Post-processing
        orig_size = torch.as_tensor([img.height, img.width]).unsqueeze(0).to(device)
        results = postprocessors["bbox"](outputs, orig_size)
        
        scores = results[0]['scores']
        labels = results[0]['labels']
        boxes = results[0]['boxes']

        keep = scores > args.threshold

        # --- END: Total Processing Time Measurement ---
        end_total_time = time.time()
        total_img_processing_time_s = end_total_time - start_total_time

        # Accumulate times for averaging
        total_pure_inference_time_ms += pure_inference_time_ms
        total_processing_time_s += total_img_processing_time_s
        processed_image_count += 1
        
        # Print per-image results
        print(f"  - Pure Model Inference Time: {pure_inference_time_ms:.2f} ms")
        print(f"  - Total Processing Time (load+transform+infer+post): {total_img_processing_time_s:.4f} s")
        
        visualize_and_save(
            original_img=img,
            pred_boxes=boxes[keep],
            pred_labels=[CLASS_LABELS.get(l.item(), "Unknown") for l in labels[keep]],
            pred_scores=scores[keep],
            ground_truth=targets_by_filename.get(img_path.name),
            gt_class_labels=CLASS_LABELS,
            output_path=output_dir / img_path.name,
            pred_color=PRED_COLOR,
            gt_color=GT_COLOR
        )

    print("\n--- Inference Complete ---")
    print(f"Outputs saved to: {args.output_dir}")

    # ==============================================================================
    # --- FINAL AVERAGE TIMING REPORT ---
    # ==============================================================================
    if processed_image_count > 0:
        avg_pure_inference_ms = total_pure_inference_time_ms / processed_image_count
        avg_processing_s = total_processing_time_s / processed_image_count
        print("\n--- Average Performance ---")
        print(f"Processed {processed_image_count} images.")
        print(f"Average Pure Model Inference Time: {avg_pure_inference_ms:.2f} ms per image")
        print(f"Average Total Processing Time: {avg_processing_s:.4f} s per image ({1/avg_processing_s:.2f} FPS)")
    else:
        print("\nNo images were processed.")


def visualize_and_save(original_img, pred_boxes, pred_labels, pred_scores, ground_truth, gt_class_labels, output_path, pred_color, gt_color):
    """Draws prediction and ground truth boxes on an image and saves it."""
    img_cv = np.array(original_img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    if ground_truth:
        for box, label_id in zip(ground_truth['boxes'], ground_truth['labels']):
            x1, y1, x2, y2 = map(int, box)
            label_text = gt_class_labels.get(label_id, "Unknown GT")
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), gt_color, 2)
            cv2.putText(img_cv, f"GT: {label_text}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 2)

    for box, label, score in zip(pred_boxes.tolist(), pred_labels, pred_scores.tolist()):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), pred_color, 2)
        text = f"Pred: {label} ({score:.2f})"
        cv2.putText(img_cv, text, (x1, y1 + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 2)
        
    cv2.imwrite(str(output_path), img_cv)
    print(f"  Saved visualization to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO Inference Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)