# object-detectors/DINO/infer_onnx.py

import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import cv2
import onnxruntime

# --- DINO-specific imports for post-processing ---
from models.registry import MODULE_BUILD_FUNCS
from util.slconfig import SLConfig
import datasets.transforms as T

def get_args_parser():
    """Parses command-line arguments for ONNX inference."""
    parser = argparse.ArgumentParser('DINO ONNX Inference Script', add_help=False)
    
    parser.add_argument('--onnx_path', '-m', type=str, required=True, help="Path to the exported ONNX model file.")
    parser.add_argument('--config_file', '-c', type=str, required=True, help="Path to the original model config file for post-processing setup.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to a single image for inference.")
    
    parser.add_argument('--output_dir', default='onnx_inference_results', help="Directory to save the output visualization.")
    parser.add_argument('--threshold', type=float, default=0.3, help="Confidence threshold to filter predictions.")
    parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'], help="Device to run inference on ('cpu' or 'gpu').")
    
    return parser

def main(args):
    """Main function to run inference with the ONNX model."""
    print(f"Loading ONNX model for {args.device.upper()} execution...")
    
    # --- 1. Set up ONNX Runtime Session ---
    providers = ['CUDAExecutionProvider'] if args.device == 'gpu' else ['CPUExecutionProvider']
    try:
        session = onnxruntime.InferenceSession(args.onnx_path, providers=providers)
    except Exception as e:
        print(f"Error loading ONNX model. Make sure onnxruntime-gpu is installed if using --device gpu. Error: {e}")
        return

    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    print(f"Running on provider: {session.get_providers()}")

    # --- 2. Load DINO Config for Post-Processor ---
    cfg = SLConfig.fromfile(args.config_file)
    _, _, postprocessors = MODULE_BUILD_FUNCS.get(cfg.modelname)(cfg)
    postprocessor = postprocessors['bbox']
    
    # --- 3. Prepare Image and Inputs ---
    img_path = Path(args.image_path)
    original_img = Image.open(img_path).convert("RGB")

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor, _ = transform(original_img, None)
    
    images_np = image_tensor.unsqueeze(0).numpy()
    # Create a dummy mask. For single image inference without padding, it's all False.
    mask_np = np.zeros((1, images_np.shape[2], images_np.shape[3]), dtype=bool)
    
    onnx_inputs = {input_names[0]: images_np, input_names[1]: mask_np}

    # --- 4. Run Inference ---
    print(f"Running inference on {img_path.name}...")
    onnx_outputs = session.run(output_names, onnx_inputs)
    pred_logits_np, pred_boxes_np = onnx_outputs

    # --- 5. Post-process the Output ---
    outputs = {'pred_logits': torch.from_numpy(pred_logits_np), 'pred_boxes': torch.from_numpy(pred_boxes_np)}
    orig_size = torch.as_tensor([original_img.height, original_img.width]).unsqueeze(0)
    results = postprocessor(outputs, orig_size)[0]
    
    scores = results['scores']
    labels = results['labels']
    boxes = results['boxes']

    keep = scores > args.threshold
    
    # --- 6. Visualize and Save ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Replace with your actual class names if needed
    CLASS_LABELS = {i: f'class_{i}' for i in range(cfg.num_classes)}

    visualize_and_save(
        original_img, boxes[keep], labels[keep], scores[keep],
        CLASS_LABELS, output_dir / f"pred_{img_path.name}"
    )
    print(f"\nInference complete. Output saved in: {output_dir}")

def visualize_and_save(image, boxes, labels, scores, class_labels, path):
    """Draws prediction boxes on an image and saves it."""
    img_cv = np.array(image.copy())
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    for box, label_id, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        x1, y1, x2, y2 = map(int, box)
        label_text = class_labels.get(label_id, f"CLS-{label_id}")
        color = (0, 255, 0)
        
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        text = f"{label_text}: {score:.2f}"
        cv2.putText(img_cv, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    cv2.imwrite(str(path), img_cv)
    print(f"  Saved visualization to {path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO ONNX Inference Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)