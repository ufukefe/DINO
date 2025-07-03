#!/usr/bin/env python3
r"""
infer_onnx.py
=============

Run inference with a DINO ONNX model. This script is refactored to use the
official `PostProcess` class from the DINO project, ensuring its output is
identical to the PyTorch inference script (`infer.py`).

This approach guarantees consistency and robustness by eliminating duplicated code.

Example
-------
python object-detectors/DINO/infer_onnx.py \
    -m dino_swinL_dynamic.onnx \
    --image_path "/path/to/image.png" \
    --output_dir "inference_results/single_image_onnx" \
    --threshold 0.5 \
    --nms_threshold 0.5 \
    --num_select 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import onnxruntime as ort
import torch

# --- 1. Project-level Imports for Robustness ---
# Add the project's root directory to the Python path to allow direct imports
# of modules like `PostProcess`. This is crucial for code reuse.
DINO_ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(DINO_ROOT))

from models.dino.dino import PostProcess
# ------------------------------------------------

# --- 2. Visualization Configuration ---
# Hardcoded class names and colors for consistent visualization.
CLASS_NAMES = {
    0: "Elektrik Diregi",
    1: "Trafo"
}
CLASS_COLORS_RGB = {
    0: (0, 0, 255),      # Blue for Elektrik Diregi
    1: (255, 0, 0),      # Red for Trafo
}
DEFAULT_COLOR_RGB = (0, 255, 0) # Green for any other class

# -----------------------------------------------------------------------------#
#                               Helper functions                               #
# -----------------------------------------------------------------------------#
def load_image(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads an image and preprocesses it for DINO model inference.

    Returns
    -------
    img_rgb : np.ndarray
        The original image in RGB format, shape [H, W, 3], type uint8.
    img_proc: np.ndarray
        The processed image tensor for the model, shape [1, 3, H, W], type float32.
    """
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_float = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    normalized_img = (img_float - mean) / std

    img_proc = np.transpose(normalized_img, (2, 0, 1))[None]
    return img_rgb, img_proc.astype(np.float32)


def draw(
    img: np.ndarray,
    detections: List[Tuple[int, float, Tuple[int, int, int, int]]],
) -> np.ndarray:
    """Draws detection boxes and labels on an image with class-specific colors."""
    out = img.copy()
    
    for lbl, score, (x0, y0, x1, y1) in detections:
        # Look up the color and name from our predefined constants
        color = CLASS_COLORS_RGB.get(lbl, DEFAULT_COLOR_RGB)
        name = CLASS_NAMES.get(lbl, f"cls_{lbl}")
        
        # Draw the bounding box
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        
        label_text = f"{name}: {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw a filled rectangle as a background for the label
        cv2.rectangle(out, (x0, y0 - text_h - baseline), (x0 + text_w, y0), color, -1)
        
        # Put the label text on top of the background
        cv2.putText(
            out,
            label_text,
            (x0, y0 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text for good contrast
            1,
            cv2.LINE_AA,
        )
    return out


# -----------------------------------------------------------------------------#
#                                 CLI parsing                                  #
# -----------------------------------------------------------------------------#
def get_args() -> argparse.Namespace:
    """Parses and returns command-line arguments."""
    ap = argparse.ArgumentParser("DINO-ONNX inference with official PostProcess")
    ap.add_argument("-m", "--model", required=True, type=str, help="Path to the ONNX model file.")
    ap.add_argument("--image_path", required=True, type=str, help="Path to the input image.")
    ap.add_argument("--output_dir", default="onnx_infer_out", type=str, help="Directory to save output images.")
    ap.add_argument("--threshold", default=0.5, type=float, help="Confidence score threshold for filtering final detections.")
    ap.add_argument("--nms_threshold", default=0.5, type=float, help="IoU threshold for NMS. Set to -1 to disable.")
    ap.add_argument("--num_select", default=10, type=int, help="Number of top-k predictions to select (a.k.a. num_queries).")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Execution provider ('cpu' or 'cuda').")
    return ap.parse_args()


# -----------------------------------------------------------------------------#
#                                    Main                                      #
# -----------------------------------------------------------------------------#
def main() -> None:
    """Main execution function."""
    args = get_args()
    img_rgb, img_in = load_image(args.image_path)
    H, W, _ = img_rgb.shape

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.device == "cuda"
        else ["CPUExecutionProvider"]
    )
    
    print(f"Loading ONNX model from {args.model} for {args.device.upper()} execution...")
    sess = ort.InferenceSession(str(args.model), providers=providers)
    inp_name = sess.get_inputs()[0].name
    
    print("Running inference...")
    logits_np, boxes_np = sess.run(None, {inp_name: img_in})

    # --- Use the Official PostProcess Class ---
    postprocessor = PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_threshold)
    outputs = {
        'pred_logits': torch.from_numpy(logits_np),
        'pred_boxes': torch.from_numpy(boxes_np)
    }
    target_sizes = torch.tensor([[H, W]])
    
    print("Post-processing detections using the official PostProcess module...")
    results = postprocessor(outputs, target_sizes)
    
    # --- Apply Final Confidence Thresholding ---
    scores = results[0]['scores']
    labels = results[0]['labels']
    boxes = results[0]['boxes']
    
    keep = scores > args.threshold
    
    final_detections = [
        (int(label), float(score), tuple(map(int, box)))
        for label, score, box in zip(labels[keep], scores[keep], boxes[keep])
    ]

    print(
        f"Found {len(final_detections)} final detections "
        f"(score ≥ {args.threshold}, IoU ≤ {args.nms_threshold}, top-k={args.num_select})"
    )

    vis = draw(img_rgb, final_detections)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(args.image_path).stem
    out_filename = f"{base_name}_det_s{args.threshold}_n{args.nms_threshold}_k{args.num_select}.png"
    out_path = out_dir / out_filename
    
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to: {out_path}")


if __name__ == "__main__":
    main()