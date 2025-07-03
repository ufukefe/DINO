#!/usr/bin/env python3
r"""
infer_onnx.py
=============

Run inference with a DINO ONNX model that was exported with `export_to_onnx.py`.
This script includes a robust post-processing pipeline that precisely mimics the
candidate selection of the original DINO implementation and then applies
Non-Maximum Suppression (NMS) for clean, final detections.

Example
-------
# Standard filtering
python object-detectors/DINO/infer_onnx.py \
    -m dino_swinL_dynamic.onnx \
    --image_path "/path/to/image.png" \
    --output_dir "inference_results/single_image_onnx" \
    --threshold 0.5 \
    --nms_threshold 0.5 \
    --num_select 300

# For stricter filtering (fewer boxes)
python object-detectors/DINO/infer_onnx.py \
    -m dino_swinL_dynamic.onnx \
    --image_path "/path/to/image.png" \
    --output_dir "inference_results/single_image_onnx" \
    --threshold 0.6 \
    --nms_threshold 0.4 \
    --num_select 300
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import onnxruntime as ort


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

    # Normalize the image
    img_float = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    normalized_img = (img_float - mean) / std

    # Transpose to NCHW format
    img_proc = np.transpose(normalized_img, (2, 0, 1))[None]
    return img_rgb, img_proc.astype(np.float32)


def box_cxcywh_to_xyxy(x: np.ndarray) -> np.ndarray:
    """(center_x, center_y, w, h) -> (x0, y0, x1, y1)"""
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [
        (x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)
    ]
    return np.stack(b, axis=-1)


def postprocess(
    logits: np.ndarray,
    boxes: np.ndarray,
    prob_thres: float,
    nms_thres: float,
    img_size: Tuple[int, int],
    num_select: int = 300,
) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
    """
    Replicates the DINO PostProcess module and applies NMS.

    Parameters
    ----------
    logits : np.ndarray
        Raw logits from the model, shape [num_queries, num_classes].
    boxes : np.ndarray
        Box predictions from the model in (cx, cy, w, h) normalized format.
    prob_thres : float
        Probability threshold to filter detections before NMS.
    nms_thres : float
        IoU threshold for NMS.
    img_size : Tuple[int, int]
        The (height, width) of the original image.
    num_select : int
        The number of top-scoring predictions to consider across all queries and classes.

    Returns
    -------
    List[Tuple[int, float, Tuple[int, int, int, int]]]
        Final detections: (class_id, score, (x0, y0, x1, y1)_pixels).
    """
    H, W = img_size

    # 1. Candidate selection using global top-k (mimics PostProcess class)
    # The 'no object' class is typically excluded in the model export or ignored here.
    # If logits include it, they should be sliced `logits[..., :-1]`
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    
    # Flatten scores and find top `num_select` scores
    scores_flat = probs.flatten()
    topk_indices = np.argsort(scores_flat)[-num_select:]
    
    # Get scores, labels, and corresponding query indices
    scores = scores_flat[topk_indices]
    topk_boxes_queries = topk_indices // probs.shape[1]
    labels = topk_indices % probs.shape[1]
    
    # Select the boxes associated with the top queries
    boxes_selected = boxes[topk_boxes_queries]

    # 2. Filter by confidence threshold
    keep_by_threshold = scores > prob_thres
    scores = scores[keep_by_threshold]
    labels = labels[keep_by_threshold]
    boxes_selected = boxes_selected[keep_by_threshold]

    if boxes_selected.shape[0] == 0:
        return []

    # 3. Convert boxes to pixel coordinates [x0, y0, x1, y1]
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_selected)
    boxes_xyxy[:, [0, 2]] *= W
    boxes_xyxy[:, [1, 3]] *= H

    # 4. Apply Non-Maximum Suppression (NMS) for each class
    final_detections = []
    for class_id in np.unique(labels):
        class_mask = (labels == class_id)
        class_boxes = boxes_xyxy[class_mask]
        class_scores = scores[class_mask]

        if class_boxes.shape[0] == 0:
            continue

        # OpenCV's NMSBoxes requires (x, y, w, h) format
        x, y = class_boxes[:, 0], class_boxes[:, 1]
        w, h = class_boxes[:, 2] - x, class_boxes[:, 3] - y
        cv2_boxes = np.column_stack([x, y, w, h]).tolist()

        indices = cv2.dnn.NMSBoxes(
            bboxes=cv2_boxes,
            scores=class_scores.tolist(),
            score_threshold=prob_thres,
            nms_threshold=nms_thres
        )

        if len(indices) > 0:
            kept_indices = indices.flatten()
            for idx in kept_indices:
                bbox = tuple(map(int, class_boxes[idx]))
                score = float(class_scores[idx])
                final_detections.append((int(class_id), score, bbox))

    return final_detections


def draw(
    img: np.ndarray,
    detections: List[Tuple[int, float, Tuple[int, int, int, int]]],
    class_names: List[str] | None = None,
) -> np.ndarray:
    """Draws detection boxes and labels on an image."""
    out = img.copy()
    for lbl, score, (x0, y0, x1, y1) in detections:
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)  # Red boxes
        name = class_names[lbl] if class_names else f"cls_{lbl}"
        label_text = f"{name}: {score:.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x0, y0 - text_h - baseline), (x0 + text_w, y0), (0, 0, 255), -1)
        cv2.putText(
            out,
            label_text,
            (x0, y0 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # White text
            1,
            cv2.LINE_AA,
        )
    return out


# -----------------------------------------------------------------------------#
#                                 CLI parsing                                  #
# -----------------------------------------------------------------------------#
def get_args() -> argparse.Namespace:
    """Parses and returns command-line arguments."""
    ap = argparse.ArgumentParser("DINO-ONNX inference with NMS")
    ap.add_argument("-m", "--model", required=True, type=str, help="Path to the ONNX model file.")
    ap.add_argument("--image_path", required=True, type=str, help="Path to the input image.")
    ap.add_argument("--output_dir", default="onnx_infer_out", type=str, help="Directory to save output images.")
    ap.add_argument("--threshold", default=0.5, type=float, help="Confidence score threshold for filtering detections.")
    ap.add_argument("--nms_threshold", default=0.5, type=float, help="IoU threshold for Non-Maximum Suppression.")
    ap.add_argument("--num_select", default=300, type=int, help="Number of top-k predictions to select before NMS.")
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
    logits, boxes = sess.run(None, {inp_name: img_in})

    logits = logits[0]
    boxes = boxes[0]

    print("Post-processing detections...")
    detections = postprocess(
        logits, boxes, args.threshold, args.nms_threshold, (H, W), args.num_select
    )
    print(
        f"Found {len(detections)} final detections "
        f"(score ≥ {args.threshold}, IoU ≤ {args.nms_threshold}, top-k={args.num_select})"
    )

    vis = draw(img_rgb, detections)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(args.image_path).stem
    out_filename = f"{base_name}_det_s{args.threshold}_n{args.nms_threshold}_k{args.num_select}.png"
    out_path = out_dir / out_filename
    
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to: {out_path}")

    det_txt = out_path.with_suffix(".txt")
    with open(det_txt, "w") as f:
        f.write("# class_id score x_top_left y_top_left x_bottom_right y_bottom_right\n")
        for lbl, score, (x0, y0, x1, y1) in detections:
            f.write(f"{lbl} {score:.4f} {x0} {y0} {x1} {y1}\n")


if __name__ == "__main__":
    main()