#!/usr/bin/env python3
"""
infer_onnx.py
=============

Run inference with a DINO ONNX model that was exported with `export_to_onnx.py`.

Example
-------
python infer_onnx.py \
    -c object-detectors/DINO/config/DINO/custom_dataset_swin.py \    # ignored
    -r object-detectors/DINO/logs/DINO/custom_training_swinL_from_scratch_resumed/checkpoint_best_regular.pth \   # ignored
    -m dino_swinL.onnx \
    --image_path "/path/to/image.png" \
    --output_dir "inference_results/single_image_test" \
    --threshold 0.4 \
    --device cuda               # or cpu
"""

from __future__ import annotations

import argparse
import os
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
    Returns
    -------
    img_rgb : np.ndarray  uint8  [H, W, 3]  RGB
    img_proc: np.ndarray  float32 [1, 3, H, W]  normalized for DINO
    """
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))[None]        # 1×3×H×W
    return img_rgb, img.astype(np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def cxcywh_to_xyxy_norm(boxes: np.ndarray) -> np.ndarray:
    """(cx,cy,w,h) → (x0,y0,x1,y1)   all values still in [0,1]"""
    cxy = boxes[..., :2]
    wh  = boxes[..., 2:]
    xy0 = cxy - 0.5 * wh
    xy1 = cxy + 0.5 * wh
    return np.concatenate([xy0, xy1], axis=-1)


def postprocess(
    logits: np.ndarray,
    boxes: np.ndarray,
    prob_thres: float,
    img_size: Tuple[int, int],
) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
    """
    Returns list of (label, score, (x0,y0,x1,y1)) in pixel coordinates.
    """
    H, W = img_size
    probs = softmax(logits, axis=-1)                 # [Nq, C]
    scores = probs[..., 1:]                          # drop class 0 = "no-object"

    labels = np.argmax(scores, axis=-1)
    confs  = scores[np.arange(scores.shape[0]), labels]

    keep = confs > prob_thres
    
    # Add basic NMS suppression
    if boxes.shape[1] == 4:  # (cx,cy,w,h)
        boxes = cxcywh_to_xyxy_norm(boxes)
    elif boxes.shape[1] == 5:  # (cx,cy,w,h, conf)
        boxes = cxcywh_to_xyxy_norm(boxes[:, :-1])
        confs = boxes[:, -1]
    else:
        raise ValueError(f"Unexpected box shape {boxes.shape}")
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes[keep].tolist(), confs[keep].tolist(),
        score_threshold=prob_thres,
        nms_threshold=0.5,
        top_k=10, # limit to top 10 detections
    )
    if indices is not None:
        keep[keep] = np.isin(np.arange(len(keep)), indices.flatten())
    else:
        keep[keep] = False
    # If no boxes are left after NMS, return empty list
    if not keep.any():
        return []

    boxes_xyxy = cxcywh_to_xyxy_norm(boxes[keep])    # [K, 4]
    boxes_xyxy[:, [0, 2]] *= W
    boxes_xyxy[:, [1, 3]] *= H
    boxes_xyxy = boxes_xyxy.round().astype(int)

    return [
        (int(lbl), float(score), tuple(bbox))
        for lbl, score, bbox in zip(labels[keep], confs[keep], boxes_xyxy)
    ]


def draw(
    img: np.ndarray,
    detections: List[Tuple[int, float, Tuple[int, int, int, int]]],
    class_names: List[str] | None = None,
) -> np.ndarray:
    out = img.copy()
    for lbl, score, (x0, y0, x1, y1) in detections:
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 2)
        name = class_names[lbl] if class_names else f"cls{lbl}"
        cv2.putText(
            out,
            f"{name}:{score:.2f}",
            (x0, y0 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


# -----------------------------------------------------------------------------#
#                                 CLI parsing                                  #
# -----------------------------------------------------------------------------#
def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("DINO-ONNX inference")
    ap.add_argument("-c", "--config_file", type=str, help="(ignored)")
    ap.add_argument("-r", "--resume", type=str, help="(ignored)")
    ap.add_argument("-m", "--model", required=True, type=str, help="ONNX model")
    ap.add_argument("--image_path", required=True, type=str, help="Path to image")
    ap.add_argument("--output_dir", default="onnx_infer_out", type=str)
    ap.add_argument("--threshold", default=0.4, type=float, help="Score threshold")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return ap.parse_args()


# -----------------------------------------------------------------------------#
#                                    Main                                      #
# -----------------------------------------------------------------------------#
def main() -> None:
    args = get_args()
    img_rgb, img_in = load_image(args.image_path)
    H, W, _ = img_rgb.shape

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.device == "cuda"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(args.model), providers=providers)
    inp_name = sess.get_inputs()[0].name
    logits, boxes = sess.run(None, {inp_name: img_in})

    logits = logits[0]     # (Nq, C)
    boxes  = boxes[0]      # (Nq, 4)

    detections = postprocess(logits, boxes, args.threshold, (H, W))
    print(f"{len(detections)} detections ≥ {args.threshold}")

    vis = draw(img_rgb, detections)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(args.image_path).name.replace(".", "_det.")
    cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"saved → {out_path}")

    # optionally dump raw detections
    det_txt = out_path.with_suffix(".txt")
    with open(det_txt, "w") as f:
        for lbl, score, (x0, y0, x1, y1) in detections:
            f.write(f"{lbl} {score:.4f} {x0} {y0} {x1} {y1}\n")


if __name__ == "__main__":
    main()
