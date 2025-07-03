#!/usr/bin/env python3
"""
export_to_onnx.py – DINO → ONNX (dynamic H/W/B)

Example
-------
python export_to_onnx.py \
  -c object-detectors/DINO/config/DINO/custom_dataset_swin.py \
  -r object-detectors/DINO/logs/DINO/custom_training_swinL_from_scratch_resumed/checkpoint_best_regular.pth \
  -o dino_swinL_dynamic.onnx \
  --input_size 1080 1920 \
  --batch_size 1 \
  --device cpu \
  --opset 17 \
  --validate
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path

import torch
import torch.onnx

# ────────────────────────────────────────────────────────────────────────────────
# ⛑ 1. Disable the custom CUDA kernel so we can export on any machine
# ────────────────────────────────────────────────────────────────────────────────
from models.dino.ops.functions import ms_deform_attn_func as _msda_py
def _safe_forward(ctx, value, spatial_shapes, lvl_start_idx,
                  sampling_locations, attn_weights, im2col_step):
    # pure-PyTorch reference implementation (slow but export-friendly)
    return _msda_py.ms_deform_attn_core_pytorch(
        value, spatial_shapes, sampling_locations, attn_weights)
_msda_py.MSDeformAttnFunction.forward = staticmethod(_safe_forward)


# ────────────────────────────────────────────────────────────────────────────────
# 2. CLI
# ────────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("Export trained DINO model to ONNX")
    p.add_argument("-c", "--config_file", required=True)
    p.add_argument("-r", "--resume",       required=True, help="checkpoint *.pth")
    p.add_argument("-o", "--output",       default="dino_model.onnx")
    p.add_argument("--input_size",  nargs=2, type=int, default=[1080, 1920],
                   metavar=("H", "W"), help="dummy H W for tracing")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--opset",      type=int, default=17)
    p.add_argument("--device",     choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--validate",   action="store_true",
                   help="run a quick onnxruntime forward pass")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────────
# 3. Model builder – identical to infer.py but returns a wrapper with tuple outs
# ────────────────────────────────────────────────────────────────────────────────
def build_model(cfg_path: Path, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    # make repo import-able when script is called from anywhere
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root))

    from util.slconfig import SLConfig
    from models.registry import MODULE_BUILD_FUNCS

    cfg = SLConfig.fromfile(str(cfg_path))
    cfg.device = str(device)

    # ONNX can't handle torch.utils.checkpoint
    if getattr(cfg, "use_checkpoint", False):
        print("[export] use_checkpoint → False")
        cfg.use_checkpoint = False

    build_fn = MODULE_BUILD_FUNCS.get(cfg.modelname)
    if build_fn is None:
        raise KeyError(f"Unknown model '{cfg.modelname}'")

    model, _, _ = build_fn(cfg)

    #  🔑  Torch ≥2.6 defaulted weights_only=True; we need full pickle objects
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # tolerant load – warn if anything is missing / unexpected
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        warnings.warn(f"Missing keys: {missing}\nUnexpected: {unexpected}")

    model.to(device).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, net): super().__init__(); self.net = net
        def forward(self, x):
            o = self.net(x)
            return o["pred_logits"], o["pred_boxes"]
    return Wrapper(model)


# ────────────────────────────────────────────────────────────────────────────────
# 4. Export helper
# ────────────────────────────────────────────────────────────────────────────────
def export_onnx(model: torch.nn.Module, dummy: torch.Tensor,
                out_path: Path, opset: int):
    torch.onnx.export(
        model, dummy, str(out_path),
        opset_version=opset,
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
        dynamic_axes={
            "images":       {0: "batch", 2: "height", 3: "width"},
            "pred_logits":  {0: "batch"},
            "pred_boxes":   {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"[export] ONNX graph saved ▶ {out_path.resolve()}")


# ────────────────────────────────────────────────────────────────────────────────
# 5. Optional sanity-check
# ────────────────────────────────────────────────────────────────────────────────
def validate(out_path: Path, dummy: torch.Tensor):
    try:
        import onnxruntime as ort
    except ImportError:
        warnings.warn("onnxruntime not installed – skipping validation")
        return
    sess = ort.InferenceSession(
        str(out_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    logits, boxes = sess.run(None, {sess.get_inputs()[0].name: dummy.cpu().numpy()})
    print(f"[validate] OK – logits {logits.shape}, boxes {boxes.shape}")


# ────────────────────────────────────────────────────────────────────────────────
# 6. Main
# ────────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device(args.device)

    print("‣ building model (this may take ~1-2 min)…")
    model  = build_model(Path(args.config_file), Path(args.resume), device)

    H, W   = args.input_size
    dummy  = torch.randn(args.batch_size, 3, H, W, device=device)
    print(f"‣ dummy tensor {tuple(dummy.shape)}  (H={H} W={W})")

    export_onnx(model, dummy, Path(args.output), args.opset)

    if args.validate:
        print("‣ ONNXRuntime check…")
        validate(Path(args.output), dummy)


if __name__ == "__main__":
    main()
