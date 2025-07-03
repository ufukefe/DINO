# object-detectors/DINO/export_onnx.py (Final Simplified Version)

import argparse
import torch
import torch.nn as nn
from pathlib import Path

from models.registry import MODULE_BUILD_FUNCS
from util.slconfig import SLConfig
from util.utils import clean_state_dict
from util.misc import NestedTensor

class DINOOnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, mask):
        samples = NestedTensor(images, mask)
        outputs = self.model(samples)
        return outputs['pred_logits'], outputs['pred_boxes']

def get_args_parser():
    parser = argparse.ArgumentParser('DINO ONNX Export Script', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True, help="Path to config.")
    parser.add_argument('--resume', '-r', type=str, required=True, help="Path to checkpoint.")
    parser.add_argument('--output_root', default='exported_models', help="Root directory for saved models.")
    parser.add_argument('--opset_version', type=int, default=16, help="ONNX opset version.")
    parser.add_argument('--height', type=int, default=768, help="Image height for dummy input.")
    parser.add_argument('--width', type=int, default=1152, help="Image width for dummy input.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for dummy input.")
    parser.add_argument('--device', default='cuda', help='Device for export.')
    return parser

def main(args):
    print("Loading model configuration...")
    cfg = SLConfig.fromfile(args.config_file)
    for k, v in vars(args).items(): setattr(cfg, k, v)

    if hasattr(cfg, 'use_checkpoint'):
        print("Disabling gradient checkpointing for ONNX export.")
        cfg.use_checkpoint = False
    
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f"Using device: '{device}' for the export process.")

    print("Building model architecture...")
    model, _, _ = MODULE_BUILD_FUNCS.get(cfg.modelname)(cfg)
    model.to(device)

    print(f"Loading checkpoint from: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    model.load_state_dict(clean_state_dict(checkpoint['model']))
    model.eval()

    onnx_model = DINOOnnxWrapper(model)
    onnx_model.eval()

    dummy_images = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    dummy_mask = torch.zeros(args.batch_size, args.height, args.width, device=device, dtype=torch.bool)
    dummy_input = (dummy_images, dummy_mask)

    checkpoint_path = Path(args.resume)
    model_dir_name = checkpoint_path.parent.name
    model_file_stem = checkpoint_path.stem
    
    output_dir = Path(args.output_root) / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_file_stem}_op{args.opset_version}_pytorch_ops.onnx"

    print(f"\nExporting model using PyTorch ops for compatibility to: {output_path}")
    
    try:
        torch.onnx.export(
            onnx_model,
            dummy_input,
            str(output_path),
            input_names=['images', 'mask'],
            output_names=['pred_logits', 'pred_boxes'],
            dynamic_axes={
                'images': {0: 'batch_size', 2: 'height', 3: 'width'},
                'mask': {0: 'batch_size', 1: 'height', 2: 'width'},
                'pred_logits': {0: 'batch_size', 1: 'num_queries'},
                'pred_boxes': {0: 'batch_size', 1: 'num_queries'}
            },
            opset_version=args.opset_version,
            verbose=False
        )
        print(f"\n✅ SUCCESS: Model exported to {output_path}")
        print("\n--- NOTE ---")
        print("This ONNX model uses standard PyTorch operations (like grid_sample) instead of the custom CUDA kernel.")
        print("It is now compatible with standard ONNX runtimes and ready for inference.")

    except Exception as e:
        print(f"\n❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO ONNX Export Script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)