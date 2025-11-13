import os
import sys
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

from utils import get_transform  # noqa: E402
from model import get_model, calc_vhs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='Single image inference for CDA model')
    p.add_argument('--weights', type=str, required=True, help='Path to model weights')
    p.add_argument('--image', type=str, required=True, help='Path to a single image (.png/.jpg)')
    p.add_argument('--image_size', type=int, default=384)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_json', type=str, default=None, help='Output json path; defaults to outputs/infer_<stem>.json')
    p.add_argument('--print_json', action='store_true', help='Print resulting JSON to stdout')
    p.add_argument('--pretty', action='store_true', help='Pretty print JSON (indent=2)')
    p.add_argument('--thresholds', type=str, default='8.2,10', help='Comma separated thresholds e.g. "8.2,10"')
    p.add_argument('--save_points_px', action='store_true', help='Also include points in pixels (rounded)')
    p.add_argument('--pretrained', type=lambda s: s.lower() in {'1','true','yes'}, default=True, help='Use pretrained weights (set false for offline)')
    return p.parse_args()


def safe_torch_load(path: Path, map_location):
    # PyTorch newer versions recommend weights_only=True; try first, fallback if unsupported
    try:
        return torch.load(path, map_location=map_location, weights_only=True)  # may warn or fail on older versions
    except TypeError:
        # Fallback without weights_only for older torch
        return torch.load(path, map_location=map_location)


def classify_vhs(v: float, t_low: float = 8.2, t_high: float = 10.0) -> int:
    if v < t_low:
        return 0
    if v < t_high:
        return 1
    return 2


def main():
    args = parse_args()
    try:
        img_path = Path(args.image)
        weights_path = Path(args.weights)
        if not img_path.is_file():
            print(f"Image not found: {img_path}")
            raise SystemExit(2)
        if not weights_path.is_file():
            print(f"Weights not found: {weights_path}")
            raise SystemExit(2)

        # thresholds
        try:
            t_parts = [p.strip() for p in str(args.thresholds).split(',') if p.strip()]
            t_low, t_high = float(t_parts[0]), float(t_parts[1])
        except Exception:
            t_low, t_high = 8.2, 10.0

        device = torch.device(args.device)
        model = get_model(device, pretrained=args.pretrained, image_size=args.image_size)
        state = safe_torch_load(weights_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()

        transform = get_transform(args.image_size)
        img = Image.open(img_path).convert('RGB')
        img_t = cast(torch.Tensor, transform(img)).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_t)
            out_fp32 = out.detach().to(torch.float32)
            vhs = float(calc_vhs(out_fp32).cpu().item())
            pts = out_fp32.squeeze(0).cpu().view(-1, 2).numpy()  # normalized [0,1]

        cls = classify_vhs(vhs, t_low=t_low, t_high=t_high)
        label_map = {0: "Normal", 1: "Intermedio", 2: "Alto"}
        result = {
            "image": str(img_path.resolve()),
            "VHS_pred": vhs,
            "class_pred": int(cls),
            "label": label_map.get(int(cls), str(cls)),
            "healthy": bool(cls == 0),
            "thresholds": [t_low, t_high],
            "points_pred": [[float(x), float(y)] for x, y in pts.tolist()],
            "model": {
                "weights": str(weights_path.resolve()),
                "image_size": int(args.image_size),
                "device": device.type,
            },
        }

        if args.save_points_px:
            px = (pts * float(args.image_size)).round(2)
            result["points_pred_px"] = [[float(x), float(y)] for x, y in px.tolist()]

        # output path
        if args.out_json:
            out_path = Path(args.out_json)
        else:
            out_path = REPO_ROOT / 'outputs' / f"infer_{img_path.stem}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        with open(out_path, 'w', encoding='utf-8') as f:
            if args.pretty:
                json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                json.dump(result, f, ensure_ascii=False)

        # Print either the JSON (if requested) or a simple saved message
        if args.print_json:
            txt = json.dumps(result, indent=2 if args.pretty else None, ensure_ascii=False)
            print(txt)
        else:
            print(f"Saved inference JSON -> {out_path}")

    except SystemExit as e:
        raise e
    except Exception as e:
        print(f"Inference failed: {e}")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
