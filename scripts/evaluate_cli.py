import os
import sys
import argparse
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd

# Ensure src/ on path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

from dataset import DogHeartTestDataset, DogHeartDataset  # noqa: E402
from utils import get_transform            # noqa: E402
from model import get_model, calc_vhs      # noqa: E402
from evaluate import inference_and_save    # noqa: E402
import torch.nn.functional as F
import math
from collections import Counter
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate CDA model, produce metrics, and save predictions')
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--image_size', type=int, default=384)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir', type=str, default=str(REPO_ROOT / 'outputs'))
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--splits_json', type=str, default=None, help='Optional JSON with train/val/test filenames')
    p.add_argument('--use_split', type=str, default=None, choices=['train','val','test'], help='Which split to evaluate (requires --splits_json)')
    p.add_argument('--out_csv', type=str, default=str(REPO_ROOT / 'outputs' / 'test_preds.csv'))
    p.add_argument('--metrics_out', type=str, default=str(REPO_ROOT / 'outputs' / 'test_metrics.json'))
    p.add_argument('--confusion_out', type=str, default=str(REPO_ROOT / 'outputs' / 'test_confusion.png'))
    p.add_argument('--pretrained', type=lambda s: s.lower() in {'1','true','yes'}, default=True, help='Use pretrained weights (set false for offline)')
    p.add_argument('--tta', type=int, default=1, help='Number of Test-Time Augmentations (>=1)')
    p.add_argument('--mc_passes', type=int, default=0, help='Number of MC-Dropout passes')
    return p.parse_args()


def classify_vhs(v):
    if v < 8.2: return 0
    if v < 10: return 1
    return 2


def compute_metrics(rows):
    # rows: list of dict with VHS_gt, VHS_pred, class_gt, class_pred
    import numpy as np
    vhs_gt = np.array([r['VHS_gt'] for r in rows])
    vhs_pred = np.array([r['VHS_pred'] for r in rows])
    mae = float(np.mean(np.abs(vhs_pred - vhs_gt)))
    rmse = float(math.sqrt(np.mean((vhs_pred - vhs_gt)**2)))
    cls_gt = np.array([r['class_gt'] for r in rows])
    cls_pred = np.array([r['class_pred'] for r in rows])
    accuracy = float((cls_gt == cls_pred).mean())
    # Confusion matrix 3x3
    conf = [[0]*3 for _ in range(3)]
    for g, p in zip(cls_gt, cls_pred):
        conf[g][p] += 1
    # Macro-F1
    f1s=[]
    for i in range(3):
        tp = conf[i][i]
        fp = sum(conf[j][i] for j in range(3) if j!=i)
        fn = sum(conf[i][j] for j in range(3) if j!=i)
        prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
        rec = tp / (tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        f1s.append(f1)
    macro_f1 = float(sum(f1s)/3.0)
    support = Counter(cls_gt.tolist())
    return mae, rmse, accuracy, macro_f1, conf, support


def save_confusion(conf, path):
    import numpy as np
    arr = np.array(conf)
    plt.figure(figsize=(4,4))
    plt.imshow(arr, cmap='Blues')
    plt.title('Confusion Matrix (0=Normal,1=Borderline,2=Cardiomegaly)')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, str(arr[i,j]), ha='center', va='center', color='black')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    transform = get_transform(args.image_size)

    # Determine labeled subset (DogHeartDataset) for metrics when split provided
    full_labeled = DogHeartDataset(args.data_dir, transforms=transform)
    subset_ds = None
    if args.splits_json and args.use_split:
        split = json.loads(Path(args.splits_json).read_text(encoding='utf-8'))
        target_list = split.get(args.use_split, [])
        # Build index mapping from filename
        img2idx = {Path(name).name: i for i, name in enumerate(full_labeled.imgs)}
        idx_keep = [img2idx[Path(n).name] for n in target_list if Path(n).name in img2idx]
        subset_ds = Subset(full_labeled, idx_keep)
    else:
        subset_ds = full_labeled  # fallback: evaluate all

    loader = DataLoader(subset_ds, batch_size=args.batch_size, shuffle=False)

    model = get_model(device, pretrained=args.pretrained)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    rows = []
    # Multi-pass inference supporting TTA and MC-Dropout
    with torch.no_grad():
        for batch in loader:
            inds, images, points, vhs_gt = batch
            images = images.to(device)
            pass_vhs = []
            passes = max(1, args.mc_passes, args.tta)
            for pi in range(passes):
                # Simple TTA: alternate original / horizontal flip (flip back predictions logically same for points -> just feed flipped image)
                aug_images = images
                if args.tta > 1 and (pi % 2 == 1):
                    aug_images = torch.flip(images, dims=[3])  # flip width
                if args.mc_passes > 0:
                    model.train()  # enable dropout layers
                out = model(aug_images)
                out_fp32 = out.detach().to(torch.float32)
                vhs_pass = calc_vhs(out_fp32).detach()  # [B,1]
                pass_vhs.append(vhs_pass)
            model.eval()
            stacked_vhs = torch.stack(pass_vhs)  # [passes,B,1]
            mean_vhs = stacked_vhs.mean(0).view(-1)
            std_vhs = stacked_vhs.std(0).view(-1) if passes > 1 else torch.zeros_like(mean_vhs)
            for i in range(images.size(0)):
                orig_idx = int(inds[i])
                filename = full_labeled.imgs[orig_idx]
                v_gt = float(vhs_gt[i].item())
                v_pred = float(mean_vhs[i].item())
                c_gt = classify_vhs(v_gt)
                c_pred = classify_vhs(v_pred)
                rows.append({
                    'ImageName': filename,
                    'VHS_gt': v_gt,
                    'VHS_pred': v_pred,
                    'class_gt': c_gt,
                    'class_pred': c_pred,
                    'uncertainty_pred': float(std_vhs[i].item()) if passes > 1 else 0.0
                })

    mae, rmse, accuracy, macro_f1, conf, support = compute_metrics(rows) if rows else (0,0,0,0,[[0]*3]*3,{})

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved predictions CSV -> {args.out_csv}")

    # Save metrics JSON
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'support': dict(support)
    }
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON -> {args.metrics_out}")

    save_confusion(conf, args.confusion_out)
    print(f"Saved confusion matrix -> {args.confusion_out}")


if __name__ == '__main__':
    main()
