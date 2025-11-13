import os
import sys
import argparse
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader, random_split, Subset

# Ensure src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

from dataset import DogHeartDataset  # noqa: E402
from utils import get_transform         # noqa: E402
from initial_train import train_initial_model  # noqa: E402


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description='Train initial CDA model (CLI runner)')
    p.add_argument('--data_dir', type=str, default='data', help='Root containing Images/ and Labels/')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--image_size', type=int, default=384)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out_dir', type=str, default=str(REPO_ROOT / 'models'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val_split', type=float, default=0.1)
    p.add_argument('--splits_json', type=str, default=None, help='Optional JSON with train/val/test filenames')
    p.add_argument('--pretrained', type=lambda s: s.lower() in {'1','true','yes'}, default=True, help='Use pretrained weights (set false for offline)')
    # New training robustness flags
    p.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs without MAE improvement)')
    p.add_argument('--optimizer', type=str, default='adamw', choices=['adamw'], help='Optimizer type')
    p.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for optimizer')
    p.add_argument('--sched', type=str, default='cosine', choices=['cosine','none'], help='LR scheduler')
    p.add_argument('--warmup_epochs', type=int, default=1, help='Linear warmup epochs before scheduler')
    p.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clip norm (<=0 disables)')
    p.add_argument('--amp', type=lambda s: s.lower() in {'1','true','yes'}, default=True, help='Use Automatic Mixed Precision on CUDA')
    p.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    transform = get_transform(args.image_size)
    full_ds = DogHeartDataset(str(data_dir), transforms=transform)

    if args.splits_json:
        import json
        split = json.loads(Path(args.splits_json).read_text(encoding='utf-8'))
        # Map image filename to index in DogHeartDataset.imgs
        img2idx = {Path(name).name: i for i, name in enumerate(full_ds.imgs)}
        train_names = [Path(n).name for n in split.get('train', [])]
        val_names = [Path(n).name for n in split.get('val', [])]
        train_idx = [img2idx[n] for n in train_names if n in img2idx]
        val_idx = [img2idx[n] for n in val_names if n in img2idx]
        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx)
    else:
        val_len = max(1, int(len(full_ds) * args.val_split))
        train_len = len(full_ds) - val_len
        train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    # FIX index_map: build mapping from original dataset indices -> local [0..len(train_ds)-1]
    if isinstance(train_ds, torch.utils.data.Subset):
        train_indices = list(map(int, train_ds.indices))
    else:
        train_indices = list(range(len(train_ds)))
    index_map = {orig_idx: i for i, orig_idx in enumerate(train_indices)}

    pin = (args.device == 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    valid_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    train_initial_model(
        train_loader,
        valid_loader,
        device,
        num_epochs=args.epochs,
        lr=args.lr,
        out_dir=args.out_dir,
        index_map=index_map,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        sched_name=args.sched,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        grad_clip=args.grad_clip if args.grad_clip and args.grad_clip > 0 else None,
        amp=args.amp and (args.device == 'cuda'),
        pretrained=args.pretrained,
        log_dir=str(REPO_ROOT / 'outputs')
    )

    # Report checkpoint presence
    best_alias = Path(args.out_dir) / 'cda_b7_best_mae.pth'
    last_alias = Path(args.out_dir) / 'cda_b7_last.pth'
    print(f"Best (alias): {best_alias.exists()} -> {best_alias}")
    print(f"Last: {last_alias.exists()} -> {last_alias}")


if __name__ == '__main__':
    main()
