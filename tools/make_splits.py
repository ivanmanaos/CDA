import argparse
import json
import random
import re
from pathlib import Path

IMG_EXTS = {'.png', '.jpg', '.jpeg'}
GROUP_RE = re.compile(r'^(.*)_\d+\.(png|jpg|jpeg)$', re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser(description='Create persistent train/val/test splits JSON')
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--val_ratio', type=float, default=0.15)
    p.add_argument('--test_ratio', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--group_by_id', type=lambda s: s.lower() in {'1','true','yes'}, default=False)
    p.add_argument('--out_json', type=str, default=str(Path('outputs') / 'splits_seed42.json'))
    return p.parse_args()


def group_key(name: str, enable: bool) -> str:
    if not enable:
        return name
    m = GROUP_RE.match(name)
    return m.group(1) if m else name


def main():
    args = parse_args()
    root = Path(args.data_dir)
    images_dir = root / 'Images'
    images = [p.name for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    # group by id if requested
    buckets = {}
    for nm in images:
        key = group_key(nm, args.group_by_id)
        buckets.setdefault(key, []).append(nm)

    keys = list(buckets.keys())
    random.seed(args.seed)
    random.shuffle(keys)

    n_keys = len(keys)
    n_test = int(n_keys * args.test_ratio)
    n_val = int((n_keys - n_test) * args.val_ratio)

    test_keys = keys[:n_test]
    val_keys = keys[n_test:n_test+n_val]
    train_keys = keys[n_test+n_val:]

    train = [nm for k in train_keys for nm in buckets[k]]
    val = [nm for k in val_keys for nm in buckets[k]]
    test = [nm for k in test_keys for nm in buckets[k]]

    out = {"train": train, "val": val, "test": test, "seed": args.seed, "group_by_id": bool(args.group_by_id)}

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"Saved splits -> {out_path} (train={len(train)}, val={len(val)}, test={len(test)})")


if __name__ == '__main__':
    main()
