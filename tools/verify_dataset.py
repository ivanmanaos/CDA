import argparse
from pathlib import Path
import sys
from scipy.io import loadmat

IMG_EXTS = {'.png', '.jpg', '.jpeg'}


def parse_args():
    p = argparse.ArgumentParser(description='Verify dataset structure for CDA')
    p.add_argument('--data_dir', type=str, default='data')
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_dir)
    images_dir = root / 'Images'
    labels_dir = root / 'Labels'

    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"Missing required subfolders. Expected: {images_dir} and {labels_dir}")
        sys.exit(2)

    images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    labels = [p for p in labels_dir.iterdir() if p.suffix.lower() == '.mat']

    print(f"Total images: {len(images)}")
    print(f"Total labels: {len(labels)}")

    # Map by basename without extension
    img_map = {p.stem: p for p in images}
    lbl_map = {p.stem: p for p in labels}

    # Show 5 examples
    pairs = []
    for k in list(img_map.keys())[:5]:
        if k in lbl_map:
            pairs.append((img_map[k].name, lbl_map[k].name))
    print("Examples (Image ↔ Label):")
    for a, b in pairs:
        print(f"  {a} ↔ {b}")

    # Mismatch report
    missing_lbl = [k for k in img_map.keys() if k not in lbl_map]
    missing_img = [k for k in lbl_map.keys() if k not in img_map]
    if missing_lbl:
        print(f"Images without labels: {len(missing_lbl)} e.g. {missing_lbl[:5]}")
    if missing_img:
        print(f"Labels without images: {len(missing_img)} e.g. {missing_img[:5]}")

    # Load and validate 3 random .mat files
    sample_lbls = list(lbl_map.values())[:3]
    ok = True
    for p in sample_lbls:
        try:
            m = loadmat(p)
            if 'six_points' not in m or 'VHS' not in m:
                print(f"Missing keys in {p.name}: expected 'six_points' and 'VHS'")
                ok = False
                continue
            sp = m['six_points']
            if not hasattr(sp, 'shape') or tuple(sp.shape) != (6, 2):
                print(f"Bad shape for six_points in {p.name}: got {getattr(sp, 'shape', None)}")
                ok = False
        except Exception as e:
            print(f"Error reading {p.name}: {e}")
            ok = False

    if not ok or missing_lbl or missing_img:
        sys.exit(1)
    print("Dataset verification OK")
    sys.exit(0)


if __name__ == '__main__':
    main()
