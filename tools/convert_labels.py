"""
Optional converter to adapt labels to the expected structure with keys {six_points, VHS}.
Usage example:
  python tools/convert_labels.py --in_dir D:\descargas\Synthetic\Labels --out_dir D:\descargas\Synthetic\Labels
This is a placeholder since source label format is [dato no provisto]. Implement when source format is known.
"""
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description='Convert labels to expected .mat keys (six_points, VHS) [placeholder]')
    p.add_argument('--in_dir', type=str, required=True)
    p.add_argument('--out_dir', type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    print('[dato no provisto] Implement conversion once source label format is specified.')
    print(f'in_dir={args.in_dir} out_dir={args.out_dir}')

if __name__ == '__main__':
    main()
