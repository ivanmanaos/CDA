import json
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Check train/val/test splits overlap (leakage)")
    p.add_argument("--splits_json", type=str, required=True, help="Path to splits JSON produced by make_splits.py")
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.splits_json)
    data = json.loads(path.read_text(encoding="utf-8"))
    train = set(Path(x).name for x in data.get("train", []))
    val = set(Path(x).name for x in data.get("val", []))
    test = set(Path(x).name for x in data.get("test", []))

    inter_tv = train & val
    inter_tt = train & test
    inter_vt = val & test

    leakage = {
        "train_val_overlap": sorted(inter_tv),
        "train_test_overlap": sorted(inter_tt),
        "val_test_overlap": sorted(inter_vt)
    }

    print("Leakage report:")
    for k, v in leakage.items():
        print(f"  {k}: {len(v)}")
        if v:
            print("    Samples:")
            for name in v[:20]:  # cap output
                print(f"      {name}")

    if any(leakage.values()):
        print("WARNING: Leakage detected between splits.")
    else:
        print("OK: No leakage detected.")


if __name__ == "__main__":
    main()
