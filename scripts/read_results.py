import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Read and summarize TEST results')
    p.add_argument('--metrics', type=str, required=True, help='Path to metrics JSON (e.g., outputs/test_metrics.json)')
    p.add_argument('--csv', type=str, required=True, help='Path to predictions CSV (e.g., outputs/test_preds.csv)')
    p.add_argument('--top_k', type=int, default=10, help='Number of worst cases to display')
    p.add_argument('--by', type=str, default='abs_err', choices=['abs_err','vhs_pred'], help='Sort key for TOP K list')
    return p.parse_args()


def classify_vhs(v):
    if v < 8.2:
        return 0
    if v < 10:
        return 1
    return 2


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure abs_err
    if 'abs_err' not in df.columns:
        if all(c in df.columns for c in ['VHS_pred','VHS_gt']):
            df['abs_err'] = (df['VHS_pred'] - df['VHS_gt']).abs()
        else:
            df['abs_err'] = np.nan
    # Ensure classes
    if 'class_gt' not in df.columns and 'VHS_gt' in df.columns:
        df['class_gt'] = df['VHS_gt'].apply(classify_vhs)
    if 'class_pred' not in df.columns and 'VHS_pred' in df.columns:
        df['class_pred'] = df['VHS_pred'].apply(classify_vhs)
    return df


def main():
    args = parse_args()

    metrics_path = Path(args.metrics)
    csv_path = Path(args.csv)

    print('=== Metrics Summary ===')
    if metrics_path.is_file():
        m = json.loads(metrics_path.read_text(encoding='utf-8'))
        print(f"MAE: {m.get('MAE')}")
        print(f"RMSE: {m.get('RMSE')}")
        print(f"Accuracy: {m.get('accuracy')}")
        print(f"Macro-F1: {m.get('macro_f1')}")
        sup = m.get('support', {})
        if isinstance(sup, dict):
            print(f"Support by class: {sup}")
    else:
        print(f"Metrics file not found: {metrics_path}")

    print('\n=== Predictions CSV ===')
    if not csv_path.is_file():
        print(f"Predictions CSV not found: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    df = ensure_columns(df)

    # Head(10)
    print('\nHead(10):')
    print(df.head(10))

    # Counts
    if 'class_pred' in df.columns:
        print('\nCount by class_pred:')
        print(df['class_pred'].value_counts(dropna=False).sort_index())
    if 'class_gt' in df.columns:
        print('\nCount by class_gt:')
        print(df['class_gt'].value_counts(dropna=False).sort_index())

    # Percentiles of abs_err
    if 'abs_err' in df.columns:
        percs = [50, 75, 90, 95]
        vals = np.percentile(df['abs_err'].dropna().to_numpy(), percs) if df['abs_err'].notna().any() else [np.nan]*len(percs)
        print('\nabs_err percentiles:')
        for p, v in zip(percs, vals):
            print(f"  P{p}: {v:.4f}")

    # Correlation uncertainty vs abs_err
    if 'uncertainty_pred' in df.columns and 'abs_err' in df.columns:
        x = df['uncertainty_pred'].to_numpy()
        y = df['abs_err'].to_numpy()
        if np.isfinite(x).any() and np.isfinite(y).any():
            # Pearson correlation
            x0 = x - np.nanmean(x)
            y0 = y - np.nanmean(y)
            num = np.nanmean(x0*y0)
            den = np.sqrt(np.nanmean(x0**2) * np.nanmean(y0**2))
            corr = float(num/den) if den and np.isfinite(den) else float('nan')
            print(f"\nuncertainty_pred vs abs_err (Pearson r): {corr:.4f}")

    # Top-K worst
    sort_key = args.by if args.by in df.columns else 'abs_err'
    df_sorted = df.sort_values(by=sort_key, ascending=False)
    k = min(args.top_k, len(df_sorted))
    print(f"\nTop {k} worst by '{sort_key}':")
    cols_show = [c for c in ['ImageName','VHS_gt','VHS_pred','abs_err','class_gt','class_pred','uncertainty_pred'] if c in df_sorted.columns]
    print(df_sorted[cols_show].head(k))

    # Confusion image path
    conf_png = csv_path.parent / 'test_confusion.png'
    if conf_png.is_file():
        print(f"\nConfusion matrix image: {conf_png}")
    else:
        print(f"\nConfusion matrix image not found at: {conf_png}")


if __name__ == '__main__':
    main()
