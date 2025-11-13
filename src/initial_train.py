import os
import argparse
import random
import math
import csv
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from dataset import DogHeartDataset
from model import get_model, calc_vhs
from utils import get_transform
import matplotlib.pyplot as plt


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_initial_model(
    train_loader,
    valid_loader,
    device,
    num_epochs=100,
    lr=3e-4,
    out_dir="models",
    index_map=None,
    optimizer_name="adamw",
    weight_decay=0.05,
    sched_name="cosine",
    warmup_epochs=1,
    patience=10,
    grad_clip=1.0,
    amp=True,
    pretrained=True,
    log_dir="outputs"
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = get_model(device, pretrained=pretrained)
    model.to(device)
    criterion = torch.nn.L1Loss()

    if optimizer_name.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Scheduler after warmup
    if sched_name.lower() == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=1e-6)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == 'cuda'))  # GradScaler ok

    # pred_record for soft targets (same logic as before)
    if index_map is None:
        pred_record = torch.zeros([len(train_loader.dataset), 10, 12], dtype=torch.float32)
    else:
        pred_record = torch.zeros((len(index_map), 10, 12), dtype=torch.float32)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": [], "val_acc": []}
    best_val_mae = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    print(f"Device: {device} | CUDA available: {torch.cuda.is_available()} | AMP: {amp}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (ind, images, points, vhs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = images.to(device, non_blocking=True)
            points = points.to(device, non_blocking=True)
            vhs = vhs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(amp and device.type == 'cuda')):
                outputs = model(images)
                loss1 = criterion(outputs.squeeze(), points.squeeze())
                loss2 = criterion(calc_vhs(outputs).squeeze(), vhs.squeeze())
                loss = 10 * loss1 + 0.1 * loss2
                if epoch > 10:
                    if index_map is not None:
                        local_rows = [index_map[int(i)] for i in (ind.tolist() if hasattr(ind, 'tolist') else list(ind))]
                        soft_points = pred_record[local_rows].mean(axis=1).to(device)
                    else:
                        soft_points = pred_record[ind].mean(axis=1).to(device)
                    loss3 = criterion(outputs.squeeze(), soft_points)
                    loss += loss3

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            out_fp32 = outputs.detach().to(torch.float32).cpu()
            if index_map is not None:
                local_rows = [index_map[int(i)] for i in (ind.tolist() if hasattr(ind, 'tolist') else list(ind))]
                pred_record[local_rows, epoch % 10, :] = out_fp32
            else:
                pred_record[ind, epoch % 10] = out_fp32
            if batch_idx == 0:
                print("dtypes -> outputs:", outputs.dtype, "| pred_record:", pred_record.dtype)

        # Warmup logic: linearly scale lr for first warmup_epochs
        if epoch < warmup_epochs:
            warmup_factor = float(epoch + 1) / float(max(1, warmup_epochs))
            for pg in optimizer.param_groups:
                pg['lr'] = lr * warmup_factor
        else:
            if scheduler is not None:
                scheduler.step()

        # Validation (compute losses + VHS metrics)
        model.eval()
        val_running_loss = 0.0
        vhs_preds = []
        vhs_targets = []
        correct_cls = 0
        total_cls = 0
        with torch.no_grad():
            for ind, images, points, vhs in valid_loader:
                images = images.to(device, non_blocking=True)
                points = points.to(device, non_blocking=True)
                vhs = vhs.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(amp and device.type == 'cuda')):
                    out = model(images)
                    logits_for_metrics = out.detach().to(torch.float32)
                    l1 = criterion(out.squeeze(), points.squeeze())
                    l2 = criterion(calc_vhs(out).squeeze(), vhs.squeeze())
                    batch_loss = 10 * l1 + 0.1 * l2
                val_running_loss += batch_loss.item()
                batch_vhs_pred = calc_vhs(logits_for_metrics).detach().to(torch.float32).cpu()
                vhs_preds.extend(batch_vhs_pred.view(-1).tolist())
                vhs_targets.extend(vhs.detach().cpu().view(-1).tolist())
                # Optional accuracy by thresholds
                for gp, gt in zip(batch_vhs_pred.view(-1).tolist(), vhs.detach().cpu().view(-1).tolist()):
                    c_pred = 0 if gp < 8.2 else (1 if gp < 10 else 2)
                    c_gt = 0 if gt < 8.2 else (1 if gt < 10 else 2)
                    if c_pred == c_gt:
                        correct_cls += 1
                    total_cls += 1

        val_loss = val_running_loss / max(1, len(valid_loader.dataset))
        # Compute MAE / RMSE
        if vhs_preds:
            import numpy as np
            v_pred_arr = np.array(vhs_preds)
            v_gt_arr = np.array(vhs_targets)
            val_mae = float(np.mean(np.abs(v_pred_arr - v_gt_arr)))
            val_rmse = float(math.sqrt(np.mean((v_pred_arr - v_gt_arr) ** 2)))
            val_acc = float(correct_cls / total_cls) if total_cls else 0.0
        else:
            val_mae = 0.0
            val_rmse = 0.0
            val_acc = 0.0

        print(f"Epoch [{epoch+1}/{num_epochs}] TrainLoss: {running_loss/len(train_loader.dataset):.4f} ValLoss: {val_loss:.4f} ValMAE: {val_mae:.4f} ValRMSE: {val_rmse:.4f} ValAcc: {val_acc:.4f}")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(running_loss / len(train_loader.dataset))
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)
        history["val_acc"].append(val_acc)

        improved = val_mae < best_val_mae - 1e-6  # tiny epsilon
        if improved:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_name_detailed = os.path.join(out_dir, f"cda_b7_best_mae{val_mae:.4f}_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), best_name_detailed)
            torch.save(model.state_dict(), os.path.join(out_dir, "cda_b7_best_mae.pth"))
            print(f"Saved NEW BEST: {best_name_detailed}")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1} (best epoch {best_epoch}, best MAE {best_val_mae:.4f})")
            break

    # Save last model
    torch.save(model.state_dict(), os.path.join(out_dir, "cda_b7_last.pth"))

    # Logging CSV
    log_csv = os.path.join(log_dir, "training_log.csv")
    write_header = not os.path.exists(log_csv)
    with open(log_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mae", "val_rmse", "val_acc"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i], history["train_loss"][i], history["val_loss"][i], history["val_mae"][i], history["val_rmse"][i], history["val_acc"][i]
            ])
    print(f"Wrote log CSV -> {log_csv}")

    # Plot
    try:
        plt.figure(figsize=(8,5))
        plt.plot(history["epoch"], history["train_loss"], label="TrainLoss")
        plt.plot(history["epoch"], history["val_loss"], label="ValLoss")
        plt.plot(history["epoch"], history["val_mae"], label="ValMAE")
        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Training Curve")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(log_dir, "training_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot -> {plot_path}")
    except Exception as e:
        print(f"Plot error: {e}")


def parse_args():  # retained for standalone usage
    parser = argparse.ArgumentParser(description="Initial training for CDA model")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--pretrained", type=lambda s: s.lower() in {'1','true','yes'}, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    transform = get_transform(args.image_size)
    full_dataset = DogHeartDataset(args.data_dir, transforms=transform)
    val_len = max(1, int(len(full_dataset) * args.val_split))
    train_len = len(full_dataset) - val_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    train_initial_model(train_loader, valid_loader, device, num_epochs=args.epochs, lr=args.lr, out_dir=args.out_dir, pretrained=args.pretrained)
