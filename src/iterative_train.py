import os
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import HighConfidenceDataset, DogHeartDataset, DogHeartTestDataset
from pseudo_labeling import generate_pseudo_labels
from model import get_model, calc_vhs
from evaluate import evaluate_model
from utils import get_transform

def train_with_pseudo_labels(train_loader, unlabeled_loader, valid_loader, checkpoint, device, num_epochs=100, lr=1e-5):
    # FIX minimal: ensure model and scheduler/optimizer imports exist and checkpoint can be a path
    model = get_model(device)
    # Accept either a state_dict or a path to a checkpoint file
    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    criterion = torch.nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    pred_record = torch.zeros([len(train_loader.dataset), 10, 12])
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (ind, images, points, vhs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, points, vhs = images.to(device), points.to(device), vhs.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss1 = criterion(outputs, points)
            loss2 = criterion(calc_vhs(outputs), vhs)
            loss = 10 * loss1 + 0.1 * loss2
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred_record[ind, epoch % 10] = outputs.detach().cpu()
            
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate_model(model, valid_loader, device, criterion)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Generate pseudo-labels
        high_conf_pseudo_labels, high_conf_idx = generate_pseudo_labels(model, unlabeled_loader, device)
        if len(high_conf_idx) > 0:
            high_conf_images = [unlabeled_loader.dataset[i][0] for i in high_conf_idx]
            print(f"High confidence pseudo-labels count: {len(high_conf_idx)} (out of {len(unlabeled_loader.dataset)})")
            high_conf_dataset = HighConfidenceDataset(high_conf_images, high_conf_pseudo_labels)
            combined_dataset = ConcatDataset([train_loader.dataset, high_conf_dataset])
            train_loader = DataLoader(combined_dataset, batch_size=train_loader.batch_size, shuffle=True)
        else:
            print("No high confidence pseudo-labels found.")


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative training with pseudo-labels")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory containing Images/ and Labels/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to initial model weights")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    transform = get_transform(args.image_size)
    # Use same data_dir for labeled and unlabeled for simplicity; replace with separate dir if available
    train_ds = DogHeartDataset(args.data_dir, transforms=transform)
    val_len = max(1, int(0.1 * len(train_ds)))
    train_len = len(train_ds) - val_len
    from torch.utils.data import random_split
    train_part, val_part = random_split(train_ds, [train_len, val_len])
    train_loader = DataLoader(train_part, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(val_part, batch_size=args.batch_size, shuffle=False)
    unlabeled_ds = DogHeartTestDataset(args.data_dir, transforms=transform)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    train_with_pseudo_labels(train_loader, unlabeled_loader, valid_loader, args.checkpoint, device, num_epochs=args.epochs, lr=args.lr)
