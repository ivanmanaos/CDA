import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import DogHeartTestDataset
from model import get_model
from utils import get_transform

def generate_pseudo_labels(model, unlabeled_loader, device, mc_passes=20, threshold=0.005):
    model.train()
    pseudo_labels, uncertainties = [], []

    for images, _ in tqdm(unlabeled_loader, desc="Generating Pseudo-labels"):
        images = images.to(device)
        preds = []
        for _ in range(mc_passes):
            preds.append(model(images).detach().cpu())
        preds = torch.stack(preds)
        mean_preds = preds.mean(dim=0)
        std_preds = preds.std(dim=0)
        pseudo_labels.append(mean_preds)
        uncertainties.append(std_preds)

    pseudo_labels = torch.cat(pseudo_labels)
    uncertainties = torch.cat(uncertainties)
    high_conf_idx = (uncertainties.max(dim=1)[0] < threshold).nonzero(as_tuple=True)[0]
    return pseudo_labels[high_conf_idx], high_conf_idx


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels with MC Dropout")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mc_passes", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    transform = get_transform(args.image_size)
    unlabeled_ds = DogHeartTestDataset(args.data_dir, transforms=transform)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    model = get_model(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    pseudo, idx = generate_pseudo_labels(model, unlabeled_loader, device, mc_passes=args.mc_passes, threshold=args.threshold)
    print(f"High confidence count: {len(idx)} / {len(unlabeled_ds)}")
