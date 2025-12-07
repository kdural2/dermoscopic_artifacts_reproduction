# src/train_melanoma.py

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from torchmetrics.classification import BinaryAUROC
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
import time

from .datasets import ISIC2018Dataset, get_default_transform
from .models import create_model


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    running_loss = 0.0

    for imgs, labels, _ in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        with autocast(dtype=torch.float16):
            logits = model(imgs)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)



def validate(model, loader, device):
    model.eval()
    auroc_metric = BinaryAUROC().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="val", leave=False):
            imgs = imgs.to(device)
            labels_float = labels.float().unsqueeze(1).to(device)

            with autocast(dtype=torch.float16):
                logits = model(imgs)
                loss = loss_fn(logits, labels_float)

            total_loss += loss.item() * imgs.size(0)
            auroc_metric.update(logits, labels_float)

    avg_loss = total_loss / len(loader.dataset)
    auroc = auroc_metric.compute().item()
    return avg_loss, auroc



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/ISIC2018")
    parser.add_argument("--output_dir", type=str, default="checkpoints/isic")
    parser.add_argument("--mode", type=str, default="whole",
                        help="whole, lesion, background, bbox, bbox70, bbox90, "
                             "high_whole, low_whole, high_lesion, low_lesion, high_background, low_background")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_folds", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_root / "labels.csv")
    indices = np.arange(len(df))

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    fold = 0
    results = []

    for train_idx, val_idx in kf.split(indices):
        print(f"\n=== Fold {fold}/{args.num_folds} | mode={args.mode} ===")

        train_dataset = ISIC2018Dataset(
            root=str(data_root),
            mode=args.mode,
            fold_indices=train_idx,
            transform=get_default_transform()
        )
        val_dataset = ISIC2018Dataset(
            root=str(data_root),
            mode=args.mode,
            fold_indices=val_idx,
            transform=get_default_transform()
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )

        model = create_model(pretrained=True, num_classes=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_auroc = 0.0

        start_time_all = time.time()
        for epoch in range(args.epochs):
            epoch_start = time.time()

            print(f"\nFold {fold} Epoch {epoch+1}/{args.epochs}")

            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_auroc = validate(model, val_loader, device)

            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            total_elapsed = epoch_end - start_time_all

            print(
                f"  train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_auroc={val_auroc:.4f} "
                f"| epoch_time={epoch_time:.2f}s "
                f"| total_elapsed={total_elapsed/60:.2f} min"
            )

            if val_auroc > best_auroc:
                best_auroc = val_auroc
                ckpt_path = output_dir / f"resnet50_fold{fold}.pt"

                torch.save({
                    "model_state": model.state_dict(),
                    "val_auroc": best_auroc,
                    "fold": fold,
                    "mode": args.mode
                }, ckpt_path)

        results.append(best_auroc)
        fold += 1

    print(f"\nFinal Mean AUROC for mode={args.mode}: {np.mean(results):.4f} ± {np.std(results):.4f}")


if __name__ == "__main__":
    main()
