# src/eval_transfer.py

import argparse
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC
import pandas as pd
import numpy as np

from .datasets import ISIC2018Dataset, HAM10000Dataset, get_default_transform
from .models import create_model



def eval_on_dataset(model, loader, device):
    model.eval()
    auroc_metric = BinaryAUROC().to(device)

    start = time.time()

    # inference_mode = fastest + safe (no mutation)
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)   # AUROC needs float labels

            logits = model(imgs)
            auroc_metric.update(logits, labels)

    elapsed = time.time() - start
    return auroc_metric.compute().item(), elapsed



def safe_ham_mode(mode: str):
    if mode.startswith("whole"):
        return mode
    if "high" in mode:
        return "high_whole"
    if "low" in mode:
        return "low_whole"
    return "whole"



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isic_root", type=str, default="data/ISIC2018")
    parser.add_argument("--ham_root", type=str, default="data/HAM10000")
    parser.add_argument("--mode", type=str, default="whole")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/isic")
    parser.add_argument("--num_folds", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare output directory
    results_dir = Path("results/transfer") / args.mode
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "auroc_transfer_results.csv"

    ham_mode = safe_ham_mode(args.mode)
    ham_dataset = HAM10000Dataset(
        root=args.ham_root,
        mode=ham_mode,
        transform=get_default_transform(),
    )
    ham_loader = DataLoader(
        ham_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )

    isic_results = []
    ham_results = []
    fold_times = []

    global_start = time.time()


    for fold in range(args.num_folds):

        ckpt_path = Path(args.ckpt_dir) / args.mode / f"resnet50_fold{fold}.pt"
        print(f"\nLoading checkpoint: {ckpt_path}")

        if not ckpt_path.exists():
            print(f"WARNING: checkpoint for fold {fold} NOT FOUND, skipping.")
            continue

        state = torch.load(ckpt_path, map_location=device)

        model = create_model(pretrained=False, num_classes=1).to(device)
        model.load_state_dict(state["model_state"])

        # ISIC dataset (full eval)
        isic_dataset = ISIC2018Dataset(
            root=args.isic_root,
            mode=args.mode,
            transform=get_default_transform()
        )
        isic_loader = DataLoader(
            isic_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True
        )

        print(f"Evaluating fold {fold}...")

        fold_start = time.time()

        au_isic, t_isic = eval_on_dataset(model, isic_loader, device)
        au_ham, t_ham = eval_on_dataset(model, ham_loader, device)

        fold_elapsed = time.time() - fold_start

        print(
            f"Fold {fold}: "
            f"ISIC AUROC={au_isic:.4f} (t={t_isic:.1f}s) | "
            f"HAM AUROC={au_ham:.4f} (t={t_ham:.1f}s) | "
            f"Fold time={fold_elapsed/60:.2f} min"
        )

        isic_results.append(au_isic)
        ham_results.append(au_ham)
        fold_times.append(fold_elapsed)


    summary = {
        "mode": args.mode,
        "isic_mean": np.mean(isic_results),
        "isic_std": np.std(isic_results),
        "ham_mean": np.mean(ham_results),
        "ham_std": np.std(ham_results),
        "total_time_min": (time.time() - global_start) / 60
    }

    print("\n=== Final Summary ===")
    print(f"Mode: {args.mode}")
    print(f"ISIC AUROC: {summary['isic_mean']:.4f} ± {summary['isic_std']:.4f}")
    print(f"HAM  AUROC: {summary['ham_mean']:.4f} ± {summary['ham_std']:.4f}")
    print(f"Total evaluation time: {summary['total_time_min']:.2f} min")

    # Save CSV
    df = pd.DataFrame({
        "fold": list(range(len(isic_results))),
        "isic_auroc": isic_results,
        "ham_auroc": ham_results,
        "fold_time_sec": fold_times
    })

    df.loc[len(df)] = [
        "mean±std",
        f"{summary['isic_mean']:.4f} ± {summary['isic_std']:.4f}",
        f"{summary['ham_mean']:.4f} ± {summary['ham_std']:.4f}",
        ""
    ]

    df.to_csv(results_file, index=False)
    print(f"\nSaved AUROC results to:\n  {results_file}\n")


if __name__ == "__main__":
    main()
