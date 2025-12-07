# src/predict_isic.py
import argparse
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.functional import sigmoid
from tqdm import tqdm

from .datasets import ISIC2018Dataset, get_default_transform
from .models import create_model


def predict_fold(ckpt_path, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    state = torch.load(ckpt_path, map_location=device)
    model = create_model(pretrained=False, num_classes=1).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    preds = []
    ids = []

    with torch.inference_mode():
        for imgs, labels, image_ids in tqdm(loader, desc=f"Predict {ckpt_path}"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = sigmoid(logits).cpu().numpy().flatten()

            preds.extend(probs)
            ids.extend(image_ids)

    return pd.DataFrame({"isic_id": ids, ckpt_path.stem: preds})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isic_root", type=str, default="data/ISIC2018")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/isic/whole")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_folds", type=int, default=3)
    parser.add_argument("--out_csv", type=str, default="results/isic_predictions.csv")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ISIC2018Dataset(
        root=args.isic_root,
        mode="whole",  # predictions always generated on WHOLE
        transform=get_default_transform()
    )

    all_preds = []

    for fold in range(args.num_folds):
        ckpt_path = Path(args.ckpt_dir) / f"resnet50_fold{fold}.pt"
        df_fold = predict_fold(ckpt_path, dataset, device, args.batch_size)
        df_fold = df_fold.rename(columns={f"resnet50_fold{fold}": f"pred_fold{fold}"})
        all_preds.append(df_fold)

    # merge all folds by isic_id
    df = all_preds[0]
    for d in all_preds[1:]:
        df = df.merge(d, on="isic_id")

    # mean prediction across folds
    pred_cols = [f"pred_fold{i}" for i in range(args.num_folds)]
    df["pred_mean"] = df[pred_cols].mean(axis=1)

    Path("results").mkdir(exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"\nSaved predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
