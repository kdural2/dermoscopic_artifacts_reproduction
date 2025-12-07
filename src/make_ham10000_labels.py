from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
raw_meta_path = PROJECT_ROOT / "data" / "HAM10000" / "HAM10000_metadata.csv"
out_path = PROJECT_ROOT / "data" / "HAM10000" / "labels.csv"

df = pd.read_csv(raw_meta_path)  # Kaggle HAM10000_metadata.csv style

benign_classes = {"bkl", "df", "nv", "vasc"}
malignant_classes = {"akiec", "bcc", "mel"}

def to_binary(dx):
    dx = dx.lower()
    if dx in malignant_classes:
        return 1
    elif dx in benign_classes:
        return 0
    else:
        raise ValueError(f"Unknown class: {dx}")

df["label"] = df["dx"].apply(to_binary)

labels = df[["image_id", "label"]].drop_duplicates()
labels.to_csv(out_path, index=False)
print(f"Wrote {len(labels)} rows to {out_path}")
