from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
raw_meta_path = PROJECT_ROOT / "data" / "ISIC2018" / "isic_bias_raw.csv"
out_path = PROJECT_ROOT / "data" / "ISIC2018" / "labels.csv"

# Adjust column names based on your raw metadata file
df = pd.read_csv(raw_meta_path)

# Example: if there's a 'diagnosis' column
def to_binary(label: str) -> int:
    # adjust mapping based on how ISIC labels melanoma vs benign
    # Example: 'MEL' or 'melanoma' -> 1, else 0
    label = label.lower()
    return 1 if "malignant" in label else 0

df["label"] = df["diagnosis_1"].apply(to_binary)
labels = df[["isic_id", "label"]].drop_duplicates()

labels.to_csv(out_path, index=False)
print(f"Wrote {len(labels)} rows to {out_path}")
