# src/analyze_artifacts.py

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pointbiserialr, pearsonr



def normalize_id_column(df):
    # Clean column names first
    df.columns = [c.strip().lstrip(";") for c in df.columns]

    possible_cols = ["isic_id", "image", "image_id", "img_id", "filename", "file"]

    for col in df.columns:
        if col.lower() in possible_cols:
            df = df.rename(columns={col: "isic_id"})
            df["isic_id"] = (
                df["isic_id"]
                .astype(str)
                .str.replace(".jpg", "", regex=False)
                .str.replace(".png", "", regex=False)
                .str.strip()
            )
            return df

    raise ValueError(
        f"No valid ID column found. Columns available: {df.columns.tolist()}"
    )




def load_data(label_path, artifact_path, pred_path):
    print("=== Loading ISIC labels ===")
    df_labels = pd.read_csv(label_path)

    print("=== Loading Bissotto artifact CSV ===")
    df_art = pd.read_csv(artifact_path, sep=";")
    df_art = normalize_id_column(df_art)

    print("=== Loading model predictions ===")
    df_pred = pd.read_csv(pred_path)
    df_pred = normalize_id_column(df_pred)

    print("\nDetected Columns:")
    print("Labels:", df_labels.columns.tolist())
    print("Artifacts:", df_art.columns.tolist())
    print("Predictions:", df_pred.columns.tolist())

    print("\n=== Merging dataframes ===")
    df = df_labels.merge(df_art, on="isic_id", how="inner")
    df = df.merge(df_pred, on="isic_id", how="inner")

    print(f"Merged dataframe shape: {df.shape}\n")

    # ---- FIX LABEL COLUMN ----
    if "label_x" in df.columns:
        df = df.rename(columns={"label_x": "label"})
    if "label_y" in df.columns:
        df = df.drop(columns=["label_y"])  # avoid duplicate label column

    return df




def compute_prevalence(df, artifact_cols):
    prevalence = {}
    for c in artifact_cols:
        prevalence[c] = df[c].mean()
    return pd.DataFrame.from_dict(prevalence, orient="index", columns=["prevalence"])



def compute_correlations(df, artifact_cols, pred_col="pred_mean"):
    corr_list = []

    for c in artifact_cols:
        # Skip non-artifact columns
        if c == "label" or c not in df.columns:
            continue

        # point-biserial correlation between artifact and melanoma label
        try:
            label_corr, _ = pointbiserialr(df[c], df["label"])
        except Exception:
            label_corr = np.nan

        # correlation between artifact and prediction
        try:
            pred_corr, _ = pointbiserialr(df[c], df[pred_col])
        except Exception:
            pred_corr = np.nan

        corr_list.append({
            "artifact": c,
            "corr_label": label_corr,
            "corr_prediction": pred_corr
        })

    return pd.DataFrame(corr_list)



def logistic_regression(df, artifact_cols):
    """
    Fit logistic regression: melanoma ~ artifacts.
    Only real artifact columns should be included.
    """
    # Filter only true artifact columns (binary indicators)
    true_artifacts = [
        c for c in artifact_cols
        if c in ["dark_corner", "hair", "gel_border", "gel_bubble",
                 "ruler", "ink", "patches"]
    ]

    print("\nUsing artifact predictors:", true_artifacts)

    X = df[true_artifacts].astype(float)
    X = sm.add_constant(X)

    y = df["label"].astype(int)

    # Fit logistic regression safely
    model = sm.Logit(y, X)
    try:
        result = model.fit(disp=0)
    except Exception as e:
        print("Logistic regression failed:", e)
        return None

    summary_df = pd.DataFrame({
        "artifact": ["intercept"] + true_artifacts,
        "coef": result.params.values,
        "p_value": result.pvalues.values
    })

    return summary_df



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isic_labels", type=str, default="data/ISIC2018/labels.csv")
    parser.add_argument("--bissoto_csv", type=str, default="metadata/isic_artifacts_bissoto.csv")
    parser.add_argument("--pred_csv", type=str, default="results/isic_predictions.csv")
    parser.add_argument("--out_dir", type=str, default="results/artifacts")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    df = load_data(args.isic_labels, args.bissoto_csv, args.pred_csv)

    # detect artifact columns
    artifact_cols = [
        c for c in df.columns
        if c not in ["isic_id", "label", "pred_mean"]
        and df[c].dtype in [np.float64, np.int64]
    ]

    print("\nDetected artifact columns:", artifact_cols)


    print("\n=== Artifact Prevalence ===")
    prev_df = compute_prevalence(df, artifact_cols)
    print(prev_df)
    prev_df.to_csv(out_dir / "artifact_prevalence.csv")


    print("\n=== Artifact Correlations ===")
    corr_df = compute_correlations(df, artifact_cols, pred_col="pred_mean")
    print(corr_df)
    corr_df.to_csv(out_dir / "artifact_correlations.csv", index=False)


    print("\n=== Logistic Regression (melanoma ~ artifacts) ===")
    logit_df = logistic_regression(df, artifact_cols)
    print(logit_df)
    logit_df.to_csv(out_dir / "logistic_regression.csv")

    print(f"\nAll results saved to: {out_dir}/")



if __name__ == "__main__":
    main()
