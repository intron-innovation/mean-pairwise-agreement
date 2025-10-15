#!/usr/bin/env python3
"""
Run ART ANOVA + Pairwise comparison on evaluation metrics.

Usage:
    python run_pairwise.py --input data.csv --output pvalues.csv

If no output path is given, results are printed to console.
"""

import argparse
import pandas as pd
import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp
from scipy.stats import rankdata


# -------------------------------
# Metric definitions
# -------------------------------
content_metrics = [
    "factuality",
    "appropriatness",
    "adequacy",
    "self_awareness",
    "clinical reasoning"
]

communication_metrics = [
    "empathy",
    "fluency/clarity",
]

safety_metrics = [
    "hallucination",
    "bias",
    "harm",
]

all_metrics = content_metrics + communication_metrics + safety_metrics



def metrics_mean_ci(df, metrics, group_col="model", confidence=0.95):
    """
    Create a metrics × group table with mean and margin of error (CI).
    """
    def mean_ci(series):
        data = series.dropna().values
        n = len(data)
        if n == 0:
            return np.nan, np.nan
        mean = np.mean(data)
        sem = stats.sem(data)
        moe = sem * stats.t.ppf((1 + confidence) / 2, n - 1) if n > 1 else 0
        return mean, moe
    
    results = {}
    for metric in metrics:
        row = {}
        for group, subset in df.groupby(group_col):
            mean, moe = mean_ci(subset[metric])
            row[group] = f"{mean:.2f} ({moe:.2f})"
        results[metric] = row
    
    return pd.DataFrame(results).T


def split_mean_error(df):
    """
    Splits a dataframe of strings like '0.72 (0.05)' into numeric mean values.
    """
    mean_df = df.copy()
    for col in df.columns:
        mean_df[col] = df[col].str.extract(r'([0-9]*\.?[0-9]+)').astype(float)
    return mean_df


def art_anova_pairwise(df, metric_col="score", model_col="model", subject_col=None):
    """
    Perform ART-style ANOVA + pairwise tests and return n x n p-value matrix.
    """
    df["ranked"] = rankdata(df[metric_col])
    models = df[model_col].unique()
    p_values = sp.posthoc_dunn(df, val_col="ranked", group_col=model_col, p_adjust="holm")
    p_values = p_values.loc[models, models]
    return p_values


def final_pairwise(df):
    """
    Run the full pipeline: mean+CI → reshape → ART ANOVA → pairwise p-values.
    """
    df1 = metrics_mean_ci(df, all_metrics, group_col="model")
    df2 = split_mean_error(df1)

    df_long = df2.reset_index().melt(
        id_vars="index",
        var_name="model",
        value_name="score"
    ).rename(columns={"index": "metric"})

    df3 = art_anova_pairwise(df_long, metric_col="score", model_col="model")
    return df3



def main():
    parser = argparse.ArgumentParser(description="Run ART ANOVA + Pairwise comparison.")
    parser.add_argument("--input", required=True, help="Path to input CSV file containing model and metric columns.")
    parser.add_argument("--output", help="Optional path to save the p-value matrix as CSV.")
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.input)
    print(f"Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")

    # Run analysis
    result = final_pairwise(df)

    # Output
    if args.output:
        result.to_csv(args.output)
        print(f"Results saved to {args.output}")
    else:
        print("\nPairwise p-value matrix:\n")
        print(result)


if __name__ == "__main__":
    main()
