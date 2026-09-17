from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

from utils import (
    AGE_KEY,
    ANNOTATION_KEY,
    CANDIDATE_AGE_GENES,
    CLASSIC_PATHWAYS,
    RESULT_DIR,
    SUBTYPE_ORDER,
    annotate_res03,
    bh_fdr,
    configure_plotting,
    expression_matrix,
    module_score,
    plot_heatmap,
    present_genes,
    read_adata,
    savefig,
    subtype_palette,
)


def balanced_age_subset(adata, max_cells_per_age=150, random_state=0):
    rng = np.random.default_rng(random_state)
    idx = []
    for age, positions in adata.obs.groupby(AGE_KEY, observed=True).indices.items():
        positions = np.asarray(positions)
        if len(positions) > max_cells_per_age:
            positions = rng.choice(positions, size=max_cells_per_age, replace=False)
        idx.extend(positions.tolist())
    return np.asarray(idx, dtype=int)


def run_sceptic(adata):
    from sceptic import run_sceptic_and_evaluate
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    if "X_mrvi_z" in adata.obsm:
        x = np.asarray(adata.obsm["X_mrvi_z"])
    elif "X_mrvi_u" in adata.obsm:
        x = np.asarray(adata.obsm["X_mrvi_u"])
    else:
        x = np.asarray(adata.X[:, : min(50, adata.n_vars)])

    y = adata.obs[AGE_KEY].astype(float).values
    valid = np.isfinite(y)
    x = x[valid]
    y = y[valid]
    label_list = np.sort(np.unique(y))

    subset = balanced_age_subset(adata[valid].copy(), max_cells_per_age=150)
    x_sub = x[subset]
    y_sub = y[subset]

    params = {
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 80,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 1,
        "verbosity": 0,
    }
    result = run_sceptic_and_evaluate(
        x_sub,
        y_sub,
        label_list=label_list,
        parameters=params,
        method="xgboost",
        use_gpu=False,
    )

    model = XGBClassifier(**params)
    y_codes = pd.Categorical(y_sub, categories=label_list).codes
    model.fit(x_sub, y_codes)
    prob_all = model.predict_proba(x)
    pseudotime = prob_all @ label_list[: prob_all.shape[1]]

    out = adata[valid].copy()
    out.obs["sceptic_age_pseudotime"] = pseudotime
    out.obs["sceptic_age_delta"] = pseudotime - y
    return out, result


def plot_sceptic_umap(adata, output_dir):
    sc.pl.umap(
        adata,
        color=["sceptic_age_pseudotime", "sceptic_age_delta", ANNOTATION_KEY],
        palette=subtype_palette(),
        frameon=False,
        title=["", "", ""],
        show=False,
    )
    savefig(output_dir / "sceptic_age_pseudotime_umap.pdf")


def plot_validation_and_subtypes(adata, output_dir):
    configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))

    ax = axes[0]
    ax.scatter(adata.obs[AGE_KEY], adata.obs["sceptic_age_pseudotime"], s=5, color="#4C78A8", alpha=0.35, linewidth=0)
    rho, pval = spearmanr(adata.obs[AGE_KEY], adata.obs["sceptic_age_pseudotime"])
    ax.text(0.03, 0.95, f"rho={rho:.2f}, p={pval:.2g}", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel("Age")
    ax.set_ylabel("Sceptic pseudotime")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    data = [
        adata.obs.loc[adata.obs[ANNOTATION_KEY].astype(str) == subtype, "sceptic_age_pseudotime"].dropna().values
        for subtype in SUBTYPE_ORDER
    ]
    ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    ax.set_xticks(range(1, len(SUBTYPE_ORDER) + 1))
    ax.set_xticklabels(SUBTYPE_ORDER, rotation=45, ha="right")
    ax.set_ylabel("Sceptic pseudotime")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(output_dir / "sceptic_age_pseudotime_validation_and_subtypes.pdf", fig)


def plot_confusion_matrix(result, output_dir):
    cm = None
    for key in ["confusion_matrix", "cm", "test_confusion_matrix"]:
        if isinstance(result, dict) and key in result:
            cm = result[key]
            break
    if cm is None:
        return
    configure_plotting()
    fig, ax = plt.subplots(figsize=(4, 3.6))
    im = ax.imshow(np.asarray(cm), cmap="Blues", aspect="auto")
    ax.set_xlabel("Predicted age")
    ax.set_ylabel("Observed age")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    savefig(output_dir / "sceptic_age_prediction_confusion_matrix.pdf", fig)


def pseudotime_heatmap(adata, modules, output_path):
    bins = pd.qcut(adata.obs["sceptic_age_pseudotime"], q=10, duplicates="drop")
    rows = []
    for module, genes in modules.items():
        genes = present_genes(adata, genes)
        if len(genes) < 2:
            continue
        score = module_score(adata, genes)
        tmp = pd.DataFrame({"bin": bins, "score": score.values})
        means = tmp.groupby("bin", observed=True)["score"].mean()
        means = (means - means.mean()) / (means.std() + 1e-9)
        means.name = module
        rows.append(means)
    if not rows:
        return
    mat = pd.concat(rows, axis=1).T
    plot_heatmap(mat, output_path, cmap="RdBu_r", cbar_label="z-score")


def candidate_gene_pseudotime_heatmap(adata, output_dir):
    genes = present_genes(adata, CANDIDATE_AGE_GENES)
    expr = expression_matrix(adata, genes)
    bins = pd.qcut(adata.obs["sceptic_age_pseudotime"], q=10, duplicates="drop")
    mat = expr.groupby(bins, observed=True).mean().T
    mat = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1) + 1e-9, axis=0)
    plot_heatmap(
        mat,
        output_dir / "sceptic_candidate_gene_pseudotime_heatmap.pdf",
        cmap="RdBu_r",
        cbar_label="z-score",
    )


def main():
    configure_plotting()
    output_dir = Path(RESULT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = read_adata()
    annotate_res03(adata)
    sceptic_adata, result = run_sceptic(adata)

    plot_sceptic_umap(sceptic_adata, output_dir)
    plot_validation_and_subtypes(sceptic_adata, output_dir)
    plot_confusion_matrix(result, output_dir)
    pseudotime_heatmap(sceptic_adata, CLASSIC_PATHWAYS, output_dir / "sceptic_functional_module_pseudotime_heatmap.pdf")
    candidate_gene_pseudotime_heatmap(sceptic_adata, output_dir)


if __name__ == "__main__":
    main()
