from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from utils import (
    AGE_GROUP_KEY,
    AGE_KEY,
    ANNOTATION_KEY,
    CLASSIC_PATHWAYS,
    MODEL_DIR,
    RESULT_DIR,
    SUBTYPE_ORDER,
    annotate_res03,
    bh_fdr,
    configure_plotting,
    expression_matrix,
    get_sample_key,
    plot_heatmap,
    present_genes,
    read_adata,
    sample_subtype_fractions,
    savefig,
    spearman_table,
    subtype_palette,
)


def try_load_mrvi_model(adata):
    if not MODEL_DIR.exists():
        return None
    try:
        from scvi.external import MRVI

        return MRVI.load(str(MODEL_DIR), adata=adata)
    except Exception as exc:
        print(f"MRVI model was not loaded; fallback summaries will be used. Reason: {exc}")
        return None


def inject_sample_covariates(model, adata):
    sample_key = model.sample_key
    sample_meta = sample_metadata_from_obs(adata, sample_key).set_index("sample")
    for cov in [AGE_KEY, AGE_GROUP_KEY]:
        if cov not in sample_meta.columns:
            continue
        mapped = pd.Series(model.sample_info[sample_key].astype(str).values).map(sample_meta[cov])
        model.sample_info[cov] = mapped.values


def subset_by_subtype(adata, max_cells_per_subtype=800, random_state=0):
    rng = np.random.default_rng(random_state)
    keep = []
    labels = adata.obs[ANNOTATION_KEY].astype(str).values
    for subtype in SUBTYPE_ORDER:
        idx = np.where(labels == subtype)[0]
        if len(idx) > max_cells_per_subtype:
            idx = rng.choice(idx, size=max_cells_per_subtype, replace=False)
        keep.extend(idx.tolist())
    return adata[np.asarray(keep, dtype=int)].copy()


def plot_age_umap(adata, output_dir):
    sc.pl.umap(
        adata,
        color=[AGE_KEY, ANNOTATION_KEY],
        palette=subtype_palette(),
        frameon=False,
        title=["", ""],
        show=False,
    )
    savefig(output_dir / "age_umap.pdf")


def plot_number_age_groups(fractions, output_dir):
    configure_plotting()
    groups = ["Infant", "Early Childhood", "Late Childhood", "Adolescence"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), sharey=False)
    axes = axes.ravel()
    for ax, subtype in zip(axes, SUBTYPE_ORDER):
        sub = fractions[fractions["subtype"] == subtype].copy()
        data = [sub.loc[sub[AGE_GROUP_KEY].astype(str) == group, "fraction"].values for group in groups]
        ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for i, vals in enumerate(data, 1):
            if len(vals):
                x = np.random.default_rng(0).normal(i, 0.04, size=len(vals))
                ax.scatter(x, vals, s=15, color=subtype_palette()[subtype], alpha=0.75, linewidth=0)
        ax.set_title(subtype, fontsize=10)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_ylabel("Fraction")
        ax.spines[["top", "right"]].set_visible(False)
    savefig(output_dir / "number_age_groups.pdf", fig)


def plot_number_continuous_age(fractions, output_dir):
    configure_plotting()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.8), sharey=False)
    axes = axes.ravel()
    stats = spearman_table(fractions, "fraction", ["subtype"])
    for ax, subtype in zip(axes, SUBTYPE_ORDER):
        sub = fractions[fractions["subtype"] == subtype].dropna(subset=[AGE_KEY, "fraction"])
        ax.scatter(sub[AGE_KEY], sub["fraction"], s=22, color=subtype_palette()[subtype], alpha=0.8, linewidth=0)
        if len(sub) >= 3 and sub[AGE_KEY].nunique() > 1:
            coef = np.polyfit(sub[AGE_KEY], sub["fraction"], deg=1)
            xs = np.linspace(sub[AGE_KEY].min(), sub[AGE_KEY].max(), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="black", lw=1)
        row = stats[stats["subtype"] == subtype]
        label = "rho=NA, q=NA" if row.empty else f"rho={row['rho'].iloc[0]:.2f}, q={row['qval'].iloc[0]:.3g}"
        ax.text(0.03, 0.95, label, transform=ax.transAxes, va="top", ha="left", fontsize=8)
        ax.set_title(subtype, fontsize=10)
        ax.set_xlabel("Age")
        ax.set_ylabel("Fraction")
        ax.spines[["top", "right"]].set_visible(False)
    savefig(output_dir / "number_continuous_age.pdf", fig)
    return stats


def plot_differential_abundance_age_group(fractions, output_dir, adata=None, model=None):
    if model is not None and adata is not None:
        try:
            inject_sample_covariates(model, adata)
            ds = model.differential_abundance(
                adata=adata,
                sample_cov_keys=[AGE_GROUP_KEY],
                compute_log_enrichment=True,
                omit_original_sample=True,
                batch_size=128,
            )
            key = f"{AGE_GROUP_KEY}_log_enrichs"
            if key in ds:
                values = ds[key].to_pandas()
                obs = adata.obs.loc[values.index, [ANNOTATION_KEY]]
                values[ANNOTATION_KEY] = obs[ANNOTATION_KEY].astype(str).values
                mat = values.groupby(ANNOTATION_KEY, observed=True).mean().reindex(SUBTYPE_ORDER)
                plot_heatmap(
                    mat,
                    output_dir / "differential_abundance_age_group_enrichment.pdf",
                    cmap="RdBu_r",
                    cbar_label="MRVI log enrichment",
                )
                return mat
        except Exception as exc:
            print(f"MRVI differential_abundance failed; using fraction enrichment. Reason: {exc}")

    overall = fractions.groupby("subtype", observed=True)["fraction"].mean()
    mat = (
        fractions.groupby(["subtype", AGE_GROUP_KEY], observed=True)["fraction"]
        .mean()
        .unstack(AGE_GROUP_KEY)
        .reindex(SUBTYPE_ORDER)
    )
    enrichment = np.log2((mat + 1e-4).div(overall + 1e-4, axis=0))
    plot_heatmap(
        enrichment,
        output_dir / "differential_abundance_age_group_enrichment.pdf",
        cmap="RdBu_r",
        cbar_label="log2 enrichment",
    )
    return enrichment


def plot_differential_abundance_continuous_age(fractions, output_dir):
    stats = spearman_table(fractions, "fraction", ["subtype"]).set_index("subtype").reindex(SUBTYPE_ORDER)
    mat = pd.DataFrame({"Spearman rho": stats["rho"]}, index=SUBTYPE_ORDER)
    plot_heatmap(
        mat,
        output_dir / "differential_abundance_continuous_age.pdf",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        cbar_label="rho",
    )
    return stats


def latent_age_effect_table(adata):
    if "X_mrvi_z" in adata.obsm:
        z = np.asarray(adata.obsm["X_mrvi_z"])
    elif "X_mrvi_u" in adata.obsm:
        z = np.asarray(adata.obsm["X_mrvi_u"])
    else:
        z = np.asarray(adata.X[:, : min(50, adata.n_vars)])
    rows = []
    for subtype in SUBTYPE_ORDER:
        idx = np.asarray(adata.obs[ANNOTATION_KEY].astype(str) == subtype)
        age = adata.obs.loc[idx, AGE_KEY].values
        if idx.sum() < 10:
            continue
        for dim in range(z.shape[1]):
            values = z[idx, dim]
            if pd.Series(age).nunique() < 3 or pd.Series(values).nunique() < 3:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(age, values)
            rows.append({"subtype": subtype, "latent_dim": dim, "rho": rho, "pval": pval})
    out = pd.DataFrame(rows)
    out["qval"] = out.groupby("subtype", observed=True)["pval"].transform(lambda x: bh_fdr(x.values))
    return out


def pick_covariate_name(de, preferred=AGE_KEY):
    for dim in ["covariate", "covariate_sub"]:
        if dim in de.coords:
            names = [str(x) for x in de.coords[dim].values]
            for name in names:
                if name == preferred:
                    return dim, name
            for name in names:
                if preferred in name or "age" in name.lower():
                    return dim, name
            if names:
                return dim, names[0]
    return None, None


def plot_age_effect(adata, output_dir, model=None):
    if model is not None:
        try:
            inject_sample_covariates(model, adata)
            adata_sub = subset_by_subtype(adata, max_cells_per_subtype=1000, random_state=0)
            de = model.differential_expression(
                adata=adata_sub,
                sample_cov_keys=[AGE_KEY],
                batch_size=64,
                use_vmap="auto",
                mc_samples=25,
                store_lfc=False,
                filter_inadmissible_samples=False,
            )
            dim, cov = pick_covariate_name(de, AGE_KEY)
            if dim is not None:
                effect = de["effect_size"].sel({dim: cov}).to_pandas()
                padj = de["padj"].sel({dim: cov}).to_pandas()
                meta = adata_sub.obs.loc[effect.index, [ANNOTATION_KEY]]
                tmp = pd.DataFrame(
                    {
                        "subtype": meta[ANNOTATION_KEY].astype(str).values,
                        "effect_size": effect.values,
                        "padj": padj.values,
                    },
                    index=effect.index,
                )
                summary = (
                    tmp.groupby("subtype", observed=True)
                    .agg(
                        mean_effect_size=("effect_size", "mean"),
                        median_effect_size=("effect_size", "median"),
                        significant_cell_fraction=("padj", lambda x: float(np.mean(x < 0.05))),
                    )
                    .reindex(SUBTYPE_ORDER)
                )
                plot_heatmap(
                    summary,
                    output_dir / "differential_expression_age_effect.pdf",
                    cmap="Reds",
                    center=None,
                    vmin=0,
                    cbar_label="MRVI age effect",
                )
                return de, summary
        except Exception as exc:
            print(f"MRVI differential_expression effect_size failed; using latent effect. Reason: {exc}")

    table = latent_age_effect_table(adata)
    summary = (
        table.assign(abs_rho=lambda x: x["rho"].abs())
        .groupby("subtype", observed=True)
        .agg(mean_abs_age_effect=("abs_rho", "mean"), max_abs_age_effect=("abs_rho", "max"), significant_dims=("qval", lambda x: int((x < 0.05).sum())))
        .reindex(SUBTYPE_ORDER)
    )
    plot_heatmap(
        summary,
        output_dir / "differential_expression_age_effect.pdf",
        cmap="Reds",
        center=None,
        vmin=0,
        cbar_label="effect size",
    )
    return table, summary


def age_gene_effect_table(adata, candidate_genes):
    genes = present_genes(adata, candidate_genes)
    expr = expression_matrix(adata, genes)
    rows = []
    for subtype in SUBTYPE_ORDER:
        idx = np.asarray(adata.obs[ANNOTATION_KEY].astype(str) == subtype)
        age = adata.obs.loc[idx, AGE_KEY].values
        for gene in genes:
            values = expr.loc[idx, gene].values
            if pd.Series(age).nunique() < 3 or pd.Series(values).nunique() < 3:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(age, values)
            rows.append({"subtype": subtype, "gene": gene, "rho": rho, "pval": pval})
    out = pd.DataFrame(rows)
    out["qval"] = out.groupby("subtype", observed=True)["pval"].transform(lambda x: bh_fdr(x.values))
    return out


def plot_gene_level_age_effect(adata, output_dir, model=None):
    if model is not None:
        try:
            inject_sample_covariates(model, adata)
            adata_sub = subset_by_subtype(adata, max_cells_per_subtype=350, random_state=1)
            de = model.differential_expression(
                adata=adata_sub,
                sample_cov_keys=[AGE_KEY],
                batch_size=32,
                use_vmap="auto",
                mc_samples=25,
                store_lfc=True,
                store_lfc_metadata_subset=[AGE_KEY],
                delta=0.3,
                filter_inadmissible_samples=False,
            )
            dim, cov = pick_covariate_name(de, AGE_KEY)
            if dim is not None and "lfc" in de and "pde" in de:
                lfc = de["lfc"].sel({dim: cov}).to_pandas()
                pde = de["pde"].sel({dim: cov}).to_pandas()
                meta = adata_sub.obs.loc[lfc.index, [ANNOTATION_KEY]]
                rows = []
                for subtype in SUBTYPE_ORDER:
                    idx = meta[ANNOTATION_KEY].astype(str).values == subtype
                    if idx.sum() == 0:
                        continue
                    mean_lfc = lfc.loc[idx].mean(axis=0)
                    mean_pde = pde.loc[idx].mean(axis=0)
                    tab = pd.DataFrame({"gene": mean_lfc.index, "mean_lfc": mean_lfc.values, "mean_pde": mean_pde.values})
                    tab = tab[np.isfinite(tab["mean_lfc"]) & np.isfinite(tab["mean_pde"])]
                    tab["subtype"] = subtype
                    up = tab[tab["mean_lfc"] > 0].sort_values(["mean_pde", "mean_lfc"], ascending=[False, False]).head(5)
                    down = tab[tab["mean_lfc"] < 0].sort_values(["mean_pde", "mean_lfc"], ascending=[False, True]).head(5)
                    rows.append(pd.concat([up, down], ignore_index=True))
                table = pd.concat(rows, ignore_index=True)
                genes = list(dict.fromkeys(table["gene"].tolist()))

                configure_plotting()
                fig, ax = plt.subplots(figsize=(max(7, 0.32 * len(genes) + 2), 3.5))
                y_pos = {subtype: i for i, subtype in enumerate(SUBTYPE_ORDER)}
                x_pos = {gene: i for i, gene in enumerate(genes)}
                for _, row in table.iterrows():
                    ax.scatter(
                        x_pos[row["gene"]],
                        y_pos[row["subtype"]],
                        s=25 + 130 * float(row["mean_pde"]),
                        c=[float(row["mean_lfc"])],
                        cmap="RdBu_r",
                        vmin=-1.5,
                        vmax=1.5,
                        edgecolor="black",
                        linewidth=0.35,
                    )
                ax.set_xticks(range(len(genes)))
                ax.set_xticklabels(genes, rotation=45, ha="right")
                ax.set_yticks(range(len(SUBTYPE_ORDER)))
                ax.set_yticklabels(SUBTYPE_ORDER)
                ax.invert_yaxis()
                ax.spines[["top", "right"]].set_visible(False)
                cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-1.5, 1.5)), ax=ax, fraction=0.035, pad=0.02)
                cbar.set_label("MRVI age LFC")
                savefig(output_dir / "gene_level_age_lfc_dotplot.pdf", fig)
                return table
        except Exception as exc:
            print(f"MRVI differential_expression lfc/pde failed; using expression-age correlation. Reason: {exc}")

    candidate_genes = sorted(set(sum(CLASSIC_PATHWAYS.values(), [])))
    table = age_gene_effect_table(adata, candidate_genes)
    table["score"] = table["rho"].abs()
    keep = (
        table.sort_values(["subtype", "score"], ascending=[True, False])
        .groupby("subtype", observed=True)
        .head(10)
    )
    genes = list(dict.fromkeys(keep["gene"]))
    mat = table.pivot(index="subtype", columns="gene", values="rho").reindex(SUBTYPE_ORDER)[genes]

    configure_plotting()
    fig, ax = plt.subplots(figsize=(max(7, 0.32 * len(genes) + 2), 3.5))
    y_pos = {subtype: i for i, subtype in enumerate(SUBTYPE_ORDER)}
    x_pos = {gene: i for i, gene in enumerate(genes)}
    for _, row in table[table["gene"].isin(genes)].iterrows():
        if row["subtype"] not in y_pos:
            continue
        size = 25 + 120 * min(abs(row["rho"]) if np.isfinite(row["rho"]) else 0, 1)
        ax.scatter(
            x_pos[row["gene"]],
            y_pos[row["subtype"]],
            s=size,
            c=[row["rho"]],
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            edgecolor="black" if row["qval"] < 0.05 else "none",
            linewidth=0.5,
        )
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="right")
    ax.set_yticks(range(len(SUBTYPE_ORDER)))
    ax.set_yticklabels(SUBTYPE_ORDER)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-1, 1)), ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Spearman rho")
    savefig(output_dir / "gene_level_age_lfc_dotplot.pdf", fig)
    return table


def plot_local_sample_distance(adata, output_dir):
    sample_key = get_sample_key(adata)
    if "X_mrvi_z" in adata.obsm:
        z = np.asarray(adata.obsm["X_mrvi_z"])
    elif "X_mrvi_u" in adata.obsm:
        z = np.asarray(adata.obsm["X_mrvi_u"])
    else:
        z = np.asarray(adata.X[:, : min(50, adata.n_vars)])
    sample_df = pd.DataFrame(z, index=adata.obs_names)
    sample_df["sample"] = adata.obs[sample_key].astype(str).values
    centroids = sample_df.groupby("sample").mean()
    ages = sample_metadata_from_obs(adata, sample_key).set_index("sample").loc[centroids.index, AGE_KEY]
    dist = squareform(pdist(centroids.values, metric="euclidean"))
    age_gap = squareform(pdist(ages.values.reshape(-1, 1), metric="euclidean"))
    iu = np.triu_indices_from(dist, k=1)
    x = age_gap[iu]
    y = dist[iu]
    rho, pval = spearmanr(x, y) if len(np.unique(x)) > 2 and len(np.unique(y)) > 2 else (np.nan, np.nan)

    configure_plotting()
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.scatter(x, y, s=20, color="#4C78A8", alpha=0.75, linewidth=0)
    if len(x) > 2:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="black", lw=1)
    ax.text(0.03, 0.95, f"rho={rho:.2f}, p={pval:.3g}", transform=ax.transAxes, va="top", ha="left", fontsize=8)
    ax.set_xlabel("Age gap")
    ax.set_ylabel("MRVI latent distance")
    ax.spines[["top", "right"]].set_visible(False)
    savefig(output_dir / "local_sample_distance_age.pdf", fig)


def sample_metadata_from_obs(adata, sample_key):
    meta = (
        adata.obs[[sample_key, AGE_KEY, AGE_GROUP_KEY]]
        .dropna(subset=[sample_key, AGE_KEY])
        .groupby(sample_key, observed=True)
        .agg({AGE_KEY: "median", AGE_GROUP_KEY: lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan})
        .reset_index()
        .rename(columns={sample_key: "sample"})
    )
    meta["sample"] = meta["sample"].astype(str)
    return meta


def plot_outlier_admissibility(adata, output_dir):
    if "X_mrvi_z" in adata.obsm:
        z = np.asarray(adata.obsm["X_mrvi_z"])
    elif "X_mrvi_u" in adata.obsm:
        z = np.asarray(adata.obsm["X_mrvi_u"])
    else:
        z = np.asarray(adata.X[:, : min(50, adata.n_vars)])
    labels = adata.obs[ANNOTATION_KEY].astype(str).values
    centroids = {}
    for subtype in SUBTYPE_ORDER:
        idx = labels == subtype
        if idx.sum():
            centroids[subtype] = z[idx].mean(axis=0)
    own_dist = np.zeros(adata.n_obs)
    best_dist = np.zeros(adata.n_obs)
    for i, subtype in enumerate(labels):
        ds = {name: np.linalg.norm(z[i] - center) for name, center in centroids.items()}
        own_dist[i] = ds.get(subtype, np.nan)
        best_dist[i] = min(ds.values()) if ds else np.nan
    admissibility = -(own_dist - best_dist)
    adata.obs["admissibility_proxy"] = admissibility

    sc.pl.umap(
        adata,
        color="admissibility_proxy",
        cmap="viridis",
        frameon=False,
        title="",
        show=False,
    )
    savefig(output_dir / "outlier_admissibility_umap.pdf")

    tmp = adata.obs[[ANNOTATION_KEY, AGE_GROUP_KEY, "admissibility_proxy"]].copy()
    mat = (
        tmp.groupby([ANNOTATION_KEY, AGE_GROUP_KEY], observed=True)["admissibility_proxy"]
        .mean()
        .unstack(AGE_GROUP_KEY)
        .reindex(SUBTYPE_ORDER)
    )
    plot_heatmap(
        mat,
        output_dir / "outlier_admissibility_age_heatmaps.pdf",
        cmap="viridis",
        center=None,
        cbar_label="mean admissibility",
    )


def main():
    configure_plotting()
    output_dir = Path(RESULT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = read_adata()
    annotate_res03(adata)
    model = try_load_mrvi_model(adata)

    fractions = sample_subtype_fractions(adata)
    plot_age_umap(adata, output_dir)
    plot_number_age_groups(fractions, output_dir)
    plot_number_continuous_age(fractions, output_dir)
    plot_differential_abundance_age_group(fractions, output_dir, adata=adata, model=model)
    plot_differential_abundance_continuous_age(fractions, output_dir)
    plot_age_effect(adata, output_dir, model=model)
    plot_gene_level_age_effect(adata, output_dir, model=model)
    plot_local_sample_distance(adata, output_dir)
    plot_outlier_admissibility(adata, output_dir)


if __name__ == "__main__":
    main()
