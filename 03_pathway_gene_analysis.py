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
    MATRISOME_MODULES,
    RESULT_DIR,
    SUBTYPE_ORDER,
    annotate_res03,
    bh_fdr,
    configure_plotting,
    expression_matrix,
    get_sample_key,
    module_score,
    plot_heatmap,
    present_genes,
    read_adata,
    sample_metadata,
    savefig,
    spearman_table,
)


GSEA_LIBRARIES = ["GO_Biological_Process_2025", "Reactome_2024", "MSigDB_Hallmark_2020"]


def shorten_term(term, max_chars=80):
    term = str(term)
    if len(term) <= max_chars:
        return term
    return term[: max_chars - 3] + "..."


def parse_enrichr_genes(row):
    for col in ["Genes", "Lead_genes", "Overlap Genes"]:
        if col in row.index and pd.notna(row[col]):
            text = str(row[col]).replace(",", ";")
            genes = [x.strip() for x in text.split(";") if x.strip()]
            return genes
    return []


def sample_subtype_score_table(adata, score, value_name="score"):
    sample_key = get_sample_key(adata)
    df = adata.obs[[sample_key, AGE_KEY, ANNOTATION_KEY]].copy()
    df[value_name] = np.asarray(score)
    out = (
        df.dropna(subset=[sample_key, AGE_KEY, ANNOTATION_KEY, value_name])
        .groupby([sample_key, ANNOTATION_KEY], observed=True)
        .agg(
            age_num=(AGE_KEY, "median"),
            score=(value_name, "mean"),
            n_cells=(value_name, "size"),
        )
        .reset_index()
        .rename(columns={sample_key: "sample", ANNOTATION_KEY: "subtype"})
    )
    return out


def sample_subtype_gene_table(adata, genes):
    sample_key = get_sample_key(adata)
    genes = present_genes(adata, genes)
    expr = expression_matrix(adata, genes)
    meta = adata.obs[[sample_key, AGE_KEY, ANNOTATION_KEY]].copy()
    expr[sample_key] = meta[sample_key].astype(str).values
    expr[AGE_KEY] = meta[AGE_KEY].values
    expr["subtype"] = meta[ANNOTATION_KEY].astype(str).values
    out = (
        expr.dropna(subset=[sample_key, AGE_KEY, "subtype"])
        .groupby([sample_key, "subtype"], observed=True)
        .agg({**{gene: "mean" for gene in genes}, AGE_KEY: "median"})
        .reset_index()
        .rename(columns={sample_key: "sample"})
    )
    return out, genes


def compute_marker_genes(adata, n_genes=80):
    sc.tl.rank_genes_groups(
        adata,
        groupby=ANNOTATION_KEY,
        groups=SUBTYPE_ORDER,
        reference="rest",
        method="wilcoxon",
        key_added="rank_genes_fap_subtypes",
    )
    rows = []
    result = adata.uns["rank_genes_fap_subtypes"]
    for subtype in SUBTYPE_ORDER:
        names = result["names"][subtype][:n_genes]
        scores = result["scores"][subtype][:n_genes]
        pvals_adj = result["pvals_adj"][subtype][:n_genes]
        for gene, score, qval in zip(names, scores, pvals_adj):
            if pd.isna(gene):
                continue
            rows.append({"subtype": subtype, "gene": str(gene), "score": float(score), "qval": float(qval)})
    return pd.DataFrame(rows)


def run_enrichr(genes, subtype):
    try:
        import gseapy as gp

        enr = gp.enrichr(
            gene_list=list(genes),
            gene_sets=GSEA_LIBRARIES,
            organism="Human",
            outdir=None,
            cutoff=1.0,
            verbose=False,
        )
        out = enr.results.copy()
    except Exception as exc:
        print(f"Enrichr failed for {subtype}: {exc}")
        out = pd.DataFrame()
    if out.empty:
        return pd.DataFrame(columns=["subtype", "Term", "Adjusted P-value", "Combined Score"])
    out["subtype"] = subtype
    return out


def top_marker_pathways(marker_table, output_dir):
    enrichments = []
    for subtype in SUBTYPE_ORDER:
        genes = marker_table.loc[
            (marker_table["subtype"] == subtype) & (marker_table["qval"] < 0.05),
            "gene",
        ].head(80)
        if len(genes) < 10:
            genes = marker_table.loc[marker_table["subtype"] == subtype, "gene"].head(80)
        enrichments.append(run_enrichr(genes, subtype))
    enr = pd.concat(enrichments, ignore_index=True)
    if enr.empty:
        return enr

    enr["neglog10_q"] = -np.log10(enr["Adjusted P-value"].clip(lower=1e-300))
    top = (
        enr.sort_values(["subtype", "Adjusted P-value", "Combined Score"], ascending=[True, True, False])
        .groupby("subtype", observed=True)
        .head(5)
    )
    terms = list(dict.fromkeys(top["Term"].tolist()))
    mat = (
        top.pivot_table(index="Term", columns="subtype", values="neglog10_q", aggfunc="max")
        .reindex(terms)
        .reindex(columns=SUBTYPE_ORDER)
        .fillna(0)
    )
    plot_heatmap(
        mat,
        output_dir / "gsea_top5_marker_pathways_heatmap.pdf",
        cmap="Reds",
        center=None,
        vmin=0,
        cbar_label="-log10 adjusted p",
    )
    return enr


def pathway_score_by_cell(adata, terms, marker_table):
    scores = {}
    for subtype in SUBTYPE_ORDER:
        genes = marker_table.loc[marker_table["subtype"] == subtype, "gene"].head(100).tolist()
        genes = present_genes(adata, genes)
        if len(genes) >= 5:
            scores[f"{subtype} marker program"] = module_score(adata, genes)
    return pd.DataFrame(scores, index=adata.obs_names)


def top_age_changed_pathways(adata, marker_table, output_dir):
    if marker_table is None or marker_table.empty or "Term" not in marker_table.columns:
        scores = pathway_score_by_cell(adata, None, marker_table)
        scores[AGE_KEY] = adata.obs[AGE_KEY].values
        scores["subtype"] = adata.obs[ANNOTATION_KEY].astype(str).values
        rows = []
        for subtype in SUBTYPE_ORDER:
            idx = scores["subtype"] == subtype
            for pathway in [c for c in scores.columns if c.endswith("program")]:
                sub = scores.loc[idx, [pathway, AGE_KEY]].dropna()
                if sub[AGE_KEY].nunique() < 3 or sub[pathway].nunique() < 3:
                    rho, pval = np.nan, np.nan
                else:
                    rho, pval = spearmanr(sub[AGE_KEY], sub[pathway])
                rows.append({"subtype": subtype, "pathway": pathway, "rho": rho, "pval": pval})
        out = pd.DataFrame(rows)
        out["qval"] = out.groupby("subtype", observed=True)["pval"].transform(lambda x: bh_fdr(x.values))
        out["abs_rho"] = out["rho"].abs()
        top = out.sort_values(["subtype", "qval", "abs_rho"], ascending=[True, True, False]).groupby("subtype", observed=True).head(5)
        pathways = list(dict.fromkeys(top["pathway"].tolist()))
        mat = out.pivot(index="subtype", columns="pathway", values="rho").reindex(SUBTYPE_ORDER)[pathways]
        plot_heatmap(
            mat,
            output_dir / "gsea_top5_age_changed_pathways_heatmap.pdf",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            cbar_label="Spearman rho",
        )
        return out

    marker_table = marker_table.copy()
    marker_table["neglog10_q"] = -np.log10(marker_table["Adjusted P-value"].clip(lower=1e-300))
    top_terms = (
        marker_table.sort_values(["subtype", "Adjusted P-value", "Combined Score"], ascending=[True, True, False])
        .groupby("subtype", observed=True)
        .head(5)
    )
    rows = []
    for _, term_row in top_terms.iterrows():
        subtype = term_row["subtype"]
        genes = present_genes(adata, parse_enrichr_genes(term_row))
        if len(genes) < 3:
            continue
        score = module_score(adata, genes)
        score_df = sample_subtype_score_table(adata, score)
        sub = score_df.loc[score_df["subtype"] == subtype, ["age_num", "score"]].dropna()
        if sub["age_num"].nunique() < 3 or sub["score"].nunique() < 3:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(sub["age_num"], sub["score"])
        rows.append(
            {
                "subtype": subtype,
                "pathway": shorten_term(term_row["Term"]),
                "rho": rho,
                "pval": pval,
                "n_samples": sub.shape[0],
                "genes_used": ";".join(genes),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["qval"] = bh_fdr(out["pval"].values)
    out["abs_rho"] = out["rho"].abs()
    sig = out.loc[out["qval"] < 0.05]
    source = sig if not sig.empty else out
    top = source.sort_values(["subtype", "qval", "abs_rho"], ascending=[True, True, False]).groupby("subtype", observed=True).head(5)
    pathways = list(dict.fromkeys(top["pathway"].tolist()))
    mat = out.pivot(index="subtype", columns="pathway", values="rho").reindex(SUBTYPE_ORDER)[pathways]
    plot_heatmap(
        mat,
        output_dir / "gsea_top5_age_changed_pathways_heatmap.pdf",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        cbar_label="Spearman rho",
    )
    return out


def classic_pathway_scores(adata, output_dir):
    rows = []
    for pathway, genes in CLASSIC_PATHWAYS.items():
        genes = present_genes(adata, genes)
        if len(genes) < 3:
            continue
        score = module_score(adata, genes)
        tmp = sample_subtype_score_table(adata, score)
        tmp["pathway"] = pathway
        rows.append(tmp)
    df = pd.concat(rows, ignore_index=True)
    stats = spearman_table(df.rename(columns={"age_num": AGE_KEY}), "score", ["subtype", "pathway"])
    sig = stats.loc[stats["qval"] < 0.05].copy()
    if sig.empty:
        sig = stats.sort_values("pval").head(12)

    keep = sig[["subtype", "pathway"]].drop_duplicates()
    plot_df = df.merge(keep, on=["subtype", "pathway"], how="inner")
    panels = keep.to_dict("records")
    n = len(panels)
    ncol = 3
    nrow = int(np.ceil(n / ncol))

    configure_plotting()
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, max(3.0, 2.6 * nrow)), squeeze=False)
    axes = axes.ravel()
    for ax, panel in zip(axes, panels):
        sub = plot_df[(plot_df["subtype"] == panel["subtype"]) & (plot_df["pathway"] == panel["pathway"])].dropna()
        ax.scatter(sub["age_num"], sub["score"], s=18, color="#4C78A8", alpha=0.7, linewidth=0)
        if sub["age_num"].nunique() > 2:
            coef = np.polyfit(sub["age_num"], sub["score"], 1)
            xs = np.linspace(sub["age_num"].min(), sub["age_num"].max(), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="black", lw=1)
        row = stats[(stats["subtype"] == panel["subtype"]) & (stats["pathway"] == panel["pathway"])].iloc[0]
        ax.text(0.03, 0.95, f"rho={row['rho']:.2f}, q={row['qval']:.2g}", transform=ax.transAxes, va="top", fontsize=8)
        ax.set_title(f"{panel['subtype']} | {panel['pathway']}", fontsize=9)
        ax.set_xlabel("Age")
        ax.set_ylabel("Score")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    savefig(output_dir / "classic_pathway_scores_by_age_scatter.pdf", fig)
    return stats


def candidate_novel_genes(adata, output_dir):
    genes = present_genes(adata, CANDIDATE_AGE_GENES)
    sample_expr, genes = sample_subtype_gene_table(adata, genes)
    rows = []
    for subtype in SUBTYPE_ORDER:
        sub_df = sample_expr.loc[sample_expr["subtype"] == subtype]
        age = sub_df[AGE_KEY].values
        for gene in genes:
            values = sub_df[gene].values
            if pd.Series(age).nunique() < 3 or pd.Series(values).nunique() < 3:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = spearmanr(age, values)
            rows.append({"subtype": subtype, "gene": gene, "rho": rho, "pval": pval})
    table = pd.DataFrame(rows)
    table["qval"] = table.groupby("subtype", observed=True)["pval"].transform(lambda x: bh_fdr(x.values))
    table["score"] = table["rho"].abs()
    top = table.sort_values(["subtype", "score"], ascending=[True, False]).groupby("subtype", observed=True).head(8)
    genes = list(dict.fromkeys(top["gene"].tolist()))
    mat = table.pivot(index="subtype", columns="gene", values="rho").reindex(SUBTYPE_ORDER)[genes]
    plot_heatmap(
        mat,
        output_dir / "candidate_novel_age_genes.pdf",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        cbar_label="Spearman rho",
    )
    return table


def matrisome_age_analysis(adata, output_dir):
    rows = []
    for module, genes in MATRISOME_MODULES.items():
        genes = present_genes(adata, genes)
        if len(genes) < 3:
            continue
        score = module_score(adata, genes)
        tmp = sample_subtype_score_table(adata, score)
        tmp["module"] = module
        rows.append(tmp)
    df = pd.concat(rows, ignore_index=True)
    stats = spearman_table(df.rename(columns={"age_num": AGE_KEY}), "score", ["subtype", "module"])
    mat = stats.pivot(index="subtype", columns="module", values="rho").reindex(SUBTYPE_ORDER)
    plot_heatmap(
        mat,
        output_dir / "matrisome_ecm_age_analysis.pdf",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        cbar_label="Spearman rho",
    )
    return stats


def main():
    configure_plotting()
    output_dir = Path(RESULT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = read_adata()
    annotate_res03(adata)

    marker_table = compute_marker_genes(adata)
    enrichment_table = top_marker_pathways(marker_table, output_dir)
    top_age_changed_pathways(adata, enrichment_table, output_dir)
    classic_pathway_scores(adata, output_dir)
    candidate_novel_genes(adata, output_dir)
    matrisome_age_analysis(adata, output_dir)


if __name__ == "__main__":
    main()
