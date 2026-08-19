from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
import matplotlib as mpl
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
H5AD_PATH = PROJECT_DIR / "adata_mrvi_faps_subset.h5ad"
MODEL_DIR = PROJECT_DIR / "mrvi_model_cpu"
ANNOTATION_DIR = PROJECT_DIR / "fap_mrvi_age"
RESULT_DIR = PROJECT_DIR / "mrvi_age_analysis_res0.3_pdfs"

RESOLUTION = 0.3
LEIDEN_KEY = "leiden_mrvi_res0.3"
ANNOTATION_KEY = "fap_mrvi_annotation_res0.3"
AGE_KEY = "age_num"
AGE_GROUP_KEY = "age_group"

SUBTYPE_ORDER = ["MME+", "LUM+", "CD55+", "GPC3+", "COL11A1+", "ACTA1+"]

RES03_CLUSTER_TO_SUBTYPE = {
    "0": "CD55+",
    "1": "GPC3+",
    "2": "MME+",
    "3": "COL11A1+",
    "4": "MME+",
    "5": "ACTA1+",
    "6": "MME+",
    "7": "LUM+",
}

SUBTYPE_FEATURE_GENES = {
    "MME+": ["MME", "SMOC2", "COL15A1", "LAMA2"],
    "LUM+": ["LUM", "CXCL14", "PTGDS", "DPT"],
    "CD55+": ["CD55", "MFAP5", "PCOLCE2", "DPP4"],
    "GPC3+": ["GPC3", "CNTN4", "NRP1", "FBLN1"],
    "COL11A1+": ["COL11A1", "COL11A2", "THBS4", "BMPR1B"],
    "ACTA1+": ["ACTA1", "CKM", "TTN", "TNNC2"],
}

FEATUREPLOT_GENES = ["MME", "CXCL14", "PTGDS", "CD55", "GPC3", "COL11A1", "ACTA1", "LUM"]

CLASSIC_PATHWAYS = {
    "ECM organization": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "COL5A1",
        "COL5A2",
        "COL6A1",
        "COL6A2",
        "COL6A3",
        "DCN",
        "LUM",
        "FN1",
        "TNC",
        "MMP2",
        "TIMP1",
    ],
    "Basement membrane": ["COL4A1", "COL4A2", "COL15A1", "LAMA2", "LAMB1", "LAMB2", "NID1", "NID2", "HSPG2"],
    "WNT/development": ["WNT2", "WNT5A", "WNT10B", "SFRP1", "SFRP2", "GPC3", "RSPO3", "FZD1", "LEF1"],
    "Inflammatory/chemokine": ["CXCL14", "CCL2", "CXCL10", "IL6", "PTGDS", "CFD", "C7", "C3"],
    "Adipogenic/lipid": ["PPARG", "CEBPA", "CEBPB", "ADIPOQ", "FABP4", "LPL", "PLIN2", "ABCA8", "ABCA10"],
    "Myogenic contamination/response": ["ACTA1", "CKM", "TTN", "RYR1", "MYH1", "MYH2", "MYH7", "TNNC2"],
    "Growth factor signaling": ["IGF1", "FGF7", "HGF", "TGFB1", "BMP4", "PDGFRA", "PDGFRB", "IGFBP5"],
    "Mechanotransduction": ["POSTN", "THBS4", "COMP", "FBN1", "ITGA11", "TAGLN", "ACTA2", "YAP1"],
}

MATRISOME_MODULES = {
    "Collagens": [
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "COL5A1",
        "COL5A2",
        "COL6A1",
        "COL6A2",
        "COL6A3",
        "COL11A1",
        "COL11A2",
        "COL12A1",
        "COL14A1",
        "COL15A1",
    ],
    "ECM glycoproteins": ["FN1", "TNC", "THBS1", "THBS4", "POSTN", "COMP", "LAMA2", "LAMB1", "LAMB2", "FBN1"],
    "Proteoglycans": ["DCN", "LUM", "BGN", "OGN", "HSPG2", "VCAN", "GPC3", "PRG4"],
    "ECM regulators": ["MMP2", "MMP14", "TIMP1", "TIMP2", "LOX", "LOXL1", "PCOLCE", "PCOLCE2"],
    "Secreted factors": ["IGF1", "FGF7", "HGF", "CXCL14", "PTGDS", "CFD", "C7", "CCL2"],
}

CANDIDATE_AGE_GENES = [
    "ABCA8",
    "ABCA10",
    "IGF1",
    "CCN5",
    "CRLF1",
    "VIT",
    "PTGIS",
    "ABLIM3",
    "GREB1L",
    "PRKG1",
    "AJAP1",
    "FKBP5",
    "GPHN",
    "LRRC7",
    "COL15A1",
    "LAMA2",
]


def configure_plotting():
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"
    sc.settings.set_figure_params(dpi=120, dpi_save=300, frameon=False, fontsize=9)


def savefig(path, fig=None, close=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, bbox_inches="tight")
    if close:
        plt.close(fig)


def patch_null_log1p_base(h5ad_path):
    h5ad_path = Path(h5ad_path)
    if not h5ad_path.exists():
        return
    with h5py.File(h5ad_path, "r+") as handle:
        if "uns/log1p/base" in handle:
            del handle["uns/log1p/base"]


def read_adata(h5ad_path=H5AD_PATH, patch_log1p=True):
    if patch_log1p:
        patch_null_log1p_base(h5ad_path)
    adata = sc.read_h5ad(h5ad_path)
    ensure_age_columns(adata)
    return adata


def ensure_age_columns(adata):
    if "age" not in adata.obs:
        raise KeyError("adata.obs must contain an 'age' column.")
    adata.obs[AGE_KEY] = pd.to_numeric(adata.obs["age"], errors="coerce")
    adata.obs[AGE_GROUP_KEY] = pd.cut(
        adata.obs[AGE_KEY],
        bins=[-np.inf, 3, 6, 13, np.inf],
        right=False,
        labels=["Infant", "Early Childhood", "Late Childhood", "Adolescence"],
    )


def ensure_neighbors_umap(adata):
    if "neighbors" not in adata.uns:
        rep = "X_mrvi_u" if "X_mrvi_u" in adata.obsm else "X_mrvi_z"
        if rep not in adata.obsm:
            rep = None
        sc.pp.neighbors(adata, use_rep=rep, random_state=0)
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata, random_state=0)


def ensure_leiden_res03(adata):
    ensure_neighbors_umap(adata)
    if LEIDEN_KEY not in adata.obs:
        sc.tl.leiden(
            adata,
            resolution=RESOLUTION,
            key_added=LEIDEN_KEY,
            flavor="igraph",
            n_iterations=2,
            directed=False,
            random_state=0,
        )
    adata.obs[LEIDEN_KEY] = adata.obs[LEIDEN_KEY].astype(str)
    return adata


def annotate_res03(adata):
    ensure_leiden_res03(adata)
    ann = adata.obs[LEIDEN_KEY].map(RES03_CLUSTER_TO_SUBTYPE).fillna("Unassigned")
    categories = [x for x in SUBTYPE_ORDER if x in set(ann)]
    if "Unassigned" in set(ann):
        categories.append("Unassigned")
    adata.obs[ANNOTATION_KEY] = pd.Categorical(ann, categories=categories, ordered=True)
    return adata


def present_genes(adata, genes):
    var_names = pd.Index(adata.var_names.astype(str))
    return [gene for gene in genes if gene in var_names]


def flattened_feature_genes(adata):
    genes = []
    for subtype in SUBTYPE_ORDER:
        for gene in SUBTYPE_FEATURE_GENES[subtype]:
            if gene not in genes:
                genes.append(gene)
    return present_genes(adata, genes)


def get_sample_key(adata):
    for key in ["patient", "patient_id", "sample", "sample_id", "donor", "donor_id", "orig.ident", "batch"]:
        if key in adata.obs:
            return key
    adata.obs["_sample_id"] = adata.obs_names.astype(str)
    return "_sample_id"


def sample_metadata(adata):
    sample_key = get_sample_key(adata)
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


def sample_subtype_fractions(adata, subtype_key=ANNOTATION_KEY):
    sample_key = get_sample_key(adata)
    df = adata.obs[[sample_key, subtype_key, AGE_KEY, AGE_GROUP_KEY]].dropna(subset=[sample_key, subtype_key, AGE_KEY]).copy()
    counts = df.groupby([sample_key, subtype_key], observed=True).size().rename("n").reset_index()
    totals = counts.groupby(sample_key, observed=True)["n"].transform("sum")
    counts["fraction"] = counts["n"] / totals
    grid = pd.MultiIndex.from_product(
        [df[sample_key].dropna().unique(), SUBTYPE_ORDER],
        names=[sample_key, subtype_key],
    )
    counts = counts.set_index([sample_key, subtype_key]).reindex(grid, fill_value=0).reset_index()
    counts["total_cells"] = counts.groupby(sample_key, observed=True)["n"].transform("sum")
    meta = sample_metadata(adata)
    counts = counts.merge(meta, left_on=sample_key, right_on="sample", how="left")
    return counts.rename(columns={sample_key: "sample", subtype_key: "subtype"})


def expression_matrix(adata, genes, layer=None):
    genes = present_genes(adata, genes)
    if not genes:
        return pd.DataFrame(index=adata.obs_names)
    matrix = adata[:, genes].layers[layer] if layer else adata[:, genes].X
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return pd.DataFrame(np.asarray(matrix), index=adata.obs_names, columns=genes)


def average_expression_by_group(adata, genes, group_key, layer=None):
    genes = present_genes(adata, genes)
    expr = expression_matrix(adata, genes, layer=layer)
    group = adata.obs[group_key].astype(str)
    rows = []
    for name in list(pd.Categorical(group).categories) if hasattr(group.dtype, "categories") else sorted(group.unique()):
        idx = np.asarray(group == name)
        if idx.sum() == 0:
            continue
        avg = expr.loc[idx, genes].mean(axis=0)
        pct = (expr.loc[idx, genes] > 0).mean(axis=0) * 100
        for gene in genes:
            rows.append({"group": name, "gene": gene, "avg": avg[gene], "pct": pct[gene]})
    return pd.DataFrame(rows)


def module_score(adata, genes, layer=None):
    genes = present_genes(adata, genes)
    if not genes:
        return pd.Series(np.nan, index=adata.obs_names)
    expr = expression_matrix(adata, genes, layer=layer)
    return expr.mean(axis=1)


def bh_fdr(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    out = np.full_like(pvalues, np.nan, dtype=float)
    mask = np.isfinite(pvalues)
    p = pvalues[mask]
    if len(p) == 0:
        return out
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[mask] = tmp
    return out


def spearman_table(df, value_col, group_cols, age_col=AGE_KEY):
    rows = []
    for keys, sub in df.dropna(subset=[value_col, age_col]).groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if sub[age_col].nunique() < 3 or sub[value_col].nunique() < 2:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(sub[age_col], sub[value_col])
        row = dict(zip(group_cols, keys))
        row.update({"rho": rho, "pval": pval, "n": len(sub)})
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["qval"] = bh_fdr(out["pval"].values)
    return out


def subtype_palette():
    return {
        "MME+": "#4C78A8",
        "LUM+": "#F58518",
        "CD55+": "#54A24B",
        "GPC3+": "#B279A2",
        "COL11A1+": "#E45756",
        "ACTA1+": "#72B7B2",
        "Unassigned": "#999999",
    }


def plot_heatmap(matrix, path, cmap="RdBu_r", center=0, vmin=None, vmax=None, cbar_label=""):
    configure_plotting()
    matrix = pd.DataFrame(matrix)
    height = max(2.6, 0.32 * matrix.shape[0] + 1.2)
    width = max(4.5, 0.28 * matrix.shape[1] + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    values = matrix.values.astype(float)
    if vmin is None or vmax is None:
        finite = values[np.isfinite(values)]
        if len(finite):
            lim = np.nanpercentile(np.abs(finite), 95)
            vmin = -lim if vmin is None else vmin
            vmax = lim if vmax is None else vmax
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)
    savefig(path, fig)


def log1p_normalize_if_needed(adata):
    if sparse.issparse(adata.X):
        max_value = float(adata.X.data.max()) if adata.X.data.size else 0
    else:
        max_value = float(np.nanmax(adata.X))
    if max_value > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata
