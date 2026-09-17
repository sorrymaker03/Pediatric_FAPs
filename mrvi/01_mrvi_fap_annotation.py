from pathlib import Path

import scanpy as sc

from utils import (
    ANNOTATION_DIR,
    ANNOTATION_KEY,
    FEATUREPLOT_GENES,
    LEIDEN_KEY,
    RESOLUTION,
    SUBTYPE_FEATURE_GENES,
    SUBTYPE_ORDER,
    annotate_res03,
    configure_plotting,
    flattened_feature_genes,
    present_genes,
    read_adata,
    savefig,
    subtype_palette,
)


def plot_cluster_umap(adata, output_dir):
    sc.pl.umap(
        adata,
        color=LEIDEN_KEY,
        legend_loc="right margin",
        frameon=False,
        title="",
        show=False,
    )
    savefig(output_dir / f"cluster_umap_res{RESOLUTION}.pdf")


def plot_cluster_dotplot(adata, output_dir):
    genes = flattened_feature_genes(adata)
    sc.pl.dotplot(
        adata,
        var_names=genes,
        groupby=LEIDEN_KEY,
        standard_scale="var",
        color_map="RdBu_r",
        dendrogram=False,
        show=False,
    )
    savefig(output_dir / f"cluster_dotplot_res{RESOLUTION}.pdf")


def plot_merged_umap(adata, output_dir):
    sc.pl.umap(
        adata,
        color=ANNOTATION_KEY,
        palette=subtype_palette(),
        legend_loc="right margin",
        frameon=False,
        title="",
        show=False,
    )
    savefig(output_dir / f"merged_umap_res{RESOLUTION}.pdf")


def plot_merged_dotplot(adata, output_dir):
    gene_map = {
        subtype: present_genes(adata, genes)
        for subtype, genes in SUBTYPE_FEATURE_GENES.items()
        if subtype in adata.obs[ANNOTATION_KEY].cat.categories
    }
    sc.pl.dotplot(
        adata,
        var_names=gene_map,
        groupby=ANNOTATION_KEY,
        categories_order=SUBTYPE_ORDER,
        standard_scale="var",
        color_map="RdBu_r",
        dendrogram=False,
        swap_axes=False,
        show=False,
    )
    savefig(output_dir / f"merged_dotplot_res{RESOLUTION}.pdf")


def plot_featureplots(adata, output_dir):
    genes = present_genes(adata, FEATUREPLOT_GENES)
    for gene in genes:
        sc.pl.umap(
            adata,
            color=gene,
            cmap="viridis",
            frameon=False,
            title="",
            show=False,
        )
        savefig(output_dir / f"featureplot_{gene}.pdf")


def main():
    configure_plotting()
    output_dir = Path(ANNOTATION_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = read_adata()
    annotate_res03(adata)

    print(adata.obs[LEIDEN_KEY].value_counts().sort_index())
    print(adata.obs[ANNOTATION_KEY].value_counts().loc[SUBTYPE_ORDER])

    plot_cluster_umap(adata, output_dir)
    plot_cluster_dotplot(adata, output_dir)
    plot_merged_umap(adata, output_dir)
    plot_merged_dotplot(adata, output_dir)
    plot_featureplots(adata, output_dir)


if __name__ == "__main__":
    main()
