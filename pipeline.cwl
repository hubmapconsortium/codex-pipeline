#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.0
label: CODEX analysis pipeline using Cytokit
inputs:
  data_dir:
    label: "Directory containing FASTQ files"
    type: Directory
  threads:
    label: "Number of threads for Salmon"
    type: int
    default: 1
outputs:
  count_matrix:
    outputSource: alevin_to_anndata/h5ad_file
    type: File
    label: "Count matrix from Alevin"
  qc_results:
    outputSource: qc_checks/qc_results
    type: File
    label: "Quality control metrics"
  umap_pdf:
    outputSource: dim_reduce_cluster/umap_pdf
    type: File
    label: "UMAP dimensionality reduction plot"
  dim_reduced_clustered:
    outputSource: dim_reduce_cluster/dim_reduced_clustered
    type: File
    label: "Dimensionality reduced and clustered data"
  cluster_marker_genes:
    outputSource: cluster_diffexpr/cluster_marker_genes
    type: File
    label: "Cluster marker genes"
  marker_gene_plot_t_test:
    outputSource: cluster_diffexpr/marker_gene_plot_t_test
    type: File
    label: "Cluster marker genes, t-test"
  marker_gene_plot_logreg:
    outputSource: cluster_diffexpr/marker_gene_plot_logreg
    type: File
    label: "Cluster marker genes, logreg method"
steps:
  - id: salmon
    in:
      - id: fastq_dir
        source: fastq_dir
      - id: threads
        source: threads
    out:
      - quant_mat
      - quant_mat_cols
      - quant_mat_rows
      - quant_tier_mat
    run: steps/salmon.cwl
    label: "Salmon Alevin 1.0.0, with index from GRCh38 transcriptome"
  - id: alevin_to_anndata
    in:
      - id: quant_mat
        source: salmon/quant_mat
      - id: quant_mat_cols
        source: salmon/quant_mat_cols
      - id: quant_mat_rows
        source: salmon/quant_mat_rows
    out:
      - h5ad_file
    run: steps/alevin-to-anndata.cwl
    label: "Convert Alevin output to AnnData object in h5ad format"
  - id: qc_checks
    in:
      - id: h5ad_file
        source: alevin_to_anndata/h5ad_file
    out:
      - qc_results
    run: steps/qc.cwl
    label: "Quality control checks"
  - id: filter_normalize
    in:
      - id: h5ad_file
        source: alevin_to_anndata/h5ad_file
    out:
      - filtered_normalized
    run: steps/filter-normalize.cwl
    label: "Filtering and normalization"
  - id: dim_reduce_cluster
    in:
      - id: h5ad_file
        source: filter_normalize/filtered_normalized
    out:
      - dim_reduced_clustered
      - umap_pdf
    run: steps/dim-reduction-clustering.cwl
    label: "Dimensionality reduction (UMAP) and clustering"
  - id: cluster_diffexpr
    in:
      - id: h5ad_file
        source: dim_reduce_cluster/dim_reduced_clustered
    out:
      - cluster_marker_genes
      - marker_gene_plot_t_test
      - marker_gene_plot_logreg
    run: steps/cluster-diffexpr.cwl
    label: "Compute differentially expressed genes between each cluster and rest"
