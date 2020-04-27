#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.1
label: CODEX analysis pipeline using Cytokit

inputs:
  data_dir:
    label: "Directory containing CODEX data"
    type: Directory
  gpus:
    label: >-
      GPUs to use, represented as a comma-separated list of integers.
    type: string
    default: "0"

outputs:
  pipeline_config:
    outputSource: collect_dataset_info/pipeline_config
    type: File
    label: "Pipeline config"
  experiment_config:
    outputSource: create_yaml_config/cytokit_config
    type: File
    label: "Cytokit configuration format"
  cytokit_processor_output:
    outputSource: cytokit_processor/cytokit_output
    type: Directory
    label: "Cytokit processor output"
  cytokit_operator_output:
    outputSource: cytokit_operator/cytokit_output
    type: Directory
    label: "Cytokit operator output"
  ome_tiff_output:
    outputSource: ome_tiff_creation/ome_tiffs
    type: Directory
    label: "Segmentation masks in OME-TIFF format"
  sprm_output_dir:
    outputSource: run_sprm/sprm_output_dir
    type: Directory
    label: "Directory containing all SPRM outputs"
  for_viz_dir:
    outputSource: create_dir_for_viz/for_viz_dir
    type: Directory
    label: "Symbolic links to files for visualization team"

steps:
  - id: collect_dataset_info
    in:
      - id: base_directory
        source: data_dir
    out:
      - pipeline_config
    run: steps/collect_dataset_info.cwl
    label: "Collect CODEX dataset info"

  - id: create_yaml_config
    in:
      - id: pipeline_config
        source: collect_dataset_info/pipeline_config
      - id: gpus
        source: gpus
    out:
      - cytokit_config
    run: steps/create_yaml_config.cwl
    label: "Create Cytokit experiment config"

  - id: cytokit_processor
    in:
      - id: data_dir
        source: data_dir
      - id: pipeline_config
        source: collect_dataset_info/pipeline_config
      - id: yaml_config
        source: create_yaml_config/cytokit_config
    out:
      - cytokit_output
    run: steps/cytokit_processor.cwl
    label: "CODEX analysis via Cytokit 'processor'"

  - id: cytokit_operator
    in:
      - id: data_dir
        source: cytokit_processor/cytokit_output
      - id: pipeline_config
        source: collect_dataset_info/pipeline_config
      - id: yaml_config
        source: create_yaml_config/cytokit_config
    out:
      - cytokit_output
    run: steps/cytokit_operator.cwl
    label: "CODEX analysis via Cytokit 'operator'"

  - id: ome_tiff_creation
    in:
      - id: cytokit_processor_output
        source: cytokit_processor/cytokit_output
      - id: cytokit_operator_output
        source: cytokit_operator/cytokit_output
    out:
      - expressions_ometiff_dir
      - cytometry_ometiff_dir
    run: steps/ome_tiff_creation.cwl
    label: "Create OME-TIFF versions of Cytokit segmentation and extract results"

  - id: run_sprm
    in:
      - id: expressions_ometiff_dir
        source: ome_tiff_creation/expressions_ometiff_dir
      - id: cytometry_ometiff_dir
        source: ome_tiff_creation/cytometry_ometiff_dir
    out:
      - sprm_output_dir
    run: steps/run_sprm.cwl
    label: "Run SPRM analysis of OME-TIFF files"

  - id: create_dir_for_viz
    in:
      - id: cytokit_config_file
        source: create_yaml_config/cytokit_config
      - id: expressions_ometiff_dir
        source: ome_tiff_creation/expressions_ometiff_dir
      - id: cytometry_ometiff_dir
        source: ome_tiff_creation/cytometry_ometiff_dir
      - id: sprm_output
        source: run_sprm/sprm_output_dir
    out:
      - for_viz_dir
    run: steps/create_dir_for_viz.cwl
    label: "Create directory containing symlinks to relevant files for visualization team"
