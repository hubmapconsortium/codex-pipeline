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
  experiment_config:
    outputSource: create_yaml_config/cytokit_config
    type: File
    label: "Cytokit configuration format"
  data_json:
    outputSource: run_cytokit/data_json
    type: File
    label: "JSON file containing Cytokit's calculations from deconvolution, drift compensation, and focal plan selection"
  stitched_images:
    outputSource: second_stitching/stitched_images
    type: Directory
    label: "Segmentation masks and expressions in OME-TIFF format"
  sprm_output_dir:
    outputSource: run_sprm/sprm_output_dir
    type: Directory
    label: "Directory containing all SPRM outputs"
  for_viz_dir:
    outputSource: create_dir_for_viz/for_viz_dir
    type: File
    label: "Archive of symbolic links to files for visualization team"

steps:
  - id: collect_dataset_info
    in:
      - id: base_directory
        source: data_dir
    out:
      - pipeline_config
    run: steps/collect_dataset_info.cwl
    label: "Collect CODEX dataset info"
    
  - id: first_stitching
    in:
      - id: data_dir
        source: data_dir
      - id: pipeline_config
        source: collect_dataset_info/pipeline_config
    out:
       - modified_pipeline_config
       - image_tiles
    run: steps/first_stitching.cwl

  - id: create_yaml_config
    in:
      - id: pipeline_config
        source: first_stitching/modified_pipeline_config
      - id: gpus
        source: gpus
    out:
      - cytokit_config
    run: steps/create_yaml_config.cwl
    label: "Create Cytokit experiment config"
    
  - id: run_cytokit
    in:
      - id: data_dir
        source: first_stitching/image_tiles
      - id: yaml_config
        source: create_yaml_config/cytokit_config
    out:
      - cytokit_output
      - data_json
    run: steps/run_cytokit.cwl
    label: "CODEX analysis via Cytokit processor and operator"


  - id: ome_tiff_creation
    in:
      - id: cytokit_output
        source: run_cytokit/cytokit_output
      - id: cytokit_config
        source: create_yaml_config/cytokit_config
    out:
      - ome_tiffs
    run: steps/ome_tiff_creation.cwl
    label: "Create OME-TIFF versions of Cytokit segmentation and extract results"
    
  - id: second_stitching
    in:
      - id: pipeline_config
        source: first_stitching/modified_pipeline_config
      - id: ometiff_dir
        source: ome_tiff_creation/ome_tiffs
    out: 
       - stitched_images
    run: steps/second_stitching.cwl
    

  - id: run_sprm
    in:
      - id: ometiff_dir
        source: second_stitching/stitched_images
    out:
      - sprm_output_dir
    run: steps/run_sprm.cwl
    label: "Run SPRM analysis of OME-TIFF files"

  - id: create_dir_for_viz
    in:
      - id: cytokit_config_file
        source: create_yaml_config/cytokit_config
      - id: ometiff_dir
        source: second_stitching/stitched_images
      - id: sprm_output
        source: run_sprm/sprm_output_dir
    out:
      - for_viz_dir
    run: steps/create_dir_for_viz.cwl
    label: "Create directory containing symlinks to relevant files for visualization team"
