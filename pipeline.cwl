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
  pipeline_config:
    outputSource: second_stitching/final_pipeline_config
    type: File
    label: "Pipeline config with all the modifications"

steps:
  collect_dataset_info:
    in:
      base_directory:
        source: data_dir
    out:
      - pipeline_config
    run: steps/collect_dataset_info.cwl
    label: "Collect CODEX dataset info"

  illumination_correction:
    in:
      base_directory:
        source: data_dir
      pipeline_config:
        source: collect_dataset_info/pipeline_config
    out:
      - illum_corrected_tiles
    run: steps/illumination_correction.cwl

  best_focus:
    in:
      data_dir:
        source: illumination_correction/illum_corrected_tiles
      pipeline_config:
        source: collect_dataset_info/pipeline_config
    out:
      - best_focus_tiles
    run: steps/best_focus.cwl

  first_stitching:
    in:
      data_dir:
        source: best_focus/best_focus_tiles
      pipeline_config:
        source: collect_dataset_info/pipeline_config
    out:
       - modified_pipeline_config
       - image_tiles
    run: steps/first_stitching.cwl

  create_yaml_config:
    in:
      pipeline_config:
        source: first_stitching/modified_pipeline_config
      gpus:
        source: gpus
    out:
      - cytokit_config
    run: steps/create_yaml_config.cwl
    label: "Create Cytokit experiment config"

  run_cytokit:
    in:
      data_dir:
        source: first_stitching/image_tiles
      yaml_config:
        source: create_yaml_config/cytokit_config
    out:
      - cytokit_output
      - data_json
    run: steps/run_cytokit.cwl
    label: "CODEX analysis via Cytokit processor and operator"


  ome_tiff_creation:
    in:
      cytokit_output:
        source: run_cytokit/cytokit_output
      cytokit_config:
        source: create_yaml_config/cytokit_config
    out:
      - ome_tiffs
    run: steps/ome_tiff_creation.cwl
    label: "Create OME-TIFF versions of Cytokit segmentation and extract results"

  second_stitching:
    in:
      pipeline_config:
        source: first_stitching/modified_pipeline_config
      ometiff_dir:
        source: ome_tiff_creation/ome_tiffs
    out:
       - stitched_images
       - final_pipeline_config
    run: steps/second_stitching.cwl
