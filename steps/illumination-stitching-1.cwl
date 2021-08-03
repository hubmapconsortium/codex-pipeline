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
  cytokit_config:
    outputSource: create_yaml_config/cytokit_config
    type: File
    label: "Cytokit configuration in YAML format"
  new_tiles:
    outputSource: slicing/new_tiles
    type: Directory
  slicing_pipeline_config:
    outputSource: slicing/modified_pipeline_config
    type: File
    label: "Pipeline config with all the modifications"

steps:
  collect_dataset_info:
    in:
      base_directory:
        source: data_dir
    out:
      - pipeline_config
    run: illumination_stitching_1/collect_dataset_info.cwl
    label: "Collect CODEX dataset info"

  illumination_correction:
    in:
      base_directory:
        source: data_dir
      pipeline_config:
        source: collect_dataset_info/pipeline_config
    out:
      - illum_corrected_tiles
    run: illumination_stitching_1/illumination_correction.cwl

  best_focus:
    in:
      data_dir:
        source: illumination_correction/illum_corrected_tiles
      pipeline_config:
        source: collect_dataset_info/pipeline_config
    out:
      - best_focus_tiles
    run: illumination_stitching_1/best_focus.cwl

  first_stitching:
    in:
      data_dir:
        source: best_focus/best_focus_tiles
      pipeline_config:
        source: collect_dataset_info/pipeline_config
    out:
       - stitched_images
    run: illumination_stitching_1/first_stitching.cwl

  slicing:
    in:
      base_stitched_dir:
        source: first_stitching/stitched_images
      pipeline_config:
        source: collect_dataset_info/pipeline_config
    out:
       - new_tiles
       - modified_pipeline_config
    run: illumination_stitching_1/slicing.cwl

  create_yaml_config:
    in:
      pipeline_config:
        source: slicing/modified_pipeline_config
      gpus:
        source: gpus
    out:
      - cytokit_config
    run: illumination_stitching_1/create_yaml_config.cwl
    label: "Create Cytokit experiment config in YAML format"
