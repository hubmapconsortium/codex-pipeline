#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.1
label: CODEX analysis pipeline using Cytokit

requirements:
  SubworkflowFeatureRequirement: {}

inputs:
  data_dir:
    label: "Directory containing CODEX data"
    type: Directory
  gpus:
    label: "GPUs to use, represented as a comma-separated list of integers"
    type: string
    default: "0"
  num_concurrent_tasks:
    label: "Number of parallel CPU jobs"
    type: int
    default: 10

outputs:
  experiment_config:
    outputSource: illumination_first_stitching/cytokit_config
    type: File
    label: "Cytokit configuration format"
  data_json:
    outputSource: run_cytokit/data_json
    type: File
    label: "JSON file containing Cytokit's calculations from deconvolution, drift compensation, and focal plane selection"
  stitched_images:
    outputSource: ometiff_second_stitching/stitched_images
    type: Directory
    label: "Segmentation masks and expressions in OME-TIFF format"
  pipeline_config:
    outputSource: ometiff_second_stitching/final_pipeline_config
    type: File
    label: "Pipeline config with all the modifications"

steps:
  illumination_first_stitching:
    in:
      data_dir:
        source: data_dir
      gpus:
        source: gpus
      num_concurrent_tasks:
        source: num_concurrent_tasks
    out:
      - slicing_pipeline_config
      - cytokit_config
      - new_tiles
    run: steps/illumination_first_stitching.cwl
    label: "Illumination correction, best focus selection, and stitching stage 1"

  run_cytokit:
    in:
      new_tiles:
        source: illumination_first_stitching/new_tiles
      yaml_config:
        source: illumination_first_stitching/cytokit_config
    out:
      - cytokit_output
      - data_json
    run: steps/run_cytokit.cwl
    label: "CODEX analysis via Cytokit processor and operator"

  ometiff_second_stitching:
    in:
      cytokit_output:
        source: run_cytokit/cytokit_output
      slicing_pipeline_config:
        source: illumination_first_stitching/slicing_pipeline_config
      cytokit_config:
        source: illumination_first_stitching/cytokit_config
      data_dir:
        source: data_dir
    out:
      - stitched_images
      - final_pipeline_config
    run: steps/ometiff_second_stitching.cwl
    label: "OMETIFF creation and stitching stage 2"
