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
    label: >-
      GPUs to use, represented as a comma-separated list of integers.
    type: string
    default: "0"

outputs:
  experiment_config:
    outputSource: illumination_stitching_1/cytokit_config
    type: File
    label: "Cytokit configuration format"
  data_json:
    outputSource: run_cytokit/data_json
    type: File
    label: "JSON file containing Cytokit's calculations from deconvolution, drift compensation, and focal plan selection"
  stitched_images:
    outputSource: background_stitching_2/stitched_images
    type: Directory
    label: "Segmentation masks and expressions in OME-TIFF format"
  pipeline_config:
    outputSource: background_stitching_2/final_pipeline_config
    type: File
    label: "Pipeline config with all the modifications"

steps:
  illumination_stitching_1:
    in:
      data_dir:
        source: data_dir
    out:
      - slicing_pipeline_config
      - cytokit_config
      - new_tiles
    run: steps/illumination-stitching-1.cwl
    label: "Illumination correction, best focus selection, and stitching stage 1"

  run_cytokit:
    in:
      data_dir:
        source: illumination_stitching_1/new_tiles
      yaml_config:
        source: illumination_stitching_1/cytokit_config
    out:
      - cytokit_output
      - data_json
    run: steps/run_cytokit.cwl
    label: "CODEX analysis via Cytokit processor and operator"

  background_stitching_2:
    in:
      cytokit_output:
        source: run_cytokit/cytokit_output
      slicing_pipeline_config:
        source: illumination_stitching_1/slicing_pipeline_config
      cytokit_config:
        source: illumination_stitching_1/cytokit_config
    out:
      - stitched_images
      - final_pipeline_config
    run: steps/background-stitching-2.cwl
    label: "Background subtraction and stitching stage 2"
