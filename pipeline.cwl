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
    label: "GPUs to use, comma-separated list of integers or all"
    type: string
    default: "all"
  segmentation_method:
    label: "Which segmentation method to use deepcell or cellpose"
    type: string
    default: "deepcell"


outputs:
  pipeline_output:
    outputSource: collect_output/pipeline_output
    type: Directory
    label: "Expression data and segmentation masks"
  pipeline_config:
    outputSource: image_processing/pipeline_config
    type: File
    label: "Pipeline config with all the modifications"

steps:
  image_processing:
    in:
      data_dir:
        source: data_dir
      gpus:
        source: gpus
    out:
      - pipeline_config
      - expression_channels
      - segmentation_channels
    run: steps/image_processing.cwl
    label: "Illumination correction, best focus selection, and stitching"

  run_segmentation:
    in:
      method:
        source: segmentation_method
      dataset_dir:
        source: image_processing/segmentation_channels
      gpus:
        source: gpus
    out:
      - segmentation_masks
    run: steps/run_segmentation.cwl
    label: "Nucleus and cell segmentation"

  collect_output:
    in:
      pipeline_config:
        source: image_processing/pipeline_config
      expression_channels:
        source: image_processing/expression_channels
      segmentation_masks:
        source: run_segmentation/segmentation_masks
    out:
      - pipeline_output
    run: steps/collect_output.cwl
    label: "Gathering data together in one directory"
