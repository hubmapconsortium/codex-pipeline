#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.1
label: CODEX analysis pipeline using Cytokit

inputs:
  data_dir:
    label: "Directory containing CODEX data"
    type: Directory

outputs:
  pipeline_config:
    outputSource: collect_dataset_info/pipeline_config
    type: File
    label: "Pipeline config"
  segmentation_channels:
    type: Directory
    outputSource: prepare_segmentation_channels/segmentation_channels
    label: "Channels that will be used for nucleus and cell segmentation"
  expression_channels:
    type: Directory
    outputSource: prepare_segmentation_channels/expression_channels

steps:
  collect_dataset_info:
    in:
      base_directory:
        source: data_dir
    out:
      - pipeline_config
    run: image_processing/collect_dataset_info.cwl
    label: "Collect CODEX dataset info"

  illumination_correction:
    in:
      base_directory:
        source: data_dir
      pipeline_config:
        source: collect_dataset_info/pipeline_config
    out:
      - illum_corrected_tiles
    run: image_processing/illumination_correction.cwl

  best_focus:
    in:
      data_dir:
        source: illumination_correction/illum_corrected_tiles
      pipeline_config:
        source: collect_dataset_info/pipeline_config
    out:
      - best_focus_tiles
    run: image_processing/best_focus.cwl

  image_stitching:
    in:
      data_dir:
        source: best_focus/best_focus_tiles
      pipeline_config:
        source: collect_dataset_info/pipeline_config
    out:
       - stitched_images
    run: image_processing/image_stitching.cwl

  image_registration:
    in:
      data_dir:
        source: image_stitching/stitched_images
      pipeline_config:
        source: collect_dataset_info/pipeline_config
    out:
      - registered_images
    run: image_processing/image_registration.cwl

  prepare_segmentation_channels:
    in:
      data_dir:
        source: image_registration/registered_images
      pipeline_config:
        source: collect_dataset_info/pipeline_config
    out:
      - segmentation_channels
      - expression_channels
    run: image_processing/prepare_segm_and_expr_channels.cwl
