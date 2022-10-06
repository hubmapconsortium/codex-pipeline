#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.1
label: CODEX analysis pipeline using Cytokit

inputs:
  slicing_pipeline_config:
    type: File
  cytokit_config:
    type: File
  cytokit_output:
    type: Directory

outputs:
  stitched_images:
    outputSource: second_stitching/stitched_images
    type: Directory
    label: "Segmentation masks and expressions in OME-TIFF format"
  final_pipeline_config:
    outputSource: second_stitching/final_pipeline_config
    type: File
    label: "Pipeline config with all the modifications"

steps:
  background_subtraction:
    in:
      cytokit_output:
        source: cytokit_output
      pipeline_config:
        source: slicing_pipeline_config
      cytokit_config:
        source: cytokit_config
    out:
      - bg_sub_tiles
      - bg_sub_config
    run: ometiff_second_stitching/background_subtraction.cwl

  ome_tiff_creation:
    in:
      cytokit_output:
        source: cytokit_output
      bg_sub_tiles:
          source: background_subtraction/bg_sub_tiles
      cytokit_config:
        source: cytokit_config
    out:
      - ome_tiffs
    run: ometiff_second_stitching/ome_tiff_creation.cwl
    label: "Create OME-TIFF versions of Cytokit segmentation and extract results"

  second_stitching:
    in:
      pipeline_config:
        source: background_subtraction/bg_sub_config
      ometiff_dir:
        source: ome_tiff_creation/ome_tiffs
    out:
       - stitched_images
       - final_pipeline_config
    run: ometiff_second_stitching/second_stitching.cwl
