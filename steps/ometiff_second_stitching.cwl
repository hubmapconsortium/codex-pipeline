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
  data_dir:
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
  ome_tiff_creation:
    in:
      cytokit_output:
        source: cytokit_output
      cytokit_config:
        source: cytokit_config
      input_data_dir:
        source: data_dir
    out:
      - ome_tiffs
    run: ometiff_second_stitching/ome_tiff_creation.cwl
    label: "Create OME-TIFF versions of Cytokit segmentation and extract results"

  second_stitching:
    in:
      pipeline_config:
        source: slicing_pipeline_config
      ometiff_dir:
        source: ome_tiff_creation/ome_tiffs
    out:
       - stitched_images
       - final_pipeline_config
    run: ometiff_second_stitching/second_stitching.cwl
