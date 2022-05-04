cwlVersion: v1.1
class: CommandLineTool
label: Prepare images for segmentation

hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts
    dockerOutputDirectory: "/output"

baseCommand: ["python", "/opt/prepare_segm_and_expr_channels.py"]

inputs:
  data_dir:
    type: Directory
    inputBinding:
      prefix: "--data_dir"

  pipeline_config:
    type: File
    inputBinding:
      prefix: "--pipeline_config"


outputs:
  segmentation_channels:
    type: Directory
    outputBinding:
      glob: "/output/segmentation_channels/"

  expression_channels:
    type: Directory
    outputBinding:
      glob: "/output/expression_channels/"
