cwlVersion: v1.1
class: CommandLineTool
label: Collect segmentation masks and images for the final output

hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts
    dockerOutputDirectory: "/output"

baseCommand: ["python", "/opt/collect_output.py"]

inputs:
  pipeline_config:
    type: File
    inputBinding:
      prefix: "--pipeline_config"

  expression_channels:
    type: Directory
    inputBinding:
      prefix: "--expression_channels"

  segmentation_masks:
    type: Directory
    inputBinding:
      prefix: "--segmentation_masks"


outputs:
  pipeline_output:
    type: Directory
    outputBinding:
      glob: "/output/pipeline_output"
