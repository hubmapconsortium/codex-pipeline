cwlVersion: v1.1
class: CommandLineTool
label: Run segmentation

hints:
  DockerRequirement:
    dockerPull: hubmap/segmentations:1.2.2
    dockerOutputDirectory: "/output"
  DockerGpuRequirement: {}
  NetworkAccess:
    networkAccess: true

baseCommand: ["python", "/opt/main.py"]

inputs:
  method:
    type: string
    inputBinding:
      prefix: "--method"

  dataset_dir:
    type: Directory
    inputBinding:
      prefix: "--dataset_dir"

  gpus:
    type: string
    inputBinding:
      prefix: "--gpus"

outputs:
  segmentation_masks:
    type: Directory
    outputBinding:
      glob: "/output"
