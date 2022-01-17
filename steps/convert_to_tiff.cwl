cwlVersion: v1.1
class: CommandLineTool

requirements:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts
    dockerOutputDirectory: "/output"

baseCommand: ["python", "/opt/file_conversion/run_conversion.py"]


inputs:
  data_dir:
    type: Directory
    inputBinding:
      prefix: "--data_dir"

outputs:
  converted_dataset:
    type: Directory
    outputBinding:
      glob: "/output/converted_dataset"

