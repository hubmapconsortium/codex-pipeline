cwlVersion: v1.1
class: CommandLineTool

requirements:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts
    dockerOutputDirectory: "/output"

baseCommand: ["python", "/opt/codex_stitching/run_stitching.py"]


inputs:
  data_dir:
    type: Directory
    inputBinding:
      prefix: "--data_dir"


  pipeline_config:
    type: File
    inputBinding:
      prefix: "--pipeline_config_path"

outputs:
  stitched_images:
    type: Directory
    outputBinding:
      glob: "/output/stitched_images"
