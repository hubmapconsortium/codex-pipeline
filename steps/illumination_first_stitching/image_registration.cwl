cwlVersion: v1.1
class: CommandLineTool

requirements:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts
    dockerOutputDirectory: "/output"

baseCommand: ["python", "/opt/image_registration/registration_runner.py"]


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
  registered_images:
    type: Directory
    outputBinding:
      glob: "/output/registered_images"
