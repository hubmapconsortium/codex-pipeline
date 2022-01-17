cwlVersion: v1.1
class: CommandLineTool

requirements:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts
    dockerOutputDirectory: "/output"

baseCommand: ["python", "/opt/illumination_correction/run_illumination_correction.py"]


inputs:
  base_directory:
    type: Directory
    inputBinding:
      prefix: "--data_dir"

  converted_dataset:
    type: Directory
    inputBinding:
      prefix: "--converted_dataset"

  pipeline_config:
    type: File
    inputBinding:
      prefix: "--pipeline_config_path"

outputs:
  illum_corrected_tiles:
    type: Directory
    outputBinding:
      glob: "/output/corrected_images"
