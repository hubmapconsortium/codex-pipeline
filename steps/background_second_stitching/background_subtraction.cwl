cwlVersion: v1.1
class: CommandLineTool

requirements:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts:2.1.2
    dockerOutputDirectory: "/output"

baseCommand: ["python", "/opt/background_subtraction/run_background_subtraction.py"]


inputs:
  cytokit_output:
    type: Directory
    inputBinding:
      prefix: "--data_dir"


  pipeline_config:
    type: File
    inputBinding:
      prefix: "--pipeline_config_path"

  cytokit_config:
    type: File
    inputBinding:
      prefix: "--cytokit_config_path"

outputs:
  bg_sub_tiles:
    type: Directory
    outputBinding:
      glob: "/output/background_subtraction"
