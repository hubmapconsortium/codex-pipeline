cwlVersion: v1.1
class: CommandLineTool

requirements:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts:2.2.1
    dockerOutputDirectory: "/output"

baseCommand: ["python", "/opt/best_focus/run_best_focus_selection.py"]


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
  best_focus_tiles:
    type: Directory
    outputBinding:
      glob: "/output/best_focus"

