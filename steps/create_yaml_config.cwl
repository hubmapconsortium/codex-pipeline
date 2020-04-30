cwlVersion: v1.1
class: CommandLineTool
label: Create Cytokit experiment config
hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts:20200430-193501
baseCommand: /opt/create_cytokit_config.py

inputs:
  gpus:
    type: string
    inputBinding:
      position: 1
      prefix: "--gpus="
      separate: false
  pipeline_config:
    type: File
    inputBinding:
      position: 2
outputs:
  cytokit_config:
    type: File
    outputBinding:
      glob: experiment.yaml
