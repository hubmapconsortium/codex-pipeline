cwlVersion: v1.1
class: CommandLineTool
label: Colect dataset info for Cytokit
hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts
baseCommand: /opt/create_cytokit_config.py

inputs:
  pipeline_config:
    type: File
    inputBinding:
      position: 1
outputs:
  cytokit_config:
    type: File
    outputBinding:
      glob: experiment.yaml
