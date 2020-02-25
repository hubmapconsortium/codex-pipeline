cwlVersion: v1.1
class: CommandLineTool
label: Colect dataset info for Cytokit
hints:
  DockerRequirement:
    dockerPull: hubmapconsortium/codex-scripts
baseCommand: /opt/collect_dataset_info.py

inputs:
  base_directory:
    type: Directory
    inputBinding:
      position: 1
outputs:
  pipeline_config:
    type: File
    outputBinding:
      glob: pipelineConfig.json
