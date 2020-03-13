cwlVersion: v1.1
class: CommandLineTool
label: Collect dataset info for Cytokit
hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts:20200313-124813
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
