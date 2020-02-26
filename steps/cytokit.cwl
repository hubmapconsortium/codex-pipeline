cwlVersion: v1.1
class: CommandLineTool
label: CODEX analysis via Cytokit
hints:
  DockerRequirement:
    dockerPull: hubmap/cytokit:latest
  DockerGpuRequirement: {}
baseCommand: /opt/cytokit_wrapper.py

inputs:
  cytokit_command:
    type: string
    inputBinding:
      position: 1
  data_dir:
    type: Directory
    inputBinding:
      position: 2
  pipeline_config:
    type: File
    inputBinding:
      position: 3
  yaml_config:
    type: File
    inputBinding:
      position: 4
outputs:
  cytokit_output:
    type: Directory
    outputBinding:
      glob: output
