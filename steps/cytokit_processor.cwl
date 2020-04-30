cwlVersion: v1.1
class: CommandLineTool
label: CODEX analysis via Cytokit
hints:
  DockerRequirement:
    dockerPull: hubmap/cytokit:20200430-193501
  DockerGpuRequirement: {}
baseCommand: /opt/cytokit_wrapper.py

arguments:
  - "processor"
inputs:
  data_dir:
    type: Directory
    inputBinding:
      position: 1
  pipeline_config:
    type: File
    inputBinding:
      position: 2
  yaml_config:
    type: File
    inputBinding:
      position: 3
outputs:
  cytokit_output:
    type: Directory
    outputBinding:
      glob: output
