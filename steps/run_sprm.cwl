cwlVersion: v1.1
class: CommandLineTool
label: Run Spatial Process & Relationship Modeling (SPRM)
hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts:20200430-193501
  NetworkAccess:
    networkAccess: true
baseCommand: /opt/run_sprm.py

inputs:
  ometiff_dir:
    type: Directory
    inputBinding:
      position: 2
outputs:
  sprm_output_dir:
    type: Directory
    outputBinding:
      glob: sprm_outputs
