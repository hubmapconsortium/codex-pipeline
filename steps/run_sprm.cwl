cwlVersion: v1.1
class: CommandLineTool
label: Run Spatial Process & Relationship Modeling (SPRM)
hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts
  NetworkAccess:
    networkAccess: true
baseCommand: /opt/run_sprm.py

inputs:
  sprm_dir:
    type: Directory
    inputBinding:
      position: 1
  expressions_ometiff_dir:
    type: Directory
    inputBinding:
      position: 2
  cytometry_ometiff_dir:
    type: Directory
    inputBinding:
      position: 3
outputs:
  sprm_output_dir:
    type: Directory
    outputBinding:
      glob: sprm_outputs/results
