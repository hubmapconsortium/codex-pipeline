cwlVersion: v1.1
class: CommandLineTool
label: Create directory containing symlinks to relevant files for visualization team
hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts:1.6.2
baseCommand: /opt/create_visualization_dir.py

inputs:
  cytokit_config_file:
    type: File
    inputBinding:
      position: 1
  ometiff_dir:
    type: Directory
    inputBinding:
      position: 2
  sprm_output:
    type: Directory
    inputBinding:
      position: 3
outputs:
  for_viz_dir:
    type: File
    outputBinding:
      glob: symlinks.tar
