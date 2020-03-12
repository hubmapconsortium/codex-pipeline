cwlVersion: v1.1
class: CommandLineTool
label: Create CSVs containing Cytokit cytometry information and cell shape polygons
hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts
baseCommand: /opt/create_cellshapes_csv.py

inputs:
  cytokit_output_dir:
    type: Directory
    inputBinding:
      position: 1
outputs:
  cytokit_output:
    type: Directory
    outputBinding:
      glob: output
