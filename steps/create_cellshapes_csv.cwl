cwlVersion: v1.1
class: CommandLineTool
label: Create CSVs containing Cytokit cytometry information and cell shape polygons
hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts
baseCommand: /opt/create_cellshapes_csv.py

inputs:
  ome_tiffs:
    type: Directory
    inputBinding:
      position: 1
  cytokit_processor_output:
    type: Directory
    inputBinding:
      position: 2
  cytokit_operator_output:
    type: Directory
    inputBinding:
      position: 3
outputs:
  cell_shapes_csv:
    type: Directory
    outputBinding:
      glob: output
