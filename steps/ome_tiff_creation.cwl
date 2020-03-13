cwlVersion: v1.1
class: CommandLineTool
label: Create OME-TIFF versions of Cytokit segmentation and extract results
hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts
  NetworkAccess:
    networkAccess: true
baseCommand: /opt/convert_to_ometiff.py

inputs:
  cytokit_processor_output:
    type: Directory
    inputBinding:
      position: 1
  cytokit_operator_output:
    type: Directory
    inputBinding:
      position: 2
outputs:
  ome_tiffs:
    type: Directory
    outputBinding:
      glob: output
