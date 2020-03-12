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
  cytokit_output_dir:
    type: Directory
    inputBinding:
      position: 1
outputs:
  cytokit_output:
    type: Directory
    outputBinding:
      glob: output
