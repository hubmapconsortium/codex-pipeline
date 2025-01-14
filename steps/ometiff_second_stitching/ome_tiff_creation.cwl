cwlVersion: v1.1
class: CommandLineTool
label: Create OME-TIFF versions of Cytokit segmentation and extract results

requirements:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts:2.7.0

baseCommand: ["python", "/opt/convert_to_ometiff.py"]

inputs:
  cytokit_output:
    type: Directory
    inputBinding:
      position: 0
  bg_sub_tiles:
    type: Directory
    inputBinding:
      position: 1
  cytokit_config:
    type: File
    inputBinding:
      position: 2
  input_data_dir:
    type: Directory
    inputBinding:
        position: 3

outputs:
  ome_tiffs:
    type: Directory
    outputBinding:
      glob: output


