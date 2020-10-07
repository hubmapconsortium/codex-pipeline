cwlVersion: v1.1
class: CommandLineTool

hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts
  NetworkAccess:
    networkAccess: true
baseCommand: /opt/codex_stitching/secondary_stitcher_runner.py


inputs: 
  pipeline_config_path:
    type: File
  inputBinding:
    prefix: "--pipeline_config_path"
  
  mask_tiles:
    type: Directory
  inputBinding:
    prefix: "--path_to_mask_tiles"
   
  codex_tiles:
     type: Directory
  inputBinding:
    prefix: "--path_to_mask_tiles"
  
outputs:
  stitched_images:
    type: Directory
    outputBinding:
      glob: output
   
