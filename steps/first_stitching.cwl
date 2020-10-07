cwlVersion: v1.1
class: CommandLineTool

hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts
  NetworkAccess:
    networkAccess: true
baseCommand: /opt/codex_stitching/stitch.py


inputs:
  pipeline_config_path:
    type: File
  inputBinding:
    prefix: "--pipeline_config_path"
  
outputs:
  image_tiles:
    type: Directory
    outputBinding:
      glob: output
   
   modified_pipeline_config
    type: File
    outputBinding:
        glob: ./output/pipelineConfig.json
