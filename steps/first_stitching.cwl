cwlVersion: v1.1
class: CommandLineTool

hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts
    dockerOutputDirectory: "/output"
  NetworkAccess:
    networkAccess: true
baseCommand: ["python", "/opt/codex_stitching/run_stitching.py"]


inputs:
  data_dir:
    type: Directory
    inputBinding:
      prefix: "--data_dir"


  pipeline_config:
    type: File
    inputBinding:
      prefix: "--pipeline_config_path"

outputs:
  image_tiles:
    type: Directory
    outputBinding:
      glob: "/output/processed_images"

  modified_pipeline_config:
    type: File
    outputBinding:
      glob: "/output/pipeline_conf/pipelineConfig.json"

