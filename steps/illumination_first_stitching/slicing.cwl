cwlVersion: v1.1
class: CommandLineTool

requirements:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts:2.1.3
    dockerOutputDirectory: "/output"

baseCommand: ["python", "/opt/slicing/run_slicing.py"]


inputs:
  base_stitched_dir:
    type: Directory
    inputBinding:
      prefix: "--base_stitched_dir"

  pipeline_config:
    type: File
    inputBinding:
      prefix: "--pipeline_config_path"

outputs:
  new_tiles:
    type: Directory
    outputBinding:
      glob: "/output/new_tiles"

  modified_pipeline_config:
    type: File
    outputBinding:
      glob: "/output/pipeline_conf/pipelineConfig.json"
