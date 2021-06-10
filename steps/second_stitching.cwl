cwlVersion: v1.1
class: CommandLineTool

hints:
  DockerRequirement:
    dockerPull: hubmap/codex-scripts:2.0b8
    dockerOutputDirectory: /output
  NetworkAccess:
    networkAccess: true
baseCommand: ["python", "/opt/codex_stitching/secondary_stitcher/secondary_stitcher_runner.py"]


inputs:
  pipeline_config:
    type: File
    inputBinding:
      prefix: "--pipeline_config_path"

  ometiff_dir:
    type: Directory
    inputBinding:
      prefix: "--ometiff_dir"

outputs:
  stitched_images:
    type: Directory
    outputBinding:
      glob: /output/stitched

  final_pipeline_config:
    type: File
    outputBinding:
      glob: /output/pipelineConfig.json
