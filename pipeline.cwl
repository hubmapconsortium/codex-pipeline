#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.1
label: CODEX analysis pipeline using Cytokit
inputs:
  data_dir:
    label: "Directory containing CODEX data"
    type: Directory
outputs:
  cytokit_output:
    outputSource: cytokit/cytokit_output
    type: Directory
    label: "Cytokit output"
steps:
  - id: collect_dataset_info
    in:
      - id: base_directory
        source: data_dir
    out:
      - pipeline_config
    run: steps/collect_dataset_info.cwl
    label: "Collect CODEX dataset info"

  - id: create_yaml_config
    in:
      - id: pipeline_config
        source: collect_dataset_info/pipeline_config
    out:
      - cytokit_config
    run: steps/create_yaml_config.cwl
    label: "Create Cytokit experiment config"

  - id: cytokit
    in:
      - id: data_dir
        source: data_dir
      - id: pipeline_config
        source: collect_dataset_info/pipeline_config
      - id: yaml_config
        source: create_yaml_config/cytokit_config
    out:
      - cytokit_output
    run: steps/cytokit.cwl
    label: "CODEX analysis via Cytokit"
