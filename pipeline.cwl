#!/usr/bin/env cwl-runner

class: Workflow
cwlVersion: v1.1
label: CODEX analysis pipeline using Cytokit
requirements:
  StepInputExpressionRequirement: {}

inputs:
  data_dir:
    label: "Directory containing CODEX data"
    type: Directory

outputs:
  pipeline_config:
    outputSource: collect_dataset_info/pipeline_config
    type: File
    label: "Pipeline config"
  experiment_config:
    outputSource: collect_dataset_info/pipeline_config
    type: File
    label: "Cytokit configuration format"
  cytokit_processor_output:
    outputSource: cytokit_processor/cytokit_output
    type: Directory
    label: "Cytokit processor output"
  cytokit_operator_output:
    outputSource: cytokit_operator/cytokit_output
    type: Directory
    label: "Cytokit operator output"

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

  - id: cytokit_processor
    in:
      - id: cytokit_command
        valueFrom: "processor"
      - id: data_dir
        source: data_dir
      - id: pipeline_config
        source: collect_dataset_info/pipeline_config
      - id: yaml_config
        source: create_yaml_config/cytokit_config
    out:
      - cytokit_output
    run: steps/cytokit.cwl
    label: "CODEX analysis via Cytokit 'processor'"

  - id: cytokit_operator
    in:
      - id: cytokit_command
        valueFrom: "operator"
      - id: data_dir
        source: cytokit_processor/cytokit_output
      - id: pipeline_config
        source: collect_dataset_info/pipeline_config
      - id: yaml_config
        source: create_yaml_config/cytokit_config
    out:
      - cytokit_output
    run: steps/cytokit.cwl
    label: "CODEX analysis via Cytokit 'operator'"
