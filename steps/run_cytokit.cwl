#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.1
baseCommand: ["sh", "run_cytokit.sh"]

hints:
  DockerRequirement:
    dockerPull: hubmap/cytokit:2.0b4
    dockerOutputDirectory: "/lab/cytokit_output"
  DockerGpuRequirement: {}

  InitialWorkDirRequirement:
    listing:
      - entryname: run_cytokit.sh
        entry: |-
          __conda_setup="\$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
          if [ \$? -eq 0 ]; then
             eval "\$__conda_setup"
          else
             if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
                 . "/opt/conda/etc/profile.d/conda.sh"
             else
                 export PATH="/opt/conda/bin:$PATH"
             fi
          fi
          unset __conda_setup

          export PYTHONPATH=/lab/repos/cytokit/python/pipeline
          conda activate cytokit


          cytokit processor run_all --data-dir $(inputs.data_dir.path) --config-path $(inputs.yaml_config.path) --output-dir /lab/cytokit_output && \
          cytokit operator run_all --data-dir /lab/cytokit_output --config-path $(inputs.yaml_config.path) --output-dir /lab/cytokit_output


inputs:
  data_dir:
    type: Directory

  yaml_config:
    type: File


outputs:
  cytokit_output:
    type: Directory
    outputBinding:
      glob: /lab/cytokit_output

  data_json:
    type: File
    outputBinding:
      glob: /lab/cytokit_output/processor/data.json


