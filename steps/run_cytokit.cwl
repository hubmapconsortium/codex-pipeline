#!/usr/bin/env cwl-runner

class: CommandLineTool
cwlVersion: v1.1
baseCommand: ["sh", "run_cytokit.sh"]

requirements:
  DockerRequirement:
    dockerPull: hubmap/cytokit:2.1.2
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

          mkdir $HOME/cytokit

          cytokit processor run_all --data-dir $(inputs.data_dir.path) --config-path $(inputs.yaml_config.path) --output_dir $HOME/cytokit && \
          cytokit operator run_all --data-dir $HOME/cytokit --config-path $(inputs.yaml_config.path) --output_dir $HOME/cytokit


inputs:
  data_dir:
    type: Directory

  yaml_config:
    type: File


outputs:
  cytokit_output:
    type: Directory
    outputBinding:
      glob: cytokit

  data_json:
    type: File
    outputBinding:
      glob: cytokit/processor/data.json


