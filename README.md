# codex-pipeline
A [Nextflow](https://www.nextflow.io/index.html) pipeline for processing [CODEX](https://www.akoyabio.com/codextm/technology) image data, using [Cytokit](https://github.com/hammerlab/cytokit).

## Setting up the environment
1. Check out this repo.
2. Make a copy of `codex-pipeline/pipeline_env` somewhere and add appropriate values for environment variables listed therein.
3. `source` your new copy of `pipeline_env`.

## Locating data
Currently, manual inspection of submission directories is necessary to locate image data and appropriate config files containing required metadata. Once the locations of these are known, the script `codex-pipeline/bin/collect_dataset_info.py` can be used to create a JSON file serving as a config for the pipeline, containing the path to the image files, as well as dataset-specific microscope and cytometry parameters.

Run the script, depending on where all the necessary data is, e.g.
```
collect_dataset_info.py \
  HBM123.ABDC.456 \ 
  /path/to/image/data \ 
  /path/to/experiment.json \
  --segm-json /path/to/segmentation.json
```
...or
```
collect_dataset_info.py \
  HBM123.ABDC.456 \ 
  /path/to/image/data \ 
  /path/to/experiment.json \
  --segm-text /path/to/config.txt \
  --channel-names /path/to/channelNames.txt
```

* The first three arguments are required.
* `/path/to/image/data` should contain directories named according to `'^cyc0*(\d+)_reg0*(\d+).*'` (case insensitive), in turn containing TIFF files named according to `'^\d_\d{5}_Z\d{3}_CH\d\.tif$'` (case sensitive).
* One of `--segm-json` or `--segm-text` must be provided. 
* `--channel-names` is not needed if the channel names are included in `experiment.json`.

Run `collect_dataset_info.py` to create JSON pipeline configs for every dataset to be processed.

## Running the pipeline
1. Create a tab-separated file containing the dataset ID and location of the pipeline config created above, for each dataset to be processed, e.g.
```
HBM123.ABCD.456    pipeline_configs_dir/HBM123.ABCD.456_pipelineConfig.json
HBM234.EFGH.567    pipeline_configs_dir/HBM234.EFGH.567_pipelineConfig.json
```
2. Start the pipeline, either by running the command below at the commandline, or via Slurm.
```
nextflow $CODEX_PIPELINE_CODEBASE/main.nf -profile standard,slurm --datasetsFile tab_separated_datasets_list.txt
```

## Monitoring
The Nextflow log looks like e.g.
```
[f3/fe2c92] process > setup_data_directory (8)  [100%] 8 of 8 ✔
[4f/f34751] process > create_yaml_config (8)    [100%] 8 of 8 ✔
[da/d335eb] process > run_cytokit_processor (4) [ 75%] 3 of 4
```
The above may be in your Slurm logfile if you're running on Slurm.

You can also `tail` the Nextflow logfile of the process of interest, e.g. the following will show the log from `run_cytokit_processor`:
```
tail -f work/da/d335eb07637081e16ae3d18d444894/.command.log
```
Note: the location of the working directory is hinted at in the square brackets alongside each process in the Nextflow log, e.g. `[da/d335eb]` -- you can get the full path using e.g. `cd work/da/d335eb` and then tab-autocompletion.

## Results
Currently, the pipeline places all results under Nextflow's working directory (the Nextflow default). The results for the above process would be in `work/da/d335eb07637081e16ae3d18d444894`.

## Building containers
Two `Dockerfile`s are included in this repository. A `docker_images.txt` manifest is included, which is intended
for use in the `build_docker_containers` script provided by the
[`multi-docker-build`](https://github.com/mruffalo/multi-docker-build) Python package. This package can be installed
with
```shell script
python -m pip install multi-docker-build
```

## Release process

The `master` branch is intended to be production-ready at all times, and should always reference Docker containers
with the `latest` tag.

The `release` branch is used to publish a tagged version of this pipeline, referring to timestamp-tagged Docker
containers. To do this, given that `master` is up-to-date:
```shell script
git checkout release
git merge master
build_docker_containers --tag-latest --push
```
Update all timestamped tags in the CWL files in `steps` to refer to the new timestamped containers you just built.
*TODO*: automate this.

Then, tag a new version of the repository and push the tag and `release` branch. If you have a GPG key you can use
to sign tags, replace the `-a` option with `-s` below.
```shell script
git tag -a v0.whatever
git push
git push --tags
```
