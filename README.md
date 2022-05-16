[![Build Status](https://travis-ci.com/hubmapconsortium/codex-pipeline.svg?branch=master)](https://travis-ci.com/hubmapconsortium/codex-pipeline)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# codex-pipeline
A [CWL](https://www.commonwl.org/) pipeline for processing [CODEX](https://www.akoyabio.com/codextm/technology) image data.

## Pipeline steps
* Collect required parameters from metadata files.
* Perform illumination correction with Fiji plugin [BaSiC](https://github.com/VasylVaskivskyi/BaSiC_Mod) 
* Find sharpest z-plane for each channel, using variation of Laplacian
* Perform stitching of tiles using Fiji plugin [BigStitcher](https://imagej.net/plugins/bigstitcher/)
* Align images across cycles using linear and non-linear registration
* Segment nuclei and cells using [Segmentations pipeline](https://github.com/hubmapconsortium/segmentations)
* Generate [OME-TIFF](https://docs.openmicroscopy.org/ome-model/6.0.1/ome-tiff/specification.html)
* Perform downstream analysis using [SPRM](https://github.com/hubmapconsortium/sprm).


## Requirements

Please use [HuBMAP Consortium fork of cwltool](https://github.com/hubmapconsortium/cwltool) 
to be able to run pipeline with GPU in Docker and Singularity containers.\
For the list of python packages check `environment.yml`.


## How to run

`cwltool pipeline.cwl subm.yaml`

If you use Singularity containers add `--singularity`. Example of submission file `subm.yaml` is provided in the repo.


## Expected input directory and file structure

```
codex_dataset/
  raw
    ├── dataset.json
    ├── channelnames.txt
    ├── channelnames_report.csv
    ├── experiment.json
    ├── exposure_times.txt
    ├── segmentation.json
    ├── Cyc001_reg001  
    │     ├── 1_00001_Z001_CH1.tif
    │     ├── 1_00001_Z001_CH2.tif
    │     │              ...
    │     └── 1_0000N_Z00N_CHN.tif
    └── Cyc001_reg002  
          ├── 2_00001_Z001_CH1.tif
          ├── 2_00001_Z001_CH2.tif
          │             ...
          └── 1_0000N_Z00N_CHN.tif

```

Images should be separated into directories by cycles and regions using the following pattern `Cyc{cycle:d}_reg{region:d}`.
The file names must contain region, tile, z-plane and channel ids starting from 1, and follow this pattern 
`{region:d}_{tile:05d}_Z{zplane:03d}_CH{channel:d}.tif`.

The acquisition metadata in either new or old format must be present in the `raw` directory:

### New metadata format
`dataset.json` - image acquisition parameters. For more details check [metadata repo](https://github.com/hubmapconsortium/codex-common-acquisition-metadata)

### Old metadata format
* `experiment.json` - acquisition parameters and data structure;
* `segmentation.json` - which channel from which cycle to use for segmentation;
* `channelnames.txt` - list of channel names, one per row;
* `channelnames_report.csv` - which channels to use, and which to exclude;
* `exposure_times.txt` - not used at the moment, but will be useful for background subtraction.

Examples of these files are present in the directory `metadata_examples`. 
Note: all fields related to regions, cycles, channels, z-planes and tiles start from 1, 
and xyResolution, zPitch are measured in `nm`.

## Output file structure

```
pipeline_output/
├── expr
│   ├── reg001_expr.ome.tiff
│   └── reg002_expr.ome.tiff
└── mask
    ├── reg001_mask.ome.tiff
    └── reg002_expr.ome.tiff
```

Where `expr` directory contains processed images and `mask` contains segmentation masks.
The output of SPRM will be different, see https://github.com/hubmapconsortium/sprm .


## Development
Code in this repository is formatted with [black](https://github.com/psf/black) and
[isort](https://pypi.org/project/isort/), and this is checked via Travis CI.

A [pre-commit](https://pre-commit.com/) hook configuration is provided, which runs `black` and `isort` before committing.
Run `pre-commit install` in each clone of this repository which you will use for development (after `pip install pre-commit`
into an appropriate Python environment, if necessary).

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

Publication of tagged "release" versions of the pipeline is handled with the
[HuBMAP pipeline release management](https://github.com/hubmapconsortium/pipeline-release-mgmt) Python package. To
release a new pipeline version, *ensure that the `master` branch contains all commits that you want to include in the release,*
then run
```shell
tag_releae_pipeline v0.whatever
```
See the pipeline release managment script usage notes for additional options, such as GPG signing.
