[![Build Status](https://travis-ci.com/hubmapconsortium/codex-pipeline.svg?branch=master)](https://travis-ci.com/hubmapconsortium/codex-pipeline)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# codex-pipeline
A [CWL](https://www.commonwl.org/) pipeline for processing [CODEX](https://www.akoyabio.com/codextm/technology) image data, using [Cytokit](https://github.com/hammerlab/cytokit).

## Pipeline steps
* Collect required parameters from metadata files.
* Create Cytokit YAML config file containing parameters collected above.
* Run Cytokit's `processor` command to perform tile pre-processing, and nucleus and cell segmentation.
* Run Cytokit's `operator` command to extract all antigen fluoresence images (discarding blanks and empty channels).
* Generate [OME-TIFF](https://docs.openmicroscopy.org/ome-model/6.0.1/ome-tiff/specification.html) versions of TIFFs created by Cytokit.
* Perform downstream analysis using [SPRM](https://github.com/hubmapconsortium/sprm).

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
