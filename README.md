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
git add steps
git commit -m 'Update container timestamps for version v0.whatever'
git tag -a v0.whatever
git push
git push --tags
```
