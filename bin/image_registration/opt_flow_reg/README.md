[![DOI](https://zenodo.org/badge/405296622.svg)](https://zenodo.org/badge/latestdoi/405296622)
## Optical flow based registration for immunofluorescence images

These scripts perform fine registration using warping. 
A map for warping is calculated using Farneback optical flow algorithm, by OpenCV.
Although images are MINMAX normalized during processing, optical flow algorithms expect images to have 
similar pixel intensities. 

Currently does not support z-stacks.

### Command line arguments

**`-i`**  path to image stack

**`-c`**  name of reference channel

**`-o`**  output directory

**`-n`**  multiprocessing: number of processes, default 1

####Optional

**`--tile_size`**  size of a square tile, default 1000, which corresponds to 1000x1000px tile

**`--overlap`**  overlap between tiles, default 100

**`--method`**  optical flow method: farneback, denselk, deepflow, rlof, pcaflow, default rlof

### Example usage

**`python opt_flow_reg.py -i /path/to/iamge/stack/out.tif -c "DAPI" -o /path/to/output/dir/ -n 3`**


### Dependencies
`numpy tifffile opencv-contrib-python dask`

