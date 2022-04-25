from typing import Tuple, List
import gc
import sys

import numpy as np
import cv2 as cv

sys.path.append("/opt/image_registration")
from opt_flow_reg.slicer import split_image_into_tiles_of_size
from opt_flow_reg.stitcher import stitch_image

Image = np.ndarray


class Warper:
    def __init__(self):
        self.image = None
        self.image_tiles = []
        self.slicer_info = {}
        self.flow_tiles = []
        self.block_w = 1000
        self.block_h = 1000
        self.overlap = 50

    def warp(self):
        if self.image is not None:
            self.image_tiles, self.slicer_info = split_image_into_tiles_of_size(
                self.image, self.block_w, self.block_h, self.overlap
            )
            self.image = None  # cleanup
            gc.collect()

        warped_image_tiles = self.warp_image_tiles(self.image_tiles, self.flow_tiles)

        self.flow_tiles = []  # cleanup
        self.image_tiles = []  # cleanup
        gc.collect()
        stitched_warped_image = stitch_image(warped_image_tiles, self.slicer_info)

        self.slicer_info = {}  # cleanup
        del warped_image_tiles  # cleanup
        gc.collect()
        return stitched_warped_image

    def make_flow_for_remap(self, flow):
        h, w = flow.shape[:2]
        new_flow = np.negative(flow)
        new_flow[:, :, 0] = new_flow[:, :, 0] + np.arange(w)
        new_flow[:, :, 1] = new_flow[:, :, 1] + np.arange(h).reshape(-1, 1)
        return new_flow

    def warp_with_flow(self, img: Image, flow: np.ndarray) -> Image:
        """Warps input image according to optical flow"""
        new_flow = self.make_flow_for_remap(flow)
        res = cv.remap(img, new_flow, None, cv.INTER_LINEAR)
        gc.collect()
        return res

    def warp_image_tiles(
        self, image_tiles: List[Image], flow_tiles: List[np.ndarray]
    ) -> List[Image]:
        warped_tiles = []
        # parallelizing this loop is not worth it - it only increases memory consumption and processing time
        for t in range(0, len(image_tiles)):
            warped_tiles.append(self.warp_with_flow(image_tiles[t], flow_tiles[t]))

        return warped_tiles
