import gc
from typing import List, Tuple

import cv2 as cv
import numpy as np
from opt_flow_reg.slicer import split_image_into_tiles_of_size
from opt_flow_reg.stitcher import stitch_image

Image = np.ndarray


class Warper:
    def __init__(self):
        self.image = np.array([])
        self.flow = np.array([])
        self.tile_size = 1000
        self.overlap = 100
        self._slicer_info = {}

    def warp(self):
        image_tiles, self._slicer_info = split_image_into_tiles_of_size(
            self.image, self.tile_size, self.tile_size, self.overlap
        )
        self.image = np.array([])
        flow_tiles, s_ = split_image_into_tiles_of_size(
            self.flow, self.tile_size, self.tile_size, self.overlap
        )
        self.flow = np.array([])
        warped_image_tiles = self._warp_image_tiles(image_tiles, flow_tiles)
        del image_tiles, flow_tiles
        stitched_warped_image = stitch_image(warped_image_tiles, self._slicer_info)

        self._slicer_info = {}
        del warped_image_tiles
        gc.collect()
        return stitched_warped_image

    def _make_flow_for_remap(self, flow):
        h, w = flow.shape[:2]
        new_flow = np.negative(flow)
        new_flow[:, :, 0] += np.arange(w)
        new_flow[:, :, 1] += np.arange(h).reshape(-1, 1)
        return new_flow

    def _warp_with_flow(self, img: Image, flow: np.ndarray) -> Image:
        """Warps input image according to optical flow"""
        new_flow = self._make_flow_for_remap(flow)
        res = cv.remap(img, new_flow, None, cv.INTER_LINEAR)
        gc.collect()
        return res

    def _warp_image_tiles(
        self, image_tiles: List[Image], flow_tiles: List[np.ndarray]
    ) -> List[Image]:
        warped_tiles = []
        # parallelizing this loop is not worth it - it only increases memory consumption and processing time
        for t in range(0, len(image_tiles)):
            warped_tiles.append(self._warp_with_flow(image_tiles[t], flow_tiles[t]))
        return warped_tiles
