import gc
import sys
from typing import List, Tuple

import cv2 as cv
import dask
import numpy as np
from opt_flow_reg.flow_calc import TileFlowCalc
from opt_flow_reg.slicer import split_image_into_tiles_of_size
from opt_flow_reg.stitcher import stitch_image
from opt_flow_reg.warper import Warper
from sklearn.metrics import normalized_mutual_info_score

Image = np.ndarray


def merge_two_flows(flow1: np.ndarray, flow2: np.ndarray) -> np.ndarray:
    # https://openresearchsoftware.metajnl.com/articles/10.5334/jors.380/
    # m_flow = of.combine_flows(flow1, flow2, 3, ref="s")
    if flow1.max() == 0:
        return flow2
    elif flow2.max() == 0:
        return flow1
    else:
        m_flow = flow1 + cv.remap(flow2, -flow1, None, cv.INTER_LINEAR)
    gc.collect()
    return m_flow


def mi_tiled(arr1: Image, arr2: Image, tile_size: int) -> float:
    if max(arr1.shape) / tile_size < 2:
        return normalized_mutual_info_score(arr1.flatten(), arr2.flatten())
    else:
        indices = list(range(tile_size * tile_size, arr1.size, tile_size * tile_size))
        arr1_parts = np.array_split(arr1.flatten(), indices)
        arr2_parts = np.array_split(arr2.flatten(), indices)
        tasks = []
        for i in range(0, len(arr1_parts)):
            if arr1_parts[i].size != 0:
                task = dask.delayed(normalized_mutual_info_score)(arr1_parts[i], arr2_parts[i])
                tasks.append(task)
        scores = dask.compute(*tasks)
        mi_score = np.mean(scores)
    return mi_score


class PyrRegOF:
    def __init__(self):
        self.ref_img = np.array([])
        self.mov_img = np.array([])
        self.num_pyr_lvl = 4
        self.num_iterations = 3
        self.tile_size = 1000
        self.overlap = 100
        self._warper = Warper()
        self._tile_flow_calc = TileFlowCalc()

    def _init_warper(self):
        self._warper = Warper()
        self._warper.tile_size = self.tile_size
        self._warper.overlap = self.overlap

    def _init_tile_flow_calc(self):
        self._tile_flow_calc = TileFlowCalc()
        self._tile_flow_calc.tile_size = self.tile_size
        self._tile_flow_calc.overlap = self.overlap
        self._tile_flow_calc.num_iter = self.num_iterations
        self._tile_flow_calc.win_size = self.overlap - (1 - self.overlap % 2)

    def register(self) -> np.ndarray:
        self._init_tile_flow_calc()
        self._init_warper()

        ref_pyr, factors = self._generate_img_pyr(self.ref_img)
        mov_pyr, f_ = self._generate_img_pyr(self.mov_img)

        for lvl, factor in enumerate(factors):
            print("Pyramid factor", factor)
            mov_this_lvl = mov_pyr[lvl].copy()

            # apply previous flow
            if lvl == 0:
                pass
            else:
                self._warper.image = mov_this_lvl
                self._warper.flow = m_flow
                mov_this_lvl = self._warper.warp()
                # mov_this_lvl = self.warp_with_flow(mov_this_lvl, m_flow)
            self._tile_flow_calc.ref_img = ref_pyr[lvl]
            self._tile_flow_calc.mov_img = mov_this_lvl
            this_flow = self._tile_flow_calc.calc_flow()
            self._warper.image = mov_this_lvl
            self._warper.flow = this_flow
            mov_this_lvl = self._warper.warp()
            gc.collect()
            # this_flow = self.calc_flow(ref_pyr[lvl], mov_this_lvl, 1, 0, 51)
            passed = self.check_if_passes(ref_pyr[lvl], mov_this_lvl, mov_pyr[lvl])
            if not any(passed):
                print("    No better alignment")
                if lvl == 0:
                    dstsize = list(mov_pyr[lvl + 1].shape)
                    m_flow = np.zeros(dstsize + [2], dtype=np.float32)
                else:
                    dstsize = mov_pyr[lvl + 1].shape[::-1]
                    m_flow = cv.pyrUp(m_flow * 2, dstsize=dstsize)
            else:
                print("    Found better alignment")
                # merge flows, upscale to next level
                if lvl == 0:
                    if len(factors) > 1:
                        dstsize = mov_pyr[lvl + 1].shape[::-1]
                        m_flow = cv.pyrUp(this_flow * 2, dstsize=dstsize)
                    else:
                        m_flow = this_flow
                elif lvl == len(factors) - 1:
                    m_flow = self._merge_list_of_flows([m_flow, this_flow])
                else:
                    m_flow = self._merge_list_of_flows([m_flow, this_flow])
                    dstsize = mov_pyr[lvl + 1].shape[::-1]
                    m_flow = cv.pyrUp(m_flow * 2, dstsize=dstsize)
                del this_flow
        del mov_pyr, ref_pyr
        gc.collect()
        return m_flow

    def _generate_img_pyr(self, arr: Image) -> Tuple[List[Image], List[int]]:
        # Pyramid scales from smallest to largest
        if self.num_pyr_lvl < 0:
            raise ValueError("Number of pyramid levels cannot be less than 0")
        # Pyramid scales from smallest to largest
        pyramid: List[Image] = []
        factors = []
        pyr_lvl = arr.copy()
        for lvl in range(0, self.num_pyr_lvl):
            factor = 2 ** (lvl + 1)
            if arr.shape[0] / factor < 100 or arr.shape[1] / factor < 100:
                break
            else:
                pyramid.append(cv.pyrDown(pyr_lvl))
                pyr_lvl = pyramid[lvl]
                factors.append(factor)
        factors = list(reversed(factors))
        pyramid = list(reversed(pyramid))
        pyramid.append(arr)
        factors.append(1)
        return pyramid, factors

    def _merge_flow_in_tiles(self, flow1: np.ndarray, flow2: np.ndarray):
        flow1_list, slicer_info = split_image_into_tiles_of_size(
            flow1, self.tile_size, self.tile_size, self.overlap
        )
        flow2_list, s_ = split_image_into_tiles_of_size(
            flow2, self.tile_size, self.tile_size, self.overlap
        )
        del flow1, flow2

        tasks = []
        for i in range(0, len(flow1_list)):
            task = dask.delayed(merge_two_flows)(flow1_list[i], flow2_list[i])
            tasks.append(task)
        merged_flow_tiles = dask.compute(*tasks)
        del flow1_list, flow2_list
        merged_flow = stitch_image(merged_flow_tiles, slicer_info)
        return merged_flow

    def _merge_list_of_flows(self, flow_list: List[np.ndarray]) -> np.ndarray:
        m_flow = flow_list[0]
        if len(flow_list) > 1:
            for i in range(1, len(flow_list)):
                m_flow = self._merge_flow_in_tiles(m_flow, flow_list[i])
        return m_flow

    def mutual_information_test(
        self, ref_arr: Image, test_arr: Image, init_arr: Image
    ) -> Tuple[float, float]:
        after_mi_score = mi_tiled(ref_arr, test_arr, self.tile_size)
        before_mi_score = mi_tiled(ref_arr, init_arr, self.tile_size)
        return after_mi_score, before_mi_score

    def check_if_passes(self, ref_arr: Image, test_arr: Image, init_arr: Image) -> List[bool]:
        mi_scores = self.mutual_information_test(ref_arr, test_arr, init_arr)
        checks = list()
        checks.append(mi_scores[0] > mi_scores[1])
        print("    MI score after:", mi_scores[0], "| MI score before:", mi_scores[1])
        return checks
