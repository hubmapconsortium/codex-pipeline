import gc
from copy import deepcopy
from typing import List, Tuple, Union

import cv2 as cv
import dask
import numpy as np
from skimage.transform import AffineTransform, warp
from sklearn.metrics import normalized_mutual_info_score

from feature_reg.feature_detection import Features
from feature_reg.tile_registration import find_features, register_img_pair

Image = np.ndarray


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


class PyrReg:
    def __init__(self):
        self.ref_img = np.array([])
        self.num_pyr_lvl = 4
        self.num_iterations = 3
        self.tile_size = 1000
        self._ref_pyr_features = []
        self._ref_img_pyr = []
        self._factors = [16, 8, 4, 2]
        self._this_pyr_factor = 1

    def calc_ref_img_features(self):
        if len(self.ref_img) == 0:
            raise ValueError("No reference image specified")
        self._ref_img_pyr, self._factors = self._generate_img_pyr(self.ref_img)
        self._ref_pyr_features = []
        for pyr_level in self._ref_img_pyr:
            self._ref_pyr_features.append(find_features(self.dog(pyr_level), self.tile_size))

    def register(self, mov_img) -> np.ndarray:
        if len(self.ref_img) == 0:
            raise ValueError("No reference image specified")
        if len(self._ref_img_pyr) == [] or len(self._ref_pyr_features) == []:
            raise ValueError("Calculate reference image features first")

        mov_img_pyrs, factors = self._generate_img_pyr(mov_img)

        fullscale_t_mat_list = []
        for i, factor in enumerate(self._factors):
            print("Pyramid scale", factor)
            self._this_pyr_factor = factor
            if i == 0:
                mov_img_this_scale_transform, t_mat = self._iterative_alignment(
                    self._ref_img_pyr[i], self._ref_pyr_features[i], mov_img_pyrs[i]
                )
            else:
                rescaled_t_mat_list = [
                    self._rescale_t_mat(m, 1 / factor) for m in fullscale_t_mat_list
                ]
                this_scale_t_mat = self._multiply_transform_matrices(rescaled_t_mat_list)
                mov_img_prev_scale_transform = self.transform_img(
                    mov_img_pyrs[i], this_scale_t_mat
                )
                mov_img_this_scale_transform, t_mat = self._iterative_alignment(
                    self._ref_img_pyr[i],
                    self._ref_pyr_features[i],
                    mov_img_prev_scale_transform,
                )
            fullscale_t_mat_list.append(self._rescale_t_mat(t_mat, factor))
            gc.collect()
        final_transform = self._multiply_transform_matrices(fullscale_t_mat_list)
        return final_transform

    def transform_big_img(self, img: Image, transform_matrix: np.ndarray) -> Image:
        orig_dtype = deepcopy(img.dtype)
        homogenous_transform_matrix = np.append(transform_matrix, [[0, 0, 1]], axis=0)
        inv_matrix = np.linalg.pinv(homogenous_transform_matrix)
        AT = AffineTransform(inv_matrix)
        img = warp(img, AT, output_shape=img.shape, preserve_range=True).astype(orig_dtype)
        return img

    def transform_img(self, img: Image, transform_matrix: np.ndarray) -> Image:
        if max(img.shape) > 32000:
            return self.transform_big_img(img, transform_matrix)
        else:
            return cv.warpAffine(img, transform_matrix, dsize=img.shape[::-1])

    def _generate_img_pyr(self, arr: Image) -> Tuple[List[Image], List[int]]:
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

    def _iterative_alignment(
        self, ref_img: Image, ref_features: Features, mov_img: Image
    ) -> Tuple[Image, np.ndarray]:
        if self.num_iterations < 1:
            raise ValueError("Number of iterations cannot be less than 1")
        t_matrices = []
        aligned_img = mov_img.copy()
        for i in range(0, self.num_iterations):
            print("    Iteration", i + 1, "/", self.num_iterations)
            mov_img_aligned, est_t_mat_pyr = self._align_imgs(ref_features, aligned_img)
            check_results = self.check_if_passes(ref_img, mov_img_aligned, aligned_img)
            if any(check_results) and self._check_if_valid_transform(est_t_mat_pyr, mov_img.shape):
                print("    Found better alignment")
                t_matrices.append(est_t_mat_pyr)
                aligned_img = self._realign_img(mov_img, t_matrices)
            else:
                print("    No better alignment")
                t_matrices.append(np.eye(2, 3))
                aligned_img = aligned_img
        final_t_mat = self._multiply_transform_matrices(t_matrices)
        return aligned_img, final_t_mat

    def _align_imgs(self, ref: Union[Image, Features], mov_img: Image) -> Tuple[Image, np.ndarray]:
        if not isinstance(ref, Features):
            ref_features = find_features(self.dog(ref), self.tile_size)
        else:
            ref_features = ref
        mov_features = find_features(self.dog(mov_img), self.tile_size)
        transform_mat = register_img_pair(ref_features, mov_features)
        if np.equal(transform_mat, np.eye(2, 3)).all():
            return mov_img, np.eye(2, 3)
        else:
            img_aligned = self.transform_img(mov_img, transform_mat)
            return img_aligned, transform_mat

    def _realign_img(self, mov_img: Image, mat_list: List[np.ndarray]) -> Image:
        mul_mat = self._multiply_transform_matrices(mat_list)
        img_aligned = self.transform_img(mov_img, mul_mat)
        return img_aligned

    def _multiply_transform_matrices(self, mat_list: List[np.ndarray]) -> np.ndarray:
        if len(mat_list) == 1:
            return mat_list[0]
        hom_mats = [np.append(mat, [[0, 0, 1]], axis=0) for mat in mat_list]
        res_mat = hom_mats[0]
        for i in range(1, len(hom_mats)):
            res_mat = res_mat @ hom_mats[i]
        res_mat_short = res_mat[:2, :]
        return res_mat_short

    def _rescale_t_mat(self, t_mat: np.ndarray, scale: float) -> np.ndarray:
        t_mat_copy = t_mat.copy()
        t_mat_copy[0, 2] *= scale
        t_mat_copy[1, 2] *= scale
        return t_mat_copy

    def _check_if_valid_transform(self, t_mat: np.ndarray, img_shape: Tuple[int, int]) -> bool:
        is_inside_border = self._check_if_inside_borders(t_mat, img_shape)
        is_proper_scale = self._check_if_proper_scale(t_mat)
        if all((is_inside_border, is_proper_scale)):
            return True
        else:
            return False

    def _check_if_proper_scale(self, t_mat: np.ndarray):
        # https://frederic-wang.fr/decomposition-of-2d-transform-matrices.html
        # |a c e|
        # |b d f|
        a = t_mat[0, 0]
        b = t_mat[1, 0]
        c = t_mat[0, 1]
        d = t_mat[1, 1]

        det = a * d - b * c
        if a != 0 or b != 0:
            r = np.sqrt(a**2 + b**2)
            scale = (r, det / r)
        elif c != 0 or d != 0:
            s = np.sqrt(c**2 + d**2)
            scale = (det / s, s)
        else:
            scale = (0, 0)
        print(t_mat)
        print(scale)
        if scale == (0, 0):
            return False
        elif abs(scale[0]) > 3 or abs(scale[1]) > 3:
            return False
        elif abs(scale[0]) < 0.3 or abs(scale[1]) < 0.3:
            return False
        else:
            return True

    def _check_if_inside_borders(self, t_mat: np.ndarray, img_shape: Tuple[int, int]) -> bool:
        cy = img_shape[0] // 2
        cx = img_shape[1] // 2
        center_coords = np.array([[cx], [cy], [1]])
        border_coords = np.array([[img_shape[1]], [img_shape[0]], [1]])
        t_mat_hom = np.append(t_mat, [[0, 0, 1]], axis=0)
        transf_center = t_mat_hom @ center_coords
        if np.any((border_coords - np.abs(transf_center)) < 0):
            return False
        else:
            return True

    def mutual_information_test(self, ref_arr, test_arr, init_arr):
        after_mi_score = mi_tiled(self.dog(ref_arr), self.dog(test_arr), self.tile_size)
        before_mi_score = mi_tiled(self.dog(ref_arr), self.dog(init_arr), self.tile_size)
        return after_mi_score, before_mi_score

    def check_if_passes(self, ref_arr, test_arr, init_arr):
        mi_scores = self.mutual_information_test(ref_arr, test_arr, init_arr)
        checks = list()
        checks.append(mi_scores[0] > mi_scores[1])
        print("    MI score after:", mi_scores[0], "| MI score before:", mi_scores[1])
        return checks

    def get_dog_sigmas(self, pyr_factor: int) -> Tuple[int, int]:
        if pyr_factor > 16:
            return 1, 2
        else:
            sigmas = {1: (5, 9), 2: (4, 7), 4: (3, 5), 8: (2, 3), 16: (1, 2)}
        return sigmas[pyr_factor]

    def dog(self, img: Image, low_sigma: int = 5, high_sigma: int = 9) -> Image:
        """Difference of Gaussian"""
        if img.max() == 0:
            return img
        else:
            low_sigma, high_sigma = self.get_dog_sigmas(self._this_pyr_factor)

            fimg = cv.normalize(img, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
            kernel = (low_sigma * 4 * 2 + 1, low_sigma * 4 * 2 + 1)  # as in opencv
            ls = cv.GaussianBlur(fimg, kernel, sigmaX=low_sigma, dst=None, sigmaY=low_sigma)
            hs = cv.GaussianBlur(fimg, kernel, sigmaX=high_sigma, dst=None, sigmaY=high_sigma)
            diff_of_gaussians = hs - ls
            del hs, ls
            return cv.normalize(diff_of_gaussians, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
