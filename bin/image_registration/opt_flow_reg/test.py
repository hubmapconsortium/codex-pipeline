import sys
from pathlib import Path
import numpy as np
import cv2 as cv
import tifffile as tif
import matplotlib.pyplot as plt
sys.path.append("C:/Files/source/image_registration/image_registration/feature_reg")
from pyr_reg import PyrReg

def calculate_padding_size(bigger_shape, smaller_shape):
    """Find difference between shapes of bigger and smaller image."""
    diff = bigger_shape - smaller_shape

    if diff == 1:
        dim1 = 1
        dim2 = 0
    elif diff % 2 != 0:
        dim1 = int(diff // 2)
        dim2 = int((diff // 2) + 1)
    else:
        dim1 = dim2 = int(diff / 2)

    return dim1, dim2


def pad_to_size(target_shape, img):
    if img.shape == target_shape:
        return img, (0, 0, 0, 0)
    else:
        left, right = calculate_padding_size(target_shape[1], img.shape[1])
        top, bottom = calculate_padding_size(target_shape[0], img.shape[0])
        return cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, None, 0), (
            left,
            right,
            top,
            bottom,
        )


img_ref_path = Path("C:/Files/Data/acrobat_train_x5_example/100_PGR_x5_z0.tif")
img_mov_path = Path("C:/Files/Data/acrobat_train_x5_example/100_HE_x5_z0.tif")

img_ref = cv.cvtColor(tif.imread(img_ref_path), cv.COLOR_RGB2GRAY)
img_mov = cv.cvtColor(tif.imread(img_mov_path), cv.COLOR_RGB2GRAY)


img_ref_padded, pad = pad_to_size(img_mov.shape, img_ref)

registrator = PyrReg()
registrator.ref_img = img_ref_padded
registrator.num_scales = 5
registrator.num_iterations = 5
registrator.calc_ref_img_features()
reg_matrix = registrator.register(img_mov)

reg_matrix = np.array([[ 8.73035363e-01, -8.64448508e-04,  1.56102476e+03],
                      [ 8.64448508e-04,  8.73035363e-01,  6.93355614e+02]])

reg_res = cv.warpAffine(img_mov, reg_matrix, img_ref_padded.shape[::-1])

fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
ax[0].imshow(img_ref_padded)
ax[1].imshow(reg_res)
plt.imshow(img_ref_padded - reg_res, cmap="gray")

#tif.imwrite("C:/Files/Data/acrobat_train_x5_example/test1.tif", np.stack((img_ref_padded, reg_res), axis=0))

tif.imwrite("C:/Files/Data/acrobat_train_x5_example/test1.tif", img_ref_padded)
tif.imwrite("C:/Files/Data/acrobat_train_x5_example/test2.tif", reg_res)



sys.path.append("C:/Files/source/image_registration/image_registration/opt_flow_reg")
Image = np.ndarray


def farneback(mov_img: Image, ref_img: Image) -> np.ndarray:
    flow = cv.calcOpticalFlowFarneback(
        mov_img,
        ref_img,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=101,
        iterations=3,
        poly_n=1,
        poly_sigma=1.7,
        flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    # large values of poly_n produce smudges
    return flow


def rlof(mov_img: Image, ref_img: Image) -> np.ndarray:
    # I do not know if param can be pickled so it is better to create new instance each time
    param = cv.optflow.RLOFOpticalFlowParameter_create()
    param.setSupportRegionType(cv.optflow.SR_FIXED)  # SR_CROSS works with RGB images only
    param.setSolverType(cv.optflow.ST_BILINEAR)  # ST_STANDART produces smudges
    param.setUseGlobalMotionPrior(True)
    param.setUseIlluminationModel(True)
    param.setUseMEstimator(False)  # produces artefacts if on
    param.setMaxLevel(5)  # max pyramid level, default 4
    param.setMaxIteration(30)  # default 30
    param.setLargeWinSize(51)  # default 21
    flow = cv.optflow.calcOpticalFlowDenseRLOF(
        mov_img, ref_img, None, rlofParam=param, interp_type=cv.optflow.INTERP_GEO
    )
    return flow

def make_flow_for_remap(flow):
    h, w = flow.shape[:2]
    new_flow = np.negative(flow)
    new_flow[:, :, 0] = new_flow[:, :, 0] + np.arange(w)
    new_flow[:, :, 1] = new_flow[:, :, 1] + np.arange(h).reshape(-1, 1)
    return new_flow

def warp_with_flow(img: Image, flow: np.ndarray) -> Image:
    """Warps input image according to optical flow"""
    new_flow = make_flow_for_remap(flow)
    res = cv.remap(img, new_flow, None, cv.INTER_LINEAR)
    return res

def diff_of_gaus(img: Image, low_sigma: int = 5, high_sigma: int = 9) -> Image:
    if img.max() == 0:
        return img
    else:
        fimg = cv.normalize(img, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
        kernel = (low_sigma * 4 * 2 + 1, low_sigma * 4 * 2 + 1)  # as in opencv
        ls = cv.GaussianBlur(fimg, kernel, sigmaX=low_sigma, dst=None, sigmaY=low_sigma)
        hs = cv.GaussianBlur(fimg, kernel, sigmaX=high_sigma, dst=None, sigmaY=high_sigma)
        dog = hs - ls
        del hs, ls
        return cv.normalize(dog, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

scales = [0.005, 0.01, 0.03, 0.05, 0.1]   #0.01, 0.03
sigma_small = [1, 2, 3, 5]
sigma_big = [2, 3, 5, 9]
mov_img_ = reg_res.copy()
num_iter = 3
for i, scale in enumerate(scales):
    for n in range(0, num_iter):
        print(n, scale)
        reg_res_small = cv.resize(mov_img_, dsize=(0,0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        img_ref_padded_small = cv.resize(img_ref_padded, dsize=(0,0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)

        #flow_small = farneback(diff_of_gaus(reg_res_small, sigma_small[i], sigma_big[i]), diff_of_gaus(img_ref_padded_small, sigma_small[i], sigma_big[i]))
        flow_small = farneback(reg_res_small, img_ref_padded_small)

        flow_big = cv.resize(flow_small * (1 / scale), dsize=img_ref_padded.shape[::-1], interpolation=cv.INTER_CUBIC)
        mov_img_ = warp_with_flow(mov_img_, flow_big)

tif.imwrite("C:/Files/Data/acrobat_train_x5_example/stack_dog2.tif", np.stack((img_ref_padded, mov_img_, reg_res), axis=0), photometric="minisblack")
tif.imwrite("C:/Files/Data/acrobat_train_x5_example/stack.tif", np.stack((img_ref_padded, mov_img_, reg_res), axis=0), photometric="minisblack")
tif.imwrite("C:/Files/Data/acrobat_train_x5_example/test3.tif", reg_res2)


def generate_img_pyr(arr: Image, num_lvl, start_from_lvl=0):
    # Pyramid scales from smallest to largest
    pyramid = []
    factors = []
    pyr_scale = arr
    for scale in range(0, num_lvl):
        pyramid.append(cv.pyrDown(pyr_scale))
        pyr_scale = pyramid[scale]
        factors.append(2 ** (scale + 1))
    if start_from_lvl > 0 and not start_from_lvl > num_lvl:
        factors = list(reversed(factors[start_from_lvl:]))
        pyramid = list(reversed(pyramid[start_from_lvl:]))
    else:
        factors = list(reversed(factors))
        pyramid = list(reversed(pyramid))
    return pyramid, factors

from math import log

def upscale_pyr_lvl(arr, factor, final_size):
    lvl = round(log(factor, 2))
    res = arr
    for i in range(0, lvl):
        if i == lvl - 1:
            res = cv.pyrUp(res, dstsize=final_size)
        else:
            res = cv.pyrUp(res)
    return res

ref_pyr, factors = generate_img_pyr(img_ref_padded, 6, start_from_lvl=3)

sigma_small = [1, 1, 2, 3, 5]
sigma_big = [1, 2, 3, 5, 9]


def iterative_registration(ref_pyr, mov_img):
    aligned_img = mov_img.copy()
    num_iter = 3
    flows = []
    for i, factor in enumerate(factors):
        for n in range(0, num_iter):
            print(n, factor)
            mov_img_small_pyr, f_ = generate_img_pyr(aligned_img, 6, start_from_lvl=3)

            #flow_small = farneback(diff_of_gaus(mov_img_small_pyr[i], sigma_small[i], sigma_big[i]), diff_of_gaus(ref_pyr[i], sigma_small[i], sigma_big[i]))
            flow_small = farneback(mov_img_small_pyr[i], ref_pyr[i])
            flow_big = upscale_pyr_lvl(flow_small * factor, factor, img_ref_padded.shape[::-1])
            aligned_img = warp_with_flow(aligned_img, flow_big)
            flows.append(flow_big)
    return aligned_img, flows


tif.imwrite("C:/Files/Data/acrobat_train_x5_example/stack_pyr.tif", np.stack((img_ref_padded, mov_img_, reg_res), axis=0), photometric="minisblack")
tif.imwrite("C:/Files/Data/acrobat_train_x5_example/stack_pyr_dog.tif", np.stack((img_ref_padded, mov_img_, reg_res), axis=0), photometric="minisblack")
