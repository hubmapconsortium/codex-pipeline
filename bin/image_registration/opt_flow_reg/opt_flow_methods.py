import numpy as np
import cv2 as cv

Image = np.ndarray


def farneback(mov_img: Image, ref_img: Image) -> np.ndarray:
    flow = cv.calcOpticalFlowFarneback(
        mov_img,
        ref_img,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=51,
        iterations=3,
        poly_n=1,
        poly_sigma=1.7,
        flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    # large values of poly_n produce smudges
    return flow


def denselk(mov_img: Image, ref_img: Image) -> np.ndarray:
    # rlof produces better results
    flow = cv.optflow.calcOpticalFlowSparseToDense(mov_img, ref_img, None, grid_step=2, k=500)
    return flow


def deepflow(mov_img: Image, ref_img: Image) -> np.ndarray:
    # takes long time, 2-3 times more than other methods
    df = cv.optflow.createOptFlow_DeepFlow()
    flow = df.calc(mov_img, ref_img, None)
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


def pcaflow(mov_img: Image, ref_img: Image) -> np.ndarray:
    pcaf = cv.optflow.createOptFlow_PCAFlow()
    flow = pcaf.calc(mov_img, ref_img, None)
    return flow
