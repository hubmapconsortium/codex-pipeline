import sys
import gc
from typing import List

import cv2 as cv
import numpy as np

sys.path.append("/opt/image_registration")
from feature_reg.slicer import split_image_into_tiles_of_size
from feature_reg.feature_detection import find_features_parallelized, match_features, Features


def split_image_into_tiles(img, tile_size: int):
    x_size = tile_size
    y_size = tile_size
    img_tiles, tile_info = split_image_into_tiles_of_size(
        img, x_size, y_size, overlap=51
    )
    return img_tiles, tile_info


def combine_features(
    feature_list: List[Features],
    x_ntiles: int,
    y_ntiles: int,
    tile_size_x: int,
    tile_size_y: int,
) -> Features:
    keypoints_combined = []
    descriptors_list = []
    for tile_id, feature in enumerate(feature_list):
        if feature.is_valid():
            keypoints = feature.keypoints
            descriptors = feature.descriptors

            descriptors_list.append(descriptors)
            for i, kp in enumerate(keypoints):
                tile_coord_x = tile_id % x_ntiles * tile_size_x
                tile_coord_y = tile_id // x_ntiles * tile_size_y
                new_coords = (tile_coord_x + kp.pt[0], tile_coord_y + kp.pt[1])
                new_kp = cv.KeyPoint(
                    x=new_coords[0],
                    y=new_coords[1],
                    size=kp.size,
                    angle=kp.angle,
                    response=kp.response,
                    octave=kp.octave,
                    class_id=kp.class_id,
                )
                keypoints_combined.append(new_kp)
    combined_features = Features()
    if keypoints_combined == [] or descriptors_list == []:
        combined_features.keypoints = None
        combined_features.descriptors = None
    else:
        descriptors_combined = np.concatenate(descriptors_list, axis=0)
        combined_features.keypoints = keypoints_combined
        combined_features.descriptors = descriptors_combined
    return combined_features


def find_features(img, tile_size):
    img_tiles, img_tile_info = split_image_into_tiles(img, tile_size)

    x_ntiles = img_tile_info["ntiles"]["x"]
    y_ntiles = img_tile_info["ntiles"]["y"]
    tile_size_y, tile_size_x = img_tile_info["tile_shape"]

    tiles_features = find_features_parallelized(img_tiles)
    del img_tiles
    combined_features = combine_features(
        tiles_features, x_ntiles, y_ntiles, tile_size_x, tile_size_y
    )
    del tiles_features
    gc.collect()
    return combined_features


def register_img_pair(ref_combined_features, mov_combined_features):
    transform_matrix = match_features(ref_combined_features, mov_combined_features)
    return transform_matrix
