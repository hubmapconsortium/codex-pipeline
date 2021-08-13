from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from skimage.segmentation import find_boundaries

Image = np.ndarray

"""
Package functions that repair and generate matched cell, nuclear,
cell membrane and nuclear membrane segmentation masks
Author: Haoran Chen
Version: 1.1
08/09/2021
"""


def get_matched_cells(cell_arr, cell_membrane_arr, nuclear_arr, mismatch_repair):
    a = set((tuple(i) for i in cell_arr))
    b = set((tuple(i) for i in cell_membrane_arr))
    c = set((tuple(i) for i in nuclear_arr))
    d = a - b
    # remove cell membrane from cell
    mismatch_pixel_num = len(list(c - d))
    mismatch_fraction = len(list(c - d)) / len(list(c))
    if not mismatch_repair:
        if mismatch_pixel_num == 0:
            return np.array(list(a)), np.array(list(c)), 0
        else:
            return False, False, False
    else:
        if mismatch_pixel_num < len(c):
            return np.array(list(a)), np.array(list(d & c)), mismatch_fraction
        else:
            return False, False, False


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def list_remove(c_list, indexes):
    for index in sorted(indexes, reverse=True):
        del c_list[index]
    return c_list


def get_indexed_mask(mask, boundary):
    boundary = boundary * 1
    boundary_loc = np.where(boundary == 1)
    boundary[boundary_loc] = mask[boundary_loc]
    return boundary


def get_boundary(mask: Image):
    mask_boundary = find_boundaries(mask, mode="inner")
    mask_boundary_indexed = get_indexed_mask(mask, mask_boundary)
    return mask_boundary_indexed


def get_mask(cell_list, shape: Tuple[int]):
    mask = np.zeros(shape)
    for cell_num in range(len(cell_list)):
        mask[tuple(cell_list[cell_num].T)] = cell_num + 1
    return mask


def get_cell_num(mask: Image):
    return len(np.unique(mask))


def get_mismatched_fraction(
    whole_cell_mask: Image,
    nuclear_mask: Image,
    cell_matched_mask: Image,
    nuclear_matched_mask: Image,
) -> float:
    whole_cell_mask_binary = np.sign(whole_cell_mask)
    nuclear_mask_binary = np.sign(nuclear_mask)
    cell_matched_mask_binary = np.sign(cell_matched_mask)
    nuclear_matched_mask_binary = np.sign(nuclear_matched_mask)
    total_area = np.sum(np.sign(whole_cell_mask_binary + nuclear_mask_binary))
    mismatched_area = np.sum(
        np.sign(
            (nuclear_mask_binary - nuclear_matched_mask_binary)
            + (whole_cell_mask_binary - cell_matched_mask_binary)
        )
    )
    mismatched_fraction = mismatched_area / total_area
    return mismatched_fraction


def get_fraction_matched_cells(
    whole_cell_mask: Image, nuclear_mask: Image, cell_matched_mask: Image
) -> float:
    matched_cell_num = len(np.unique(cell_matched_mask)) - 1
    total_cell_num = len(np.unique(whole_cell_mask)) - 1
    total_nuclei_num = len(np.unique(nuclear_mask)) - 1
    mismatched_cell_num = total_cell_num - matched_cell_num
    mismatched_nuclei_num = total_nuclei_num - matched_cell_num
    fraction_matched_cells = matched_cell_num / (
        mismatched_cell_num + mismatched_nuclei_num + matched_cell_num
    )
    return fraction_matched_cells


def get_matched_masks(
    cell_mask: Image, nucleus_mask: Image, dtype, do_mismatch_repair: bool
) -> Tuple[List[Image], float]:
    """
    returns masks with matched cells and nuclei
    """
    whole_cell_mask = cell_mask.copy()
    nuclear_mask = nucleus_mask.copy()
    cell_membrane_mask = get_boundary(whole_cell_mask)

    cell_coords = get_indices_sparse(whole_cell_mask)[1:]
    nucleus_coords = get_indices_sparse(nuclear_mask)[1:]
    cell_membrane_coords = get_indices_sparse(cell_membrane_mask)[1:]

    cell_coords = list(map(lambda x: np.array(x).T, cell_coords))
    nucleus_coords = list(map(lambda x: np.array(x).T, nucleus_coords))
    cell_membrane_coords = list(map(lambda x: np.array(x).T, cell_membrane_coords))

    cell_matched_index_list = []
    nucleus_matched_index_list = []
    cell_matched_list = []
    nucleus_matched_list = []

    for i in range(len(cell_coords)):
        if len(cell_coords[i]) != 0:
            current_cell_coords = cell_coords[i]
            nuclear_search_num = np.unique(
                list(map(lambda x: nuclear_mask[tuple(x)], current_cell_coords))
            )
            best_mismatch_fraction = 1
            whole_cell_best = []
            for j in nuclear_search_num:
                if j != 0:
                    if (j - 1 not in nucleus_matched_index_list) and (
                        i not in cell_matched_index_list
                    ):
                        whole_cell, nucleus, mismatch_fraction = get_matched_cells(
                            cell_coords[i],
                            cell_membrane_coords[i],
                            nucleus_coords[j - 1],
                            mismatch_repair=do_mismatch_repair,
                        )
                        if type(whole_cell) != bool:
                            if mismatch_fraction < best_mismatch_fraction:
                                best_mismatch_fraction = mismatch_fraction
                                whole_cell_best = whole_cell
                                nucleus_best = nucleus
                                i_ind = i
                                j_ind = j - 1
            if len(whole_cell_best) > 0:
                cell_matched_list.append(whole_cell_best)
                nucleus_matched_list.append(nucleus_best)
                cell_matched_index_list.append(i_ind)
                nucleus_matched_index_list.append(j_ind)

    del cell_coords
    del nucleus_coords

    cell_matched_mask = get_mask(cell_matched_list, whole_cell_mask.shape)
    nuclear_matched_mask = get_mask(nucleus_matched_list, whole_cell_mask.shape)
    cell_membrane_mask = get_boundary(cell_matched_mask)
    nuclear_membrane_mask = get_boundary(nuclear_matched_mask)

    if do_mismatch_repair:
        fraction_matched_cells = 1.0
    else:
        fraction_matched_cells = get_fraction_matched_cells(
            whole_cell_mask, nuclear_mask, cell_matched_mask
        )

    out_list = [
        cell_matched_mask.astype(dtype),
        nuclear_matched_mask.astype(dtype),
        cell_membrane_mask.astype(dtype),
        nuclear_membrane_mask.astype(dtype),
    ]
    return out_list, fraction_matched_cells
