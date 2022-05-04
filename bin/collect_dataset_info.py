import argparse
import csv
import json
import logging
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from os import fspath, walk
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pint
import psutil

from pipeline_utils.dataset_listing import get_tile_dtype, get_tile_shape


class ConfigCreator:
    def __init__(self):
        self.dataset_dir = Path("")
        self._std_meta = dict()
        self._raw_data_dir = Path("")

    def read_metadata(self):
        path_to_meta = self._raw_data_dir / "dataset.json"
        meta = self._read_json_meta(path_to_meta)
        processed_meta = meta.copy()

        ch_names = []
        for ch in meta["ChannelDetails"]["ChannelDetailsArray"]:
            ch_names.append(ch["Name"])

        new_ch_names = self._make_ch_names_unique(ch_names)

        new_channel_details_array = []
        for i, ch in enumerate(processed_meta["ChannelDetails"]["ChannelDetailsArray"]):
            new_ch = ch.copy()
            new_ch["Name"] = new_ch_names[i]
            new_channel_details_array.append(new_ch)
        processed_meta["ChannelDetails"]["ChannelDetailsArray"] = new_channel_details_array
        self._std_meta = processed_meta

    def find_raw_data_dir(self):
        NONRAW_DIRECTORY_NAME_PIECES = [
            "processed",
            "drv",
            "metadata",
            "extras",
            "Overview",
        ]
        raw_data_dir_possibilities = []

        for child in self.dataset_dir.iterdir():
            if not child.is_dir():
                continue
            if not any(piece in child.name for piece in NONRAW_DIRECTORY_NAME_PIECES):
                raw_data_dir_possibilities.append(child)

        if len(raw_data_dir_possibilities) > 1:
            message_pieces = ["Found multiple raw data directory possibilities:"]
            message_pieces.extend(f"\t{path}" for path in raw_data_dir_possibilities)
            raise ValueError("\n".join(message_pieces))
        self._raw_data_dir = raw_data_dir_possibilities[0]
        return self._raw_data_dir

    def create_config(self) -> dict:
        config = {
            "name": self._std_meta["DatasetName"],
            "date": self._create_proc_date(),
            "raw_data_location": self.find_raw_data_dir().name,
            "channel_names_qc_pass": self._get_qc_info_per_ch(),
            "emission_wavelengths": self._get_emission_wavelengths(),
            "excitation_wavelengths": self._get_excitation_wavelengths(),
            "axial_resolution": self._get_axial_resolution(),
            "lateral_resolution": self._get_lateral_resolution(),
            "magnification": self._std_meta["NominalMagnification"],
            "num_z_planes": self._std_meta["NumZPlanes"],
            "numerical_aperture": self._std_meta["NumericalAperture"],
            "objective_type": self._std_meta["ImmersionMedium"].lower(),
            "region_height": self._std_meta["RegionHeight"],
            "region_width": self._std_meta["RegionWidth"],
            "region_names": self._get_region_names(),
            "tile_overlap_x": self._get_tile_overlap_x_in_px(),
            "tile_overlap_y": self._get_tile_overlap_y_in_px(),
            "tile_height": self._get_tile_shape_no_overlap()[0],
            "tile_width": self._get_tile_shape_no_overlap()[1],
            "tile_dtype": self._get_tile_dtype(),
            "tiling_mode": self._std_meta["TileLayout"].lower(),
            "per_cycle_channel_names": self._get_per_cycle_ch_names(),
            "channel_names": self._get_channel_names(),
            "num_cycles": self._std_meta["NumCycles"],
            "best_focus": self._get_nuc_ch(),
            "drift_compensation": self._get_nuc_ch(),
            "nuclei_channel": self._get_nuc_ch(),
            "membrane_channel": self._get_membr_ch(),
            "nuclei_channel_loc": self._std_meta["NuclearStainForSegmentation"],
            "membrane_channel_loc": self._std_meta["MembraneStainForSegmentation"],
            "target_shape": self._calc_target_shape(),
            "num_concurrent_tasks": self._get_num_concur_tasks(),
        }
        return config

    def _read_json_meta(self, path_to_meta: Path) -> Dict[str, Union[str, int, dict, list]]:
        with open(path_to_meta, "r") as s:
            json_meta = json.load(s)
        return json_meta

    def _create_proc_date(self) -> str:
        processing_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return processing_date

    def _get_qc_info_per_ch(self) -> Dict[str, List[str]]:
        ch_details = self._std_meta["ChannelDetails"]["ChannelDetailsArray"]
        channel_qc_info = dict()
        channel_qc_info["Marker"] = ["Result"]
        for ch in ch_details:
            ch_name = ch["Name"]
            qc_result = ch["PassedQC"]
            if qc_result is True:
                qc_result_str = "TRUE"
            else:
                qc_result_str = "FALSE"
            channel_qc_info[ch_name] = [qc_result_str]
        return channel_qc_info

    def _make_ch_names_unique(self, channel_names: List[str]) -> List[str]:
        unique_names = Counter(channel_names)
        new_names = channel_names.copy()

        for unique_ch, count in unique_names.items():
            if count > 1:
                this_ch_count = 1
                for i, ch_name in enumerate(channel_names):
                    if ch_name == unique_ch:
                        new_name = f"{ch_name}_{this_ch_count}"
                        new_names[i] = new_name
                        this_ch_count += 1
        return new_names

    def _get_emission_wavelengths(self) -> List[float]:
        em_wav = []
        for ch in self._std_meta["ChannelDetails"]["ChannelDetailsArray"]:
            wav = ch["EmissionWavelengthNM"]
            if wav not in em_wav:
                em_wav.append(float(wav))
        return em_wav

    def _get_excitation_wavelengths(self) -> List[float]:
        exc_wav = []
        for ch in self._std_meta["ChannelDetails"]["ChannelDetailsArray"]:
            wav = ch["ExcitationWavelengthNM"]
            if wav not in exc_wav:
                exc_wav.append(float(wav))
        return exc_wav

    def _get_axial_resolution(self) -> float:
        unit = pint.UnitRegistry()
        provided_unit_z = unit[self._std_meta["ResolutionZUnit"]]
        provided_res_z = float(self._std_meta["ResolutionZ"])
        res_z_in_units = provided_res_z * provided_unit_z
        axial_res_um = res_z_in_units.to("nm")
        return axial_res_um.magnitude

    def _get_lateral_resolution(self) -> float:
        unit = pint.UnitRegistry()
        provided_unit_x = unit[self._std_meta["ResolutionXUnit"]]
        provided_unit_y = unit[self._std_meta["ResolutionYUnit"]]
        provided_res_x = float(self._std_meta["ResolutionX"])
        provided_res_y = float(self._std_meta["ResolutionY"])
        res_x_in_units = provided_res_x * provided_unit_x
        res_y_in_units = provided_res_y * provided_unit_y
        lateral_res_um = ((res_x_in_units + res_y_in_units) / 2).to("nm")
        return lateral_res_um.magnitude

    def _get_region_names(self) -> List[int]:
        num_regions = self._std_meta["NumRegions"]
        return list(range(1, num_regions + 1))

    def _get_tile_overlap_x_in_px(self) -> int:
        overlap = self._std_meta["TileOverlapX"]
        size = self._std_meta["TileWidth"]
        px_overlap = self._calc_px_overlap_from_proportional(size, overlap)
        return px_overlap

    def _get_tile_overlap_y_in_px(self) -> int:
        overlap = self._std_meta["TileOverlapY"]
        size = self._std_meta["TileHeight"]
        px_overlap = self._calc_px_overlap_from_proportional(size, overlap)
        return px_overlap

    def _calc_px_overlap_from_proportional(self, dim_size: int, dim_overlap: float) -> int:
        msg = f"Tile overlap proportion {dim_overlap} is greater than 1"
        if dim_overlap > 1:
            raise ValueError(msg)

        pixel_overlap = dim_size * dim_overlap

        if float(pixel_overlap).is_integer():
            return int(pixel_overlap)
        else:
            # if overlap is not a whole number in px
            closest_overlap = int(math.ceil(pixel_overlap))
            closest_overlap += closest_overlap % 2  # make even
            return closest_overlap

    def _get_per_cycle_ch_names(self) -> List[str]:
        per_cycle_channel_names = []
        channels = self._std_meta["ChannelDetails"]["ChannelDetailsArray"]
        channel_ids = []
        for ch in channels:
            channel_ids.append(int(ch["ChannelID"]))
        unique_ch_ids = sorted(set(channel_ids))
        for ch in unique_ch_ids:
            per_cycle_channel_names.append("CH" + str(ch))
        return per_cycle_channel_names

    def _get_channel_names(self) -> List[str]:
        channels = self._std_meta["ChannelDetails"]["ChannelDetailsArray"]
        channel_names = []
        for ch in channels:
            channel_names.append(ch["Name"])
        return channel_names

    def _get_nuc_ch(self) -> str:
        nuc_ch_loc = self._std_meta["NuclearStainForSegmentation"]
        nuc_ch_name = self._get_ch_name_by_location(nuc_ch_loc)
        return nuc_ch_name

    def _get_membr_ch(self) -> str:
        membr_ch_loc = self._std_meta["MembraneStainForSegmentation"]
        membr_ch_name = self._get_ch_name_by_location(membr_ch_loc)
        return membr_ch_name

    def _get_ch_name_by_location(self, ch_loc: Dict[str, int]) -> str:
        channels = self._std_meta["ChannelDetails"]["ChannelDetailsArray"]
        ch_name = None
        for ch in channels:
            if ch["CycleID"] == ch_loc["CycleID"]:
                if ch["ChannelID"] == ch_loc["ChannelID"]:
                    ch_name = ch["Name"]
                    break
        if ch_name is None:
            raise ValueError("Could not find channel name of", str(ch_loc))
        return ch_name

    def _get_tile_dtype(self) -> str:
        tile_dtype = str(get_tile_dtype(self._raw_data_dir))
        return tile_dtype

    def _calc_target_shape(self):
        """
        Cytokit's nuclei detection U-Net (from CellProfiler) works best at 20x magnification.
        The CellProfiler U-Net requires the height and width of the images to be
        evenly divisible by 2 raised to the number of layers in the network, in this case 2^3=8.
        https://github.com/hammerlab/cytokit/issues/14
        https://github.com/CellProfiler/CellProfiler-plugins/issues/65
        """
        dims = [self._std_meta["TileWidth"], self._std_meta["TileHeight"]]
        magnification = self._std_meta["NominalMagnification"]
        scaleFactor = 1
        if magnification != 20:
            scaleFactor = 20 / magnification

        # Width and height must be evenly divisible by 8, so we round them up to them
        # closest factor of 8 if they aren't.
        new_dims = dims.copy()
        for dim in dims:
            if dim % 8:
                new_dim = int(8 * math.ceil(float(dim) / 8))
                new_dims.append(new_dim)
        return new_dims

    def _get_num_concur_tasks(self) -> int:
        ram_stats = psutil.virtual_memory()
        num_cpus = psutil.cpu_count()
        free_ram_gb = ram_stats.available / 1024**3
        dtype = self._get_tile_dtype()
        img_dtype = int(re.search(r"(\d+)", np.dtype(dtype).name).groups()[0])  # int16 -> 16
        nbytes = img_dtype / 8

        n_tiles_per_plane = int(self._std_meta["RegionHeight"]) * int(
            self._std_meta["RegionWidth"]
        )
        n_pixels = int(self._std_meta["TileHeight"]) * int(self._std_meta["TileWidth"])
        img_plane_size = n_tiles_per_plane * n_pixels * nbytes

        img_plane_size += round(img_plane_size * 0.1)  # 10% overhead
        img_plane_size_gb = img_plane_size / 1024**3

        num_of_concurrent_tasks = free_ram_gb // img_plane_size_gb
        if num_of_concurrent_tasks < 1:
            print(
                "WARNING: Image plane size is larger than memory. "
                + "Will try to run large jobs in a single process"
            )
            num_of_concurrent_tasks = 1
        if num_of_concurrent_tasks > num_cpus:
            num_of_concurrent_tasks = num_cpus
        return num_of_concurrent_tasks

    def _get_tile_shape_no_overlap(self) -> Tuple[int, int]:
        overlap_y = self._get_tile_overlap_y_in_px()
        overlap_x = self._get_tile_overlap_x_in_px()
        tile_height_with_overlap = self._std_meta["TileHeight"]
        tile_width_with_overlap = self._std_meta["TileWidth"]
        tile_height = tile_height_with_overlap - overlap_y
        tile_width = tile_width_with_overlap - overlap_x
        return tile_height, tile_width


def write_pipeline_config(out_path: Path, pipeline_config: dict):
    with open(out_path, "w") as s:
        json.dump(pipeline_config, s, indent=4)


def main(path_to_dataset: Path):
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s - %(message)s")
    logger = logging.getLogger(__name__)

    config_creator = ConfigCreator()
    config_creator.dataset_dir = path_to_dataset
    config_creator.find_raw_data_dir()
    config_creator.read_metadata()
    pipeline_config = config_creator.create_config()

    out_path = Path("pipelineConfig.json")
    logger.info("Writing pipeline config")
    write_pipeline_config(out_path, pipeline_config)
    logger.info(f"Written pipeline config to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect information required to perform analysis of a CODEX dataset."
    )
    parser.add_argument(
        "--path_to_dataset",
        help="Path to directory containing raw data subdirectory (with with cycle and region numbers).",
        type=Path,
    )
    args = parser.parse_args()
    main(args.path_to_dataset)
