import re
import xml.etree.ElementTree as ET
from io import StringIO
from typing import List

XML = ET.ElementTree


class DatasetStructure:
    def __init__(self):
        self._ref_ch = "DAPI"
        self.ome_meta_str = ""

    @property
    def ref_channel_name(self) -> str:
        return self._ref_ch

    @ref_channel_name.setter
    def ref_channel_name(self, channel_name: str):
        self._ref_ch = self._strip_cycle_info(channel_name)

    def get_dataset_structure(self):
        ome_xml = self._str_to_xml(self.ome_meta_str)
        channel_names, channel_fluors, nchannels, nzplanes = self._extract_channel_info(ome_xml)
        channel_names_cleaned, ref_ids = self._find_where_ref_channel(
            channel_names, channel_fluors
        )
        return self._get_stack_structure(ref_ids, nchannels, nzplanes)

    def _str_to_xml(self, xmlstr: str):
        """Converts str to xml and strips namespaces"""
        it = ET.iterparse(StringIO(xmlstr))
        for _, el in it:
            _, _, el.tag = el.tag.rpartition("}")
        root = it.root
        return root

    def _extract_channel_info(self, ome_xml: XML):
        channels = ome_xml.find("Image").find("Pixels").findall("Channel")
        channel_names = [ch.get("Name") for ch in channels]
        channel_fluors = []
        for ch in channels:
            if "Fluor" in ch.attrib:
                channel_fluors.append(ch.get("Fluor"))
        image_attribs = ome_xml.find("Image").find("Pixels").attrib
        nchannels = int(image_attribs.get("SizeC", 1))
        nzplanes = int(image_attribs.get("SizeZ", 1))
        return channel_names, channel_fluors, nchannels, nzplanes

    def _strip_cycle_info(self, name):
        ch_name = re.sub(r"^(c|cyc|cycle)\d+(\s+|_)", "", name)  # strip start
        ch_name2 = re.sub(r"(-\d+)?(_\d+)?$", "", ch_name)  # strip end
        return ch_name2

    def _filter_ref_channel_ids(self, channels: List[str]) -> List[int]:
        ref_ids = []
        for _id, ch in enumerate(channels):
            if re.match(self._ref_ch, ch, re.IGNORECASE):
                ref_ids.append(_id)
        return ref_ids

    def _find_where_ref_channel(self, channel_names: List[str], channel_fluors: List[str]):
        """Find if reference channel is in fluorophores or channel names and return them"""
        # strip cycle id from channel name and fluor name
        if channel_fluors != []:
            fluors = [
                self._strip_cycle_info(fluor) for fluor in channel_fluors
            ]  # remove cycle name
        else:
            fluors = None
        names = [self._strip_cycle_info(name) for name in channel_names]

        # check if reference channel is present somewhere
        if self._ref_ch in names:
            cleaned_channel_names = names
        elif fluors is not None and self._ref_ch in fluors:
            cleaned_channel_names = fluors
        else:
            if fluors is not None:
                msg = (
                    f"Incorrect reference channel {str(self._ref_ch)}. "
                    + f"Available channel names: {str(set(names))}, fluors: {str(set(fluors))}"
                )
                raise ValueError(msg)
            else:
                msg = (
                    f"Incorrect reference channel {str(self._ref_ch)}. "
                    + f"Available channel names: {str(set(names))}"
                )
                raise ValueError(msg)
        ref_ids = self._filter_ref_channel_ids(cleaned_channel_names)
        return cleaned_channel_names, ref_ids

    def _get_stack_structure(self, ref_ids, nchannels, nzplanes):
        nchannels_per_cycle = ref_ids[1] - ref_ids[0]
        ref_ch_position_in_cyc = ref_ids[0]
        ncycles = nchannels // nchannels_per_cycle

        stack_structure = dict()
        tiff_page = 0
        for cyc in range(0, ncycles):
            img_structure = dict()
            for ch in range(0, nchannels_per_cycle):
                img_structure[ch] = dict()
                for z in range(0, nzplanes):
                    img_structure[ch][z] = tiff_page
                    tiff_page += 1
            stack_structure[cyc] = dict()
            stack_structure[cyc]["img_structure"] = img_structure
            stack_structure[cyc]["ref_channel_id"] = ref_ch_position_in_cyc
        return stack_structure
