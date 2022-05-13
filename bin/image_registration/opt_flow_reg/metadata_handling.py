import re
import xml.etree.ElementTree as ET
from io import StringIO

XML = ET.ElementTree


def str_to_xml(xmlstr: str):
    """Converts str to xml and strips namespaces"""
    it = ET.iterparse(StringIO(xmlstr))
    for _, el in it:
        _, _, el.tag = el.tag.rpartition("}")
    root = it.root
    return root


def extract_channel_info(xml: XML):
    channels = xml.find("Image").find("Pixels").findall("Channel")
    channel_names = [ch.get("Name") for ch in channels]
    channel_ids = [ch.get("ID") for ch in channels]
    channel_fluors = []
    for ch in channels:
        if "Fluor" in ch.attrib:
            channel_fluors.append(ch.get("Fluor"))
    return channels, channel_names, channel_ids, channel_fluors


def strip_cycle_info(name):
    ch_name = re.sub(r"^(c|cyc|cycle)\d+(\s+|_)", "", name)  # strip start
    ch_name2 = re.sub(r"(-\d+)?(_\d+)?$", "", ch_name)  # strip end
    return ch_name2


def find_where_ref_channel(ome_meta: str, ref_channel: str):
    """Find if reference channel is in fluorophores or channel names and return them"""

    channels, channel_names, channel_ids, channel_fluors = extract_channel_info(
        str_to_xml(ome_meta)
    )
    ref_ch = strip_cycle_info(ref_channel)
    # strip cycle id from channel name and fluor name
    if channel_fluors != []:
        fluors = [strip_cycle_info(fluor) for fluor in channel_fluors]
    else:
        fluors = None
    names = [strip_cycle_info(name) for name in channel_names]

    # check if reference channel is present somewhere
    if ref_ch in names:
        matches = names
    elif fluors is not None and ref_ch in fluors:
        matches = fluors
    else:
        if fluors is not None:
            message = (
                "Incorrect reference channel "
                + str(ref_ch)
                + ". "
                + "Available channel names: {names}, fluors: {fluors}"
            )
            raise ValueError(
                message.format(names=", ".join(set(names)), fluors=", ".join(set(fluors)))
            )
        else:
            message = (
                "Incorrect reference channel "
                + str(ref_ch)
                + ". "
                + "Available channel names: {names}"
            )
            raise ValueError(message.format(names=", ".join(set(names))))

    return matches


def get_cycle_composition(xmlstr: str, ref_channel: str):
    matches = find_where_ref_channel(xmlstr, ref_channel)

    # encode reference channels as 1 other 0

    channels = []
    for i, channel in enumerate(matches):
        if channel == strip_cycle_info(ref_channel):
            channels.append(1)
        else:
            channels.append(0)

    cycle_composition = []
    for ch in channels:
        cycle_composition.append(ch)
        if sum(cycle_composition) == 2:
            break

    first_ref_position = cycle_composition.index(1)
    second_ref_position = None
    for i, channel_type in enumerate(cycle_composition):
        if channel_type == 1 and i != first_ref_position:
            second_ref_position = i

    if second_ref_position is None:
        raise ValueError("Reference channel in second cycle is not found")

    cycle_size = second_ref_position - first_ref_position
    ncycles = len(channels) // cycle_size

    return cycle_size, ncycles, first_ref_position
