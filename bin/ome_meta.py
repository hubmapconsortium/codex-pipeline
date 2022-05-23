import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import lxml.etree
import numpy as np
import tifffile as tif
from aicsimageio.vendor.omexml import OMEXML

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s - %(message)s")
logger = logging.getLogger(__name__)


class OMEMetaCreator:
    def __init__(self):
        self.pipeline_config: dict = dict()
        self.path_to_sample_img: Path = Path("")
        self.channel_names: List[str] = []
        self.nucleus_channel: str = "DAPI"
        self.cell_channel: str = "CD45"
        # Create a template OME-XML object.

    def create_ome_meta(self) -> str:
        omeXml = OMEXML()

        img_shape, img_dtype = self._get_img_attributes()

        size_t = 1
        size_c = len(self.channel_names)
        size_z = 1  # self.pipeline_config["num_z_planes"]
        # Populate it with image metadata.
        omeXml.image().Pixels.set_SizeT(size_t)
        omeXml.image().Pixels.set_SizeC(size_c)
        omeXml.image().Pixels.set_SizeZ(size_z)
        omeXml.image().Pixels.set_SizeY(img_shape[0])
        omeXml.image().Pixels.set_SizeX(img_shape[1])
        omeXml.image().Pixels.set_PixelType(str(img_dtype))
        omeXml.image().Pixels.set_DimensionOrder("XYZCT")
        omeXml.image().Pixels.channel_count = len(self.channel_names)
        omeXml.image().Pixels.set_PhysicalSizeX(self.pipeline_config["lateral_resolution"])
        omeXml.image().Pixels.set_PhysicalSizeY(self.pipeline_config["lateral_resolution"])

        for i in range(0, len(self.channel_names)):
            omeXml.image().Pixels.Channel(i).Name = self.channel_names[i]
            omeXml.image().Pixels.Channel(i).ID = "Channel:0:" + str(i)

        omeXml = self._add_pixel_size_units(omeXml)
        omeXml = self._add_structured_annotations(omeXml, self.nucleus_channel, self.cell_channel)
        omeXml = self._add_tiffdata(omeXml, size_t, size_c, size_z)
        return omeXml.to_xml(encoding="utf-8", indent="    ")

    def _add_pixel_size_units(self, omeXml: OMEXML):
        encoding = "utf-8"
        omeXmlRoot = lxml.etree.fromstring(omeXml.to_xml(encoding=encoding).encode(encoding))

        namespace_prefix = omeXmlRoot.nsmap[None]
        image_node = omeXmlRoot.find(f"{{{namespace_prefix}}}Image")
        pixels_node = image_node.find(f"{{{namespace_prefix}}}Pixels")

        pixels_node.set("PhysicalSizeXUnit", "nm")
        pixels_node.set("PhysicalSizeYUnit", "nm")

        omexml_with_pixel_units = OMEXML(xml=lxml.etree.tostring(omeXmlRoot))
        return omexml_with_pixel_units

    def _add_structured_annotations(
        self, omexml: OMEXML, nucleus_channel: str, cell_channel: str
    ) -> OMEXML:
        """
        Will add this, to the root, after Image node
        <StructuredAnnotations>
        <XMLAnnotation ID="Annotation:0">
            <Value>
                <OriginalMetadata>
                    <Key>SegmentationChannels</Key>
                    <Value>
                        <Nucleus>DAPI-02</Nucleus>
                        <Cell>CD45</Cell>
                    </Value>
                </OriginalMetadata>
            </Value>
        </XMLAnnotation>
        </StructuredAnnotations>
        """

        structured_annotation = ET.Element("StructuredAnnotations")
        annotation = ET.SubElement(structured_annotation, "XMLAnnotation", {"ID": "Annotation:0"})
        annotation_value = ET.SubElement(annotation, "Value")
        original_metadata = ET.SubElement(annotation_value, "OriginalMetadata")
        segmentation_channels_key = ET.SubElement(
            original_metadata, "Key"
        ).text = "SegmentationChannels"
        segmentation_channels_value = ET.SubElement(original_metadata, "Value")
        ET.SubElement(segmentation_channels_value, "Nucleus").text = nucleus_channel
        ET.SubElement(segmentation_channels_value, "Cell").text = cell_channel
        sa_str = ET.tostring(structured_annotation, encoding="utf-8").decode("utf-8")

        omexml_str = omexml.to_xml(encoding="utf-8")
        closed_image_node_position = omexml_str.find("</Image>") + len("</Image>")
        omexml_str_with_sa = (
            omexml_str[:closed_image_node_position]
            + sa_str
            + omexml_str[closed_image_node_position:]
        )
        return OMEXML(omexml_str_with_sa)

    def _get_img_attributes(self) -> Tuple[Tuple[int, int], np.dtype]:
        with tif.TiffFile(self.path_to_sample_img) as TF:
            img_shape = TF.series[0].shape
            img_dtype = TF.series[0].dtype
        return img_shape, img_dtype

    def _add_tiffdata(self, omeXml: OMEXML, size_t, size_c, size_z):
        tiffdata_elements = []
        ifd = 0
        for t in range(0, size_t):
            for c in range(0, size_c):
                for z in range(0, size_z):
                    tiffdata_attrib = {
                        "FirstT": str(t),
                        "FirstC": str(c),
                        "FirstZ": str(z),
                        "IFD": str(ifd),
                    }
                    tiffdata = lxml.etree.Element("TiffData", tiffdata_attrib)
                    tiffdata_elements.append(tiffdata)
                    ifd += 1

        encoding = "utf-8"
        omeXmlRoot = lxml.etree.fromstring(omeXml.to_xml(encoding=encoding).encode(encoding))

        namespace_prefix = omeXmlRoot.nsmap[None]
        image_node = omeXmlRoot.find(f"{{{namespace_prefix}}}Image")
        pixels_node = image_node.find(f"{{{namespace_prefix}}}Pixels")
        for td in tiffdata_elements:
            pixels_node.append(td)
        omexml_with_tiffdata = OMEXML(xml=lxml.etree.tostring(omeXmlRoot))
        return omexml_with_tiffdata
