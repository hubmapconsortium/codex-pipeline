#!/usr/bin/env python3

from aicsimageio.readers import ome_tiff_reader
import argparse
import csv
import logging
from multiprocessing import Pool
from os import walk
from pathlib import Path
import re
from typing import Dict, List
import xml.etree.ElementTree as ET

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-7s - %(message)s'
)
logger = logging.getLogger(__name__)

OMEXML_NAMESPACES = {
        "ome" : "http://www.openmicroscopy.org/Schemas/ome/2013-06",
        "roi" : "http://www.openmicroscopy.org/Schemas/ROI/2016-06"
}

"""
Given a directory and a file extension/suffix, return a list of the files in
the directory with the given extension, that match Cytokit's region_tileX_tileY
naming pattern.
"""
def collect_file_list( 
        directory: Path, 
        suffix: str 
    ) -> List[ Path ] :

    fileList = []
    
    tileNamePattern = re.compile( r'^R\d{3}_X\d{3}_Y\d{3}' + suffix + '$' )

    for dirpath, dirnames, filenames in walk( directory ) :
        for filename in filenames :
            if tileNamePattern.match( filename ) :
                fileList.append( directory / filename )

    if len( fileList ) == 0 :
        raise ValueError( 
            f"No files found matching tile naming pattern with suffix {suffix}"
        )
    
    return fileList

"""
Given a list of OME-TIFF files plus a directory containing Cytokit's cytometry
statistics CSV files, return a dictionary containing each "tile name" (region
plus tile coordinates) pointing to the corresponding OME-TIFF file and Cytokit
CSV file.
"""
def collect_files_by_tile( 
        ometiffFileList: List[ Path ],
        cytometryStatsDir: Path
    ) -> Dict :

    tileNamesAndFiles = {}

    for ometiffFile in ometiffFileList :
        
        tileName = str( ometiffFile.name ).replace("".join(ometiffFile.suffixes), "")
        
        tileNamesAndFiles[ tileName ] = { "ometiff" : ometiffFile }
        
        # Find the matching CSV file for this tile.
        tileStatsCsvFilename = Path( tileName + ".csv" )
        tileStatsCsvPath = cytometryStatsDir / tileStatsCsvFilename
        
        if not tileStatsCsvPath.exists() :
            raise ValueError(
                f"No CSV file found matching tile name {tileName}"
            )
        else :
            tileNamesAndFiles[ tileName ][ "csv" ] = tileStatsCsvPath

    return tileNamesAndFiles

"""
Given a dictionary of tile names pointing to corresponding OME-TIFF and Cytokit
CSV, plus an output directory, write a new CSV file to the output directory
containing cell IDs, centroid x,y,z coordinates, and cell shape polygons for
each tile.
"""
def create_cellshapes_csv_files( 
        tileNamesAndFiles: Dict, 
        cellShapesCsvDir: Path 
    ) :
    
    for tileName in tileNamesAndFiles :
        
        logger.info( f"Creating cell shapes CSV for {tileName} ..." )

        cytokitCsvFilename = tileNamesAndFiles[ tileName ][ "csv" ]
        ometiffFilename = tileNamesAndFiles[ tileName ][ "ometiff" ]
        
        # Read Cytokit's cytometric data from CSV.
        cytokitStats = {}
        with open( cytokitCsvFilename, newline='' ) as cytokitCsvFile :
            csvReader = csv.DictReader( cytokitCsvFile )
            for row in csvReader :
                cytokitStats[ row[ "id" ] ] = {
                        "x" : row[ "x" ],
                        "y" : row[ "y" ],
                        "z" : row[ "z" ]
                }
        cytokitCsvFile.close()
        
        # Read OME-XML from OME-TIFF.
        omexmlObj = None
        with ome_tiff_reader.OmeTiffReader( ometiffFilename ) as ometiffReader :
            omexmlObj = ometiffReader.metadata
        ometiffReader.close()

        omexml = ET.fromstring( omexmlObj.to_xml() )
        
        cellShapesCsvFilename = cellShapesCsvDir / Path( tileName + ".shape.csv" )
        
        with open( cellShapesCsvFilename, 'w', newline='' ) as csvFile :
            csvWriter = csv.writer( csvFile, quoting = csv.QUOTE_MINIMAL )
            csvWriter.writerow( [ "id", "object", "x", "y", "z", "shape" ] )
            for roi in omexml.findall( "roi:ROI", OMEXML_NAMESPACES ) :
                for union in roi.findall( "roi:Union", OMEXML_NAMESPACES ) :
                    for polygon in union.findall( "roi:Polygon", OMEXML_NAMESPACES ) :
                        polygonAttributes = polygon.attrib
                        cellID = polygonAttributes[ "ID" ].replace( "Shape:", "" )
                        csvWriter.writerow(
                                [
                                    cellID,
                                    "cell",
                                    cytokitStats[ cellID ][ "x" ],
                                    cytokitStats[ cellID ][ "y" ],
                                    cytokitStats[ cellID ][ "z" ],
                                    polygonAttributes[ "Points" ]
                                ]
                        )
        csvFile.close()

        logger.info( f"{tileName} cell shape CSV created." )


########
# MAIN #
########
if __name__ == "__main__" :

    parser = argparse.ArgumentParser(
        description=(
            "Read in OME-TIFF files from Cytokit cytometry/tile/ome-tiff"
            "Read Cytokit cytometry/statistics/*.csv"
            "Create CSV containing:"
            "   * cell ID"
            "   * shape/polygon(s)"
            "   * centroid x,y,z"
        ),
    )
    parser.add_argument(
        "cytokit_output_dir",
        help="Path to Cytokit's output directory.",
        type=Path,
    )
    

    args = parser.parse_args()

    output_dir = Path( 'output' )
    output_dir.mkdir( parents = True, exist_ok = True )

    cytometry_ometiff_dir_piece = Path( "cytometry/tile/ome-tiff" )
    cytometry_stats_dir_piece = Path( "cytometry/statistics" )
    cellshapes_dir_piece = Path( "cytometry/statistics/cellshapes" )

    cytometryOmetiffDir = args.cytokit_output_dir / cytometry_ometiff_dir_piece
    cytometryStatsDir = args.cytokit_output_dir / cytometry_stats_dir_piece
    
    # Directory to write new CSVs.
    cellShapesCsvDir = output_dir / cellshapes_dir_piece
    cellShapesCsvDir.mkdir( parents = True, exist_ok = True )
    
    # Find all the OME-TIFF files with segmentation results.
    ometiffFileList = collect_file_list( cytometryOmetiffDir, ".ome.tiff" )
    
    # Get the corresponding Cytokit CSV files and index files by tile name.
    tileNamesAndFiles = collect_files_by_tile( ometiffFileList, cytometryStatsDir )
    
    # Create the new CSVs.
    create_cellshapes_csv_files( tileNamesAndFiles, cellShapesCsvDir )
