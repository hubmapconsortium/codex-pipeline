// define dataset
run("BigStitcher",
    "select=define" +
    " define_dataset=[Manual Loader (Bioformats based)]" +
    " project_filename={path_to_xml_file}" +
    " multiple_timepoints=[NO (one time-point)]" +
    " multiple_channels=[NO (one channel)]" +
    " _____multiple_illumination_directions=[NO (one illumination direction)]" +
    " multiple_angles=[NO (one angle)]" +
    " multiple_tiles=[YES (one file per tile)]" +
    " image_file_directory={img_dir}" +
    " image_file_pattern={pattern}" +
    " timepoints_=1" +
    " tiles_={num_tiles}" +
    " move_tiles_to_grid_(per_angle)?=[Move Tile to Grid (Macro-scriptable)]" +
    " grid_type={tiling_mode}" +
    " tiles_x={num_tiles_x}" +
    " tiles_y={num_tiles_y}" +
    " tiles_z=1" +
    " overlap_x_(%)={overlap_x}" +
    " overlap_y_(%)={overlap_y}" +
    " overlap_z_(%)={overlap_z}" +
    " calibration_type=[Same voxel-size for all views]" +
    " calibration_definition=[User define voxel-size(s)]" +
    " imglib2_data_container=[ArrayImg (faster)]" +
    " pixel_distance_x={pixel_distance_x}" +
    " pixel_distance_y={pixel_distance_y}" +
    " pixel_distance_z={pixel_distance_z}" +
    " pixel_unit=um");

// calculate pairwise shifts
run("Calculate pairwise shifts ...",
    "select={path_to_xml_file}" +
    " process_angle=[All angles]" +
    " process_channel=[All channels]" +
    " process_illumination=[All illuminations]" +
    " process_tile=[All tiles]" +
    " process_timepoint=[All Timepoints]" +
    " method=[Phase Correlation]");

// filter shifts with 0.7 corr. threshold
run("Filter pairwise shifts ...",
    "select={path_to_xml_file}" +
    " filter_by_link_quality" +
    " min_r=0.7" +
    " max_r=1" +
    " max_shift_in_x=0" +
    " max_shift_in_y=0" +
    " max_shift_in_z=0" +
    " max_displacement=0");

// do global optimization
run("Optimize globally and apply shifts ...",
    "select={path_to_xml_file}" +
    " process_angle=[All angles]" +
    " process_channel=[All channels]" +
    " process_illumination=[All illuminations]" +
    " process_tile=[All tiles]" +
    " process_timepoint=[All Timepoints]" +
    " relative=2.500" +
    " absolute=3.500" +
    " global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles]" +
    " fix_group_0-0,");


// quit after we are finished
run("Quit");
eval("script", "System.exit(0);");
