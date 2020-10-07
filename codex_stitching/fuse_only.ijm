// fuse dataset, save as TIFF
run("Fuse dataset ...",
    "select={path_to_xml_file}" +
    " process_angle=[All angles]" +
    " process_channel=[All channels]" +
    " process_illumination=[All illuminations]" +
    " process_tile=[All tiles]" +
    " process_timepoint=[All Timepoints]" +
    " bounding_box=[All Views]" +
    " downsampling=1" +
    " pixel_type=[16-bit unsigned integer]" +
    " interpolation=[Linear Interpolation]" +
    " image=[Precompute Image]" +
    " interest_points_for_non_rigid=[-= Disable Non-Rigid =-]" +
    " blend produce=[Each timepoint & channel]" +
    " fused_image=[Save as (compressed) TIFF stacks]" +
    " output_file_directory={out_dir}");
