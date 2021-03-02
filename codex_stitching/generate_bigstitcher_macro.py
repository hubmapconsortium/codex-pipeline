from pathlib import Path

# from datetime import datetime
from bigstitcher_dataset_meta import generate_dataset_xml


class BigStitcherMacro:
    def __init__(self):
        self.img_dir = Path(".")
        self.out_dir = Path(".")
        self.xml_file_name = "dataset.xml"
        self.pattern = "1_{xxxxx}_Z001.tif"

        # range: 1-5 or list: 1,2,3,4,5
        self.num_tiles = 1

        self.num_tiles_x = 1
        self.num_tiles_y = 1

        self.tile_shape = (1440, 1920)

        # overlap in pixels
        self.overlap_x = 10
        self.overlap_y = 10
        self.overlap_z = 1

        # distance in um
        self.pixel_distance_x = 1
        self.pixel_distance_y = 1
        self.pixel_distance_z = 1

        self.tiling_mode = "snake"
        self.is_snake = True
        self.region = 1

        self.path_to_xml_file = Path(".")

        self.__location = Path(__file__).parent.resolve()

    def generate(self):
        self.make_dir_if_not_exists(self.out_dir)
        self.create_path_to_xml_file()
        self.check_if_tiling_mode_is_snake()

        formatted_macro = self.replace_values_in_macro()
        macro_file_path = self.write_to_temp_macro_file(formatted_macro)

        generate_dataset_xml(
            self.num_tiles_x,
            self.num_tiles_y,
            self.tile_shape,
            self.overlap_x,
            self.overlap_y,
            self.pattern,
            self.path_to_xml_file,
            self.is_snake,
        )

        return macro_file_path

    def make_dir_if_not_exists(self, dir_path: Path):
        if not dir_path.exists():
            dir_path.mkdir(parents=True)

    def create_path_to_xml_file(self):
        self.path_to_xml_file = self.img_dir.joinpath(self.xml_file_name)

    def check_if_tiling_mode_is_snake(self):
        if self.tiling_mode == "snake":
            self.is_snake = True
        else:
            self.is_snake = False

    def convert_tiling_mode(self, tiling_mode):
        if tiling_mode == "snake":
            bigstitcher_tiling_mode = "[Snake: Right & Down      ]"
        elif tiling_mode == "grid":
            bigstitcher_tiling_mode = "[Grid: Right & Down      ]"
        return bigstitcher_tiling_mode

    def replace_values_in_macro(self):
        macro_template = self.estimate_stitch_param_macro_template
        formatted_macro = macro_template.format(
            img_dir=self.path_to_str(self.img_dir),
            out_dir=self.path_to_str(self.out_dir),
            path_to_xml_file=self.path_to_str(self.path_to_xml_file),
            pattern=self.path_to_str(self.img_dir.joinpath(self.pattern)),
            num_tiles=self.make_range(self.num_tiles),
            num_tiles_x=self.num_tiles_x,
            num_tiles_y=self.num_tiles_y,
            overlap_x=self.overlap_x,
            overlap_y=self.overlap_y,
            overlap_z=self.overlap_z,
            pixel_distance_x=self.pixel_distance_x,
            pixel_distance_y=self.pixel_distance_y,
            pixel_distance_z=self.pixel_distance_z,
            tiling_mode=self.convert_tiling_mode(self.tiling_mode),
        )
        return formatted_macro

    def write_to_temp_macro_file(self, formatted_macro):
        file_name = "reg" + str(self.region) + "_bigstitcher_macro.ijm"
        macro_file_path = self.img_dir.joinpath(file_name)
        with open(macro_file_path, "w") as f:
            f.write(formatted_macro)
        return macro_file_path

    def make_range(self, number):
        return ",".join([str(n) for n in range(1, number + 1)])

    def path_to_str(self, path: Path):
        return str(path.absolute().as_posix())

    estimate_stitch_param_macro_template = """
    // calculate pairwise shifts
    run("Calculate pairwise shifts ...",
        "select={path_to_xml_file}" +
        " process_angle=[All angles]" +
        " process_channel=[All channels]" +
        " process_illumination=[All illuminations]" +
        " process_tile=[All tiles]" +
        " process_timepoint=[All Timepoints]" +
        " method=[Phase Correlation]" +
        " show_expert_algorithm_parameters" +
        " downsample_in_x=1" +
        " downsample_in_y=1" +
        " number=5" +
        " minimal=10" +
        " subpixel");

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
    
    run("Quit");
    eval("script", "System.exit(0);");

    """


class FuseMacro:
    def __init__(self):
        self.img_dir = Path(".")
        self.xml_file_name = "dataset.xml"
        self.out_dir = Path(".")
        self.__location = Path(__file__).parent.absolute()

    def generate(self):
        formatted_macro = self.replace_values_in_macro()
        macro_file_path = self.write_to_macro_file_in_channel_dir(self.img_dir, formatted_macro)

    def replace_values_in_macro(self):
        macro_template = self.fuse_macro_template
        formatted_macro = macro_template.format(
            img_dir=self.path_to_str(self.img_dir),
            path_to_xml_file=self.path_to_str(self.img_dir.joinpath(self.xml_file_name)),
            out_dir=self.path_to_str(self.out_dir),
        )
        return formatted_macro

    def write_to_macro_file_in_channel_dir(self, img_dir: Path, formatted_macro: str):
        macro_file_path = img_dir.joinpath("fuse_only_macro.ijm")
        with open(macro_file_path, "w") as f:
            f.write(formatted_macro)
        return macro_file_path

    def path_to_str(self, path: Path):
        return str(path.absolute().as_posix())

    fuse_macro_template = """
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
    
    run("Quit");
    eval("script", "System.exit(0);");

    """
