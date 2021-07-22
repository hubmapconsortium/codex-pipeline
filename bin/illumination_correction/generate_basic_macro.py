from pathlib import Path


def fill_in_basic_macro_template(path_to_stack: Path, out_dir: Path) -> str:
    macro_template = """
    run("BaSiC Mod",
        "input_stack={path_to_stack}" +
        " flat-field_image_path=[]" +
        " dark-field_image_path=[]" +
        " output_dir={out_dir}" +
        " shading_estimation=[Estimate shading profiles]" +
        " shading_model=[Estimate flat-field only (ignore dark-field)]" +
        " setting_regularisation_parameters=Automatic" +
        " temporal_drift=Ignore" +
        " correction_options=[Compute shading only]" +
        " lambda_flat=0.500" +
        " lambda_dark=0.500");
    
    run("Quit");
    eval("script", "System.exit(0);");
    """
    # [Compute shading only, Compute shading and correct images]
    basic_macro = macro_template.format(
        path_to_stack=str(path_to_stack.absolute()), out_dir=str(out_dir.absolute())
    )
    return basic_macro


def save_macro(out_path: Path, macro: str):
    with open(out_path, "w", encoding="utf-8") as s:
        s.write(macro)
