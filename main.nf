#!/usr/bin/env nextflow

params.datasetsFile = "datasets.txt"

Channel
    .fromPath( params.datasetsFile )
    .splitCsv( header: false, sep: '\t' )
    .map{ row-> tuple( row[0], file( row[1] ) ) }
    .set { datasets_ch }


/* Pipeline steps:
    1. Set up data directory.
    2. Create YAML config.
    3. Run Cytokit processor.
 */


process setup_data_directory {
    
    label "cpu"

    input:
        set datasetID, file( configPath ) from datasets_ch
    
    output:
        file "${datasetID}_data" into data_dirs_ch
        val datasetID into datasets_with_data_dir_ch
        file configPath into config_paths_ch

    shell:
        '''
        $CODEX_PIPELINE_CODEBASE/bin/setup_data_directory.py !{configPath} !{datasetID}_data
        '''
}

process create_yaml_config {

    label "cpu"
    
    input:
        val datasetID from datasets_with_data_dir_ch
        file configPath from config_paths_ch

    output:
        val datasetID into datasets_with_yaml_ch
        file "${datasetID}_experiment.yaml" into yaml_files_ch

    shell:
        '''
        $CODEX_PIPELINE_CODEBASE/bin/create_cytokit_config.py !{datasetID} !{configPath}
        '''
}

process run_cytokit_processor {

    label "gpu"

    input:
        val datasetID from datasets_with_yaml_ch
        file yaml_config from yaml_files_ch
        file data_dir from data_dirs_ch

    output:
        file "output" into cytokit_output_dir_ch

    shell:
        '''
        mkdir output
        
        source $CODEX_PIPELINE_CODEBASE/conf/cytokit_env
        conda activate cytokit

        cytokit processor run_all --config-path=!{yaml_config} --data-dir=!{data_dir} --output-dir=output
        '''
}


process generate_ome_tiffs {

    label "cpu"

    input:
        file cytokit_output_dir from cytokit_output_dir_ch

    output:
        file "processor/tile/ome-tiff/*.ome.tiff" into image_ome_tiffs_ch
        file "cytometry/tile/ome-tiff/*.ome.tiff" into segm_ome_tiffs_ch

    shell:
        '''
        pushd !{cytokit_output_dir}/processor/tile
        
        echo "Generating OME-TIFFs in !{cytokit_output_dir}/processor/tile/ome-tiff"
        
        if [[ -d "ome-tiff" ]] ; then
            rm -r ome-tiff
        fi

        mkdir ome-tiff

        for f in $( ls *.tif ); do 
            fname=$( echo $f | awk -F'.' '{print $1}' )
            /opt/bftools/bfconvert $f ome-tiff/${fname}.ome.tiff
        done
        
        popd

        pushd !{cytokit_output_dir}/cytometry/tile

        echo "Generating OME-TIFFs in !{cytokit_output_dir}/cytometry/tile/ome-tiff"

        if [[ -d "ome-tiff" ]] ; then
            rm -r ome-tiff
        fi
        
        mkdir ome-tiff

        for f in $( ls *.tif ); do 
            fname=$( echo $f | awk -F'.' '{print $1}' )
            /opt/bfconvert/bfconvert $f ome-tiff/${fname}.ome.tiff
        done
        '''
}
