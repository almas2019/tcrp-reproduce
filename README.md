# tcrp-reproduce

This repository hosts the code used in our attempt to recreate the results from challenge 1 of “Few-shot learning creates predictive models of drug response that translate from high-throughput screens to individual patients” (Ma et al., 2021) in fulfillment of the final project for the Winter 2024 course MBP 1413H - Biomedical Applications of Artificial Intelligence. The repository was forked from the authors’ refactored codebase: https://github.com/shfong/tcrp-reproduce and modified in an attempt to improve reproducibility. 

## Instructions for a complete run
To reproduce our refactored results, follow the following instructions: 

### 1. Clone this repository 

### 2. Create a Conda environment:

Create and activate a new Conda environment by running the following commands: 

`conda env create -f environment_w_jupyter.yml`  
`conda activate tcrp_env` 

### 3. Gather input data  

The script /data_preparation/download_data.sh downloads all data files required. After running this script, all files will be converted to the required file format to run process_sanger_drug_cell_line.ipynb, with the exception of the conversion of WES_variants.xlsx to WES_variants.csv, which should be performed manually in step C.  

A. Make the script executable:  
`chmod +x download_data.sh`  

B. Run the script:  
`bash ./download_data.sh`  

C. Convert WES_variants.xlsx to WES_variants.csv:  
Open WES_variants.xlsx in Excel. Switch to the “WES_variants” tab at the bottom of the Excel sheet, and save to csv format through File -> Save as -> CSV UTF-8 (Comma delimited) (.csv). 

Moreover, as an alternative to running download_data.sh, we provide a Google Drive link to all raw input files containing all files downloaded in download_data.sh: https://drive.google.com/drive/folders/1fhsu0zxsSmbCV6w01tMqzeCw0fpx1wYX?usp=sharing 

### 4. Preprocess data 

Run the jupyter notebook /data_preparation/process_sanger_drug_cell_line.ipynb interactively. Check the input data is properly placed in a data/ folder in the parent directory of the data_preparation folder, as well as the network/ folder for the allComplexes.txt, core_inbiomap.sif and PathwayCommons9.txt file.

### 5. Run Models 

1. Edit paths to match your data and fewshot path in the following scripts:
   - `/tcrp/pipelines/generate_baseline_job_cv.py` --> the job directory path
   - `/tcrp/pipelines/generate_fewshot_samples.py`  --> the job directory path
2. Edit code to match your specific GPU and compute node information (i.e. gpu/cpu memory, account, node name, etc):
   - lines 95-102 in `/tcrp/pipelines/generate_MAML_job_cv.py`
   - lines  79-87 in `/tcrp/pipelines/generate_baseline_job_cv.py`
3. Change run mode to `tcrp` in `/tcrp/pipelines/prepare_complete_run.py`
4. Edit the file `/tcrp/pipelines/complete_run.sh` to compute node specifications and paths, then run
   - this will run `prepare_complete_run.sh` and create your output folder
   - this will also create your `MAML_cmd` folder which contains the run_MAML_drugs_*.sh
5. Edit the file `/tcrp/pipelines/run_tcrp_generated_files.sh` to compute node specifications and paths, then run
   - this will run the run_MAML_drugs_*.sh which runs tcrp model on the tissues and drugs
   - this will also create the fewshot_data folder
6. Edit the fewshot_data_path to the location of the fewshot_data folder just created in  `/tcrp/pipelines/prepare_complete_run.py`
7. Change run mode to `baseline` in `/tcrp/pipelines/prepare_complete_run.py`
8. Run `/tcrp/pipelines/complete_run.sh` again
   - this will create your baseline_cmd folder which contains a `run_baseline_drugs*.sh` script
9. Edit the file `/tcrp/pipelines/run_baseline_generated_files.sh` to compute node specifications and paths, then run `run_baseline_generated_files.sh`
   - this will run `run_baseline_drugs*.sh` that runs the conventional machine learning models on your data
10. Optional run either `compress_output_lz4.sh` or `compress_output.sh` if you would like to compress your output folder

Example of an output folder that was generated: https://drive.google.com/file/d/1RQD2xmNw6YXNVgTMk894zoHh6C4dCbF7/view?usp=drive_link 
### 6. Compare models

Before beginning these steps, make sure the output folder obtained from the model training is placed in the project’s home folder. Then, go to the model_comparisons folder and execute the files as follows:
1. Open and execute the 1-gather-baselines Jupyter notebook interactively. Make sure the path to the baseline performances is set properly according to the project structure (third block in the notebook). This will produce a plot in the final code block.
2. Execute the Python script called gather-results.py. Check the logs_directory path is properly set to the folder in the project (line 112). Alternatively, it is possible to run the 2-gather-results notebook interactively, but this could lead to errors if multiprocessing is not supported in the machine. The results from this step will be saved in the folder from where the Python script was executed.
3. If necessary, move the files generated in the last step (tcrp_all_log_paths, tcrp_all_results, and tcrp_fewshot-test-correlations-corrected) into the model_comparisons folder. Then, run interactively the 3-analyze-all-tcrp-performance notebook to get the comparison plot.



## OLD README: 

This part of the pipeline is not automated yet. The raw data will need to be downloaded from DepMap, and the transformed data are generated in with a jupyter notebook `tcrp/data_preparation/process_sanger_drug_cell_line.ipynb`. This notebook will generate a series of pickled files and numpy compressed files that the following steps will be dependent on. 

### TCRP complete run

The code should all be contained in `prepare_complete_run.py`. This script will create a directory that contains all of commands to sweep through all hyperpameter for all of the specific drugs. The drugs analyzed correspond to the pickle files in `data/cell_line_lists`. Code to generate the pickled files still need to be included in this repository. Feel free to edit the `run_name` variable to change the run name. 

After the code is generated, the slurm submission scrips are created in `output/{RUN NAME}/MAML_cmd`. To submit all of the slurm scripts you can run the following: 
```ls run_MAML_drugs*.sh | awk '{k = "sbatch "$0""; system(k); print(k)}'```

### Baseline run

Edit `prepare_complete_run.py`. Change `run_mode` variable to `baseline` to run `generate_baseline_jobs.py`. In addition, point to the correct `fewshot_data_path`. This is a directory that was created in the tcrp complete run. It's simply the fewshot training and testing dataset that was used in the complete run.


## Parsing results
The results are all embedded as logs in `output/{RUN NAME}/run-logs/{DRUG}/{TISSUE}`. The log will specify the selected epoch for that hyperparameter and the correspond test performance. Additional code will be needed to gather the best performance to select the final performance for task.

## Reusability efforts

Emily So, from the Haibe-Kains lab, has reproduced and extended TCRP. Their repository can be found [here](https://github.com/bhklab/TCRP_Reusability_Report). A runnable version is also available [here](https://codeocean.com/capsule/8411716/tree/v2).
