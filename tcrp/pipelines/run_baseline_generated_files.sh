#!/bin/bash
#SBATCH -p himem
#SBATCH -t 24:00:00
#SBATCH --mem=64GB
#SBATCH -J baseline_tcrp_run  
#SBATCH -c 4    
#SBATCH -N 1 

cd /cluster/projects/schwartzgroup/almas/tcrp-reproduce/output/210803_drug-baseline-models/baseline_cmd
eval "$(conda shell.bash hook)"
conda activate tcrp_env
ls run_baseline_drugs*.sh | awk '{k = "sbatch "$0""; system(k); print(k)}'