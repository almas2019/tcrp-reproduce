#!/bin/bash
#SBATCH -p himem
#SBATCH -t 24:00:00
#SBATCH --mem=64GB
#SBATCH -J tcrp_complete_run      
#SBATCH -c 4    
#SBATCH -N 1 

cd /cluster/projects/schwartzgroup/almas/tcrp-reproduce/tcrp/pipelines/
eval "$(conda shell.bash hook)"
conda activate tcrp_env
python prepare_complete_run.py
# don't do -p all this partition
#sbatch -A schwartzgroup_gpu -p gpu --gres=gpu:1 --constraint gpu32g tg_deconv.sh