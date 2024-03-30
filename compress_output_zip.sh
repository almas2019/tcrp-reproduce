#!/bin/bash
#SBATCH -p himem
#SBATCH -t 24:00:00
#SBATCH --mem=64GB
#SBATCH -J compress_output_zip      
#SBATCH -c 4    
#SBATCH -N 1 

cd /cluster/projects/schwartzgroup/almas/tcrp-reproduce/
zip -r output.zip output