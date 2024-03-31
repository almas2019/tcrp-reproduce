#!/bin/bash
#SBATCH -p himem
#SBATCH -t 24:00:00
#SBATCH --mem=64GB
#SBATCH -J lz4_compress    
#SBATCH -c 4    
#SBATCH -N 1 

cd /cluster/projects/schwartzgroup/almas/tcrp-reproduce/
#tar cvf output.tar.lz4 -I lz4 output
tar cvf - output | lz4 - output_again.tar.lz4
