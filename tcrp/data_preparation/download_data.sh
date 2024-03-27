#!/bin/bash

# Data downloading script to reproduce TCRP results 
data_file=../data/references/original_data/GDSC/ 
mkdir -p "$data_file"
cd "$data_file"

# 1 - Download Cell_line_RMA_proc_basalExp.txt
wget -nc -O Cell_line_RMA_proc_basalExp.txt.zip "http://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Data/preprocessed/Cell_line_RMA_proc_basalExp.txt.zip" &&
unzip Cell_line_RMA_proc_basalExp.txt.zip &&
rm Cell_line_RMA_proc_basalExp.txt.zip

# 2 - Download HUGO_protein-coding_gene.txt
wget -nc -O HUGO_protein-coding_gene.txt "https://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/locus_types/gene_with_protein_product.txt" 

# 3 - Download v17_fitted_dose_response.csv
wget -nc -O v17_fitted_dose_response.csv "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-6.0/v17_fitted_dose_response.xlsx"

# 4 - Download WES_variants.xlsx - converted to csv manually 
wget -nc -O WES_variants.xlsx "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-6.0/WES_variants.xlsx"
# switched tab on excel sheet and saved as WES_variants.csv manually 

# 5 - Download Cell_Lines_Details.csv
wget -nc -O Cell_Lines_Details.csv "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-6.0/Cell_Lines_Details.xlsx"

# 6 - Download Screened_Compounds.csv 
wget -nc -O Screened_Compounds.csv "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/release-6.0/Screened_Compounds.xlsx" 
# convert from .csv to .txt due to character decoding error in encountered in process_sanger_drug_cell_line.ipynb 
sed 's/,/\t/g' Screened_Compounds.csv > Screened_Compounds.tsv

# 7 - Download allComplexes.txt from https://mips.helmholtz-muenchen.de/corum/#download 
wget -nc -O allComplexes.txt.zip "https://mips.helmholtz-muenchen.de/corum/download/releases/current/allComplexes.txt.zip" --no-check-certificate
unzip allComplexes.txt.zip 

# 8 - Download pathwaycommons file 
wget -nc -O PathwayCommons9.All.hgnc.txt.gz "http://www.pathwaycommons.org/archives/PC2/v9/PathwayCommons9.All.hgnc.txt.gz"
gunzip PathwayCommons9.All.hgnc.txt.gz

