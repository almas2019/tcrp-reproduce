from pathlib import Path
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import pickle
from multiprocessing import Pool, cpu_count

Metrics = namedtuple("Metrics", ['train_loss', 'train_corr', 'test_loss', 'test_corr'])

def correctly_parse_log_path(filepath): 
    filepath = Path(filepath)
    stem = filepath.stem
    
    array = stem.split("_")
    hyperparameter_str = '_'.join(array[-4:])
    
    best_epoch = -1
    record = False
    
    all_metrics = []
    
    with open(filepath) as f: 
        for line in f.readlines(): 
            if line.startswith("epoch:"): 
                record = True
                epoch = int(line.split(':')[1].strip())
                metrics_container = []
                continue
                
            if line.startswith("Meta update"): 
                record = False
                current_best_epoch = int(line.split('best epoch')[1].strip())
                if current_best_epoch > best_epoch: 
                    best_epoch = current_best_epoch
                    
                all_metrics.append(np.vstack(metrics_container))
                    
            if record: 
                try: 
                    k, metrics = parse_line(line)
                except: 
                    continue
                    
                metrics_container.append(metrics)
    if best_epoch != -1:   
        performance = all_metrics[best_epoch]
        train_loss = performance[:, 0]
        train_corr = performance[:, 1]
        test_loss = performance[:, 2]
        test_corr = performance[:, 3]
        
        return train_loss, train_corr, test_loss, test_corr
    else:
        return None
    
                
def parse_line(line):
    k = int(line.split('Few shot')[0].strip())
    if line.startswith("0 Few shot"):
        line = line.replace('tensor(', '').replace(', device=', ' ')

    line = line.split(':')[1]
    vals = [float(i) for i in line.split()[:4]]
    
    return k, vals


def select_hyperparameter(log_directory): 
    #log_directory = Path(log_directory)
    
    results = {}
    
    train_corrs, test_corrs, names = [], [], []
    print(log_directory)
    for f in log_directory.iterdir():
        hyperparameter = '-'.join(f.stem.split('_')[-4:])
        #print(hyperparameter)
        out = correctly_parse_log_path(f)

        if out is not None:
            result = Metrics(*out)
            results[hyperparameter] = result
            
            train_corrs.append(result.train_corr)
            test_corrs.append(result.test_corr)
            names.append(hyperparameter)
    
    if len(train_corrs) > 0:
        train_corrs = np.vstack(train_corrs)
        test_corrs = np.vstack(test_corrs)
        names = np.array(names)
        
        best_models = np.argmax(train_corrs, axis=0)
        best_hyperparameters = names[best_models]
        best_performances = test_corrs[best_models, np.arange(len(best_models))]
        
    # Select model with the lowest training loss in the final k
#     best_hyperparameter, best_hyperparameter_performance = sorted(results.items(), key=lambda x: x[1].train_loss[-1])[0]
#     best_hyperparameter, best_hyperparameter_performance = sorted(results.items(), key=lambda x: x[1].train_corr[-1])[-1]

#     return best_hyperparameter, best_hyperparameter_performance

#     return results

        return best_hyperparameters, best_performances
    else:
        return None

#logs_directory = Path("../output/210803_drug-baseline-models/run-logs")
logs_directory = Path("/Users/amelanidelahoz/Documents/TCRP/tcrp-reproduce/tcrp/output/210803_drug-baseline-models/run-logs/Nutlin-3a")
#logs_directory = Path()

all_paths = []
# for drug_directory in logs_directory.glob("*"): 
#     for tissue_directory in drug_directory.glob("*"):
#         if not any(tissue_directory.iterdir()):
#             drug = str(tissue_directory).split("/")[-2]
#             if drug not in empty:
#                 empty.append(drug)
#         else:
#             all_paths.append(tissue_directory)
for tissue_directory in logs_directory.glob("*"):
    all_paths.append(tissue_directory)
         
#results = map(select_hyperparameter, all_paths)
results = []
for path in all_paths:
    result = select_hyperparameter(path)
    if result is not None:
        results.append(result)
    

all_test_corrs = np.vstack([r[1] for r in results])

np.savez("tcrp_fewshot-test-correlations-corrected", all_test_corrs)

with open("tcrp_all_log_paths.pkl", "wb") as f: 
    pickle.dump(all_paths, f)
    
with open("tcrp_all_results.pkl", "wb") as f: 
    pickle.dump(results, f)
