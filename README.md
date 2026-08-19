# ViT CBM package

This codebase was used to produce results for my MSc thesis. It allows the user to fit a concept bottleneck model (CBM) on a dataset containing paitent WSI slides and concepts, with the ability to adjust various hyperparameters to understand their impact on model performance.


## Requirements

### Packages 
A full list of required packages can be found in ```requirements.txt```

### Directory structure
As well as these scripts, the user will need to provide a dataset of whole slide images (WSIs) and patient level concepts. These should be stored as follows for some master_dir

```text
master_dir/
├── vit_cbm/
├── data/
     ├── {feature_file}.csv
     ├── bags_np
            ├── {patient_tiles}
                ├── id1.npy
                ├── ...
├── models/
    ├── {model}.bin
```

Specifically, the user needs to provide 3 parts of the above:
1. ```{feature_file}.csv``` - 1 row per patient including (time,event) survivial data and any concepts to be used in the CBM. An example for TCGA-BRCA is generated in data/get_patient_info.ipynb
2. ```{patient_tiles}``` - a directory of .npy files containing a randomly sampled set of tiles per patient (1 .npy per patient)
3. Pretrained model weights for a baseline ViT ```{model}.bin```. Our study used UNI - available from https://huggingface.co/MahmoodLab/UNI

### Compute

Training the ViT models typically requires a GPU and this codebase assumes there is one available. For MSc thesis experiment results, an NVDIA A100 was used.

## Obtaining experiment results

Once the requirements are setup, it is straightfoward to perform experiement runs. The configs directory contains a template ```config.yaml``` file which should be duplicated for each experiment run and the values changed as required.

Once the config file is finalised, run ```xp.sh``` bash file with the required experiment names (config files without .yaml). By default this will perform 2 runs for each experiment but this can be configured in the batch file. Note, bash files need to be in ```master_dir``` so vit_cbm imports as a package

Once an experiment is complete, the results will be written to /results with a sub-directory per run. This subdirectory will contain:

- ```CONFIG.json```: The experiement parameters
- ```folds.json```: The allocation of paitents into train and test set
- ```results.json```: The model outputs (per training fold)
- ```run.log```: Console output from the training run

## Exporting experiment results

In order to export experiment runs into a single file for analysis, the ```results.py``` file can be run. This will create a single csv file with all metrics from all experiments runs stored in ```/results``` (1 row per fold). This should be stored in ```/output```

## Reviewing experiment results

With a single ```results.csv``` file, analysis and charts can be created in ```/output/review_results.ipynb```

