# This file is called by the xp.sh file to perform a single experiment run

# args and logging #########################################################
import argparse
import yaml
import json
import os
import logging
from vit_cbm.utils.utils import create_logs
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

base_path = '/scratch/prj/ccc_vit_finetuning/' # can update to match master_dir

config_path = os.path.join(base_path, 'vit_cbm','configs', args.config)
run_folder = os.path.join(base_path, 'results', args.run_name)

create_logs(run_folder)
logger = logging.getLogger(__name__)  # will be "__main__"
logger.info(f'starting test with run name: {args.run_name}')


# load the config file for this experiment
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# save the config file to the run folder for reference
with open(os.path.join(run_folder, 'CONFIG.json'), 'w') as f:
    json.dump(config, f, indent=2, separators=(', ', ': '))

logger.info('config loaded')

batch_size = config['training']['batch_size']
epochs = config['training']['epochs']
feature_file = config['data']['feature_file']
np_bags_dir = config['data']['np_bags_dir']

logger.info('STARTING TEST WITH:')
logger.info(f'BATCH SIZE {batch_size}')
logger.info(f'EPOCHS {epochs}')

#### imports #########################################################
logger.info('importing')
import pandas as pd
from torch.utils.data import DataLoader
import json
from sklearn.model_selection import train_test_split

# from vit_cbm.utils import get_sweep_id, build_trial_name
from vit_cbm.data.dataset import WSITileDataset
from vit_cbm.data.tile_list import make_tile_list
from vit_cbm.training.train import train_model, predict, measure_performance
from vit_cbm.utils.utils import kfolds

logger.info(f'imports complete')


# ## DATA SETUP ###############################################
logger.info(f'making datasets')

features = pd.read_csv(feature_file)

# limit patient limit if specified
if config['data']['paitent_limit']:
    features = features.head(config['data']['paitent_limit']).copy()

# get list of patients
ids_with_features = features['ID'].tolist()
logger.info(f'paitent data for {len(ids_with_features)} patients')

# get a list of all the image bags stored
logger.info(f'np_bags_dir: {np_bags_dir}')
bag_mainfest = os.path.join(np_bags_dir, 'manifest.json')
with open(bag_mainfest, 'r') as f:
    bag_manifest = json.load(f)

# ensure that all patients have a bag
ids_with_np_bags = [x[:12] for x in list(bag_manifest.keys())]

# if we cant resample, make sure all paitents can make a full bag
if not config['data']['resample_small_bags']:
    min_tiles =config['data']['bag_size']
    ids_with_np_bags = [x for x in ids_with_np_bags if bag_manifest[x]['total_tiles'] >= min_tiles]

logger.info(f'np bags for {len(ids_with_np_bags)} patients')

# check for paitents with data AND bags
common_ids = [x for x in ids_with_features if x in ids_with_np_bags]
logger.info(f'{len(common_ids)} patients with data AND bags')

assert len(common_ids)>0, f'not enough patients with data AND bags: {len(common_ids)}'

# create K folds as required for the experiment
raw_folds = kfolds(common_ids,Kfolds=config['training']['Kfolds'])

if config['training']['first_fold_only']:
    folds = [raw_folds[0]]
else:
    folds = raw_folds

# log the folds for reference
with open(os.path.join(run_folder, 'folds.json'), 'w') as f:
    json.dump(folds, f)

fold_results = {}

# ## Modelling ###############################################

for fold in folds:
    logger.info(f'training fold {fold["fold"]}')

    train_ids = fold['train']
    val_ids = fold['val']
    
    train_feature_file = features[features['ID'].isin(train_ids)].copy()
    val_feature_file = features[features['ID'].isin(val_ids)].copy()
    
    logger.info(f'train set size: {len(train_feature_file)}')
    logger.info(f'val set size: {len(val_feature_file)}')

    train_dataset = WSITileDataset(train_feature_file,config=config)
    val_dataset = WSITileDataset(val_feature_file,config=config)

    cpu_count = int(os.cpu_count())
    logger.info(f'cpu_count: {cpu_count}')
    num_workers =2
    logger.info(f'num_workers: {num_workers}')

    # create dataloaders for training
    train_dloader = DataLoader(
        train_dataset
        , batch_size=batch_size
        , num_workers=num_workers
        , drop_last=True
        , shuffle=True
        , persistent_workers=True
        , prefetch_factor=num_workers
        , pin_memory=True
        )

    val_dloader = DataLoader(
        val_dataset
        , batch_size=batch_size
        , num_workers=num_workers
        , drop_last=False
        , shuffle=False
        , persistent_workers=True
        , prefetch_factor=num_workers
        , pin_memory=True)

    logger.info(f'completed building dataloaders')

    logger.info(f'training starts')

    # init and train a model
    model, trainer = train_model(
        train_dloader
        ,val_dloader
        ,config
        ,epochs=epochs
        )

    logger.info(f'training complete')

    logger.info(f'predicting on the validation set')

    # use trained model to make predictions
    y_test, c_test, task_logits, concept_logits = predict(model, val_dloader) 

    logger.info(f'measuring performance')
    
    # measure the quality of these predictions
    results = measure_performance(y_test, task_logits, c_test, concept_logits, config)
    
    logger.info(f'results: {results}')

    # log results to results file
    fold_results[f'fold_{fold["fold"]}']=results

    with open(os.path.join(run_folder, 'results.json'), 'w') as f:
         json.dump(fold_results, f, indent=2, separators=(', ', ': '))


logger.info(f'done')
