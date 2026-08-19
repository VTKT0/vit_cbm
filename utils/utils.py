from datetime import datetime
from sklearn.model_selection import KFold
import os
import logging
import json
import pandas as pd

def create_logs(log_dir):

    '''start a log file'''

    # Ensure log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # If the log file does not exist then create it 
    log_file = os.path.join(log_dir, 'run.log')
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            pass

    logging.basicConfig(
        level=logging.INFO
        , format="%(asctime)s | %(levelname)s | %(message)s"
        , handlers=[
            logging.StreamHandler()             
            ,logging.FileHandler(log_file)         
        ]
    )


def kfolds(ids,Kfolds=5):

    'takes a list of paitent ids and returns K folds of train and val ids'

    kf = KFold(
    n_splits=Kfolds
    ,shuffle=True      
    ,random_state=42    
    )

    cv_splits = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(ids)):
        train_ids = [ids[i] for i in train_idx]
        val_ids = [ids[i] for i in val_idx]
        
        cv_splits.append({
            "fold": fold
            ,"train": train_ids
            , "val": val_ids
        })
    return cv_splits


### HELPERS FOR RESULTS


def get_subdirs(dir_path):
    return [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]


def get_config_info(dir_path):
    '''returns the info from config file'''

    log_path = '/scratch/prj/ccc_vit_finetuning/results'


    config_file = os.path.join(log_path, dir_path , 'CONFIG.json')


    with open(config_file, 'r') as f:
        config = json.load(f)

    # this was added as a subsequent parameter so missing in early runs
    try:
        lr_scheduler = config['training']['lr_scheduler']
    except:
        lr_scheduler = 'constant'

    return {

        'xp': dir_path.split('/')[0],
        'run': dir_path.split('/')[1],

        'feature_file': config['data']['feature_file'],
        'paitent_limit': config['data']['paitent_limit'],
        'np_bags_dir': config['data']['np_bags_dir'],
        'bag_size': config['data']['bag_size'],
        'resample_small_bags': config['data']['resample_small_bags'],
        'static_bags': config['data']['static_bags'],
     
        'concept_states': config['concepts']['concept_states'],
        'no_concepts': config['concepts']['no_concepts'],
        'cpt_ids': config['concepts']['cpt_ids'],

        'batch_size': config['training']['batch_size'],
        'epochs': config['training']['epochs'],
        'learning_rate': config['training']['learning_rate'],
        'vit_learning_rate': config['training']['vit_learning_rate'] if 'vit_learning_rate' in config['training'] else config['training']['learning_rate'],
        'dropout': config['training']['dropout'],
        'weight_decay': config['training']['weight_decay'],
        'lr_scheduler': lr_scheduler,

        'model_name': config['model']['vit_model_name'],
        'fine_tune': config['model']['fine_tune'],
        'grad_checkpointing': config['model']['grad_checkpointing'],
        'pooling': config['model']['pooling'],
        'concept_loss_weight': config['model']['concept_loss_weight'],
        'concept_heads': config['model']['concept_heads'],
        'task_head': config['model']['task_head']
    }

def get_results(dir_path):
    '''returns the info from results file as a df'''

    log_path = '/scratch/prj/ccc_vit_finetuning/results'

    results_file = os.path.join(log_path, dir_path, 'results.json')
    assert os.path.isfile(results_file), f'{results_file} not found'


    # Load json
    with open(results_file, "r") as f:
        data = json.load(f)

    rows = []

    for fold_name, fold_data in data.items():
        row = {
            "fold": fold_name,
            "task_c_index": fold_data["task"]["c_index"],
        }

        for concept_name, concept_data in fold_data["concepts"].items():

            # Accuracy
            row[f"{concept_name}_accuracy"] = concept_data.get("accuracy")

            # Macro avg metrics
            macro = concept_data.get("macro avg", {})
            row[f"{concept_name}_macro_precision"] = macro.get("precision")
            row[f"{concept_name}_macro_recall"] = macro.get("recall")
            row[f"{concept_name}_macro_f1"] = macro.get("f1-score")

            # Weighted avg metrics
            weighted = concept_data.get("weighted avg", {})
            row[f"{concept_name}_weighted_precision"] = weighted.get("precision")
            row[f"{concept_name}_weighted_recall"] = weighted.get("recall")
            row[f"{concept_name}_weighted_f1"] = weighted.get("f1-score")

            # AUC
            auc = concept_data.get("auc", {})
            row[f"{concept_name}_auc_ovo"] = auc.get("ovo")

            # per-class metrics
            for cls, cls_metrics in concept_data.items():
                if cls in ["accuracy", "macro avg", "weighted avg", "auc"]:
                    continue

                row[f"{concept_name}_class{cls}_precision"] = cls_metrics.get("precision")
                row[f"{concept_name}_class{cls}_recall"] = cls_metrics.get("recall")
                row[f"{concept_name}_class{cls}_f1"] = cls_metrics.get("f1-score")
                row[f"{concept_name}_class{cls}_support"] = cls_metrics.get("support")

        rows.append(row)

    df = pd.DataFrame(rows)

    return df

