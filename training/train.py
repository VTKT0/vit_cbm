## contains helper functions for training model, validating and measuring performance

import logging
logger = logging.getLogger(__name__)


import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import torch
import numpy as np
from sksurv.metrics import concordance_index_censored
from scipy.special import softmax

from sklearn.metrics import (
    classification_report,
    roc_auc_score
)

from vit_cbm.data.dataset import WSITileDataset
from vit_cbm.models.model import CBM_VIT
from vit_cbm.training.customer_trainer import CustomTrainer

# MODEL #########################################################

def train_model(
    train_dloader
    ,val_dloader
    ,config = None
    ,epochs = 1
    ):

    """
    Trains a model using the provided data loaders.
    Returns (model, trainer) where trainer has info on training
    """
    model = CBM_VIT(
        concept_states=config['concepts']['concept_states']
        ,no_concepts=config['concepts']['no_concepts']
        ,vit_model_name=config['model']['vit_model_name']
        ,optimizer='adam'
        ,learning_rate=config['training']['learning_rate']
        ,vit_learning_rate=config['training']['vit_learning_rate'] if 'vit_learning_rate' in config['training'] else None
        ,dropout=config['training']['dropout']
        ,weight_decay=config['training']['weight_decay']
        ,fine_tune=config['model']['fine_tune']
        ,grad_checkpointing=config['model']['grad_checkpointing']
        ,pooling=config['model']['pooling']
        ,concept_heads=config['model']['concept_heads']
        ,task_head=config['model']['task_head']
        ,concept_loss_weight=config['model']['concept_loss_weight']
    )
    
    logger.info(f'model initialized')

    # use my coded from sratch trainer class
    trainer = CustomTrainer(
        max_epochs=epochs
        ,device="cuda"
        ,lr_scheduler=config['training']['lr_scheduler']
    )

    trainer.fit(model, train_dloader, val_dloader)

    return model, trainer
   

def predict(model, val_dloader):

    '''
    Predicts the hazards for the validation set.
    returns: y_probs (model hazards), y_test (censor data)
    '''

    # get the ground truth labels
    event_test, survtime_test , c_test, patient_ids_list = [], [], [], []
    
    # will get OOM issues here if not enough CPU memory (not GPU)
    for batch in val_dloader:
        
        (event, survtime), c, wsi_id = batch[1], batch[2], batch[3]
        c_test.append(c)
        event_test.append(event)
        survtime_test.append(survtime)

   # glue the batches together

    c_test = np.concatenate(c_test, axis=0) #[paitent,concept]
    event_test = np.concatenate(event_test, axis=0) #[paitent]
    survtime_test = np.concatenate(survtime_test, axis=0) #[paitent]
    #patient_ids_list = np.concatenate(patient_ids_list, axis=0) #[paitent]
    y_test = np.array([(int(event_test[i]), int(survtime_test[i])) for i in range(len(event_test))])


    # init a new training class to help with predicitions
    trainer_pred = CustomTrainer(
        max_epochs=1
        ,device="cuda"
    )

    batch_results = trainer_pred.predict(model, val_dloader)

    task_logits = []
    concept_logits = []

    for batch_result in batch_results:
        task_logits.append(batch_result[1].detach().cpu().numpy())
        concept_logits.append(batch_result[0].detach().cpu().numpy())

    task_logits = np.concatenate(task_logits, axis=0)
    concept_logits = np.concatenate(concept_logits, axis=0)


    return tuple([y_test, c_test, task_logits, concept_logits])



def measure_performance(
    y_test
    ,y_probs
    ,c_test
    ,c_probs
    ,config
):
    results = {}
    

    ## task metrics
    logger.info(f'y_test: {y_test.shape}')
    logger.info(f'y_probs: {y_probs.shape}')
    logger.info(f'c_test: {c_test.shape}')
    logger.info(f'c_probs: {c_probs.shape}')


    # true values
    events = [bool(y[0]) for y in y_test]
    survtimes = [y[1] for y in y_test]


    # log overall C-index for the model
    try:    
        c_index, _, _, _, _ = concordance_index_censored(
        events, survtimes, y_probs.squeeze()
        )
    except Exception as e:
        logger.error(f'Error calculating c-index: {e}')
        c_index = 0
        logger.error(f'Setting c-index to 0')

    results['task'] = {'c_index': float(c_index)}


    ## concept metrics
    concept_states = config['concepts']['concept_states']
    idx=0
    concept_reports = {}
    

    # for each concept, extract the logits and calc metrics
    for n, n_states in enumerate(concept_states):


        c_logits_concept = c_probs[:, idx:idx+n_states]
        c_probs_concept = softmax(c_logits_concept, axis=1)
        c_pred_concept = np.argmax(c_probs_concept, axis=1)
        c_test_concept = c_test[:, n]

        logger.info(f'c_test_concept: {c_test_concept.shape}')
        logger.info(f'c_pred_concept: {c_pred_concept.shape}')

        class_report_concept = classification_report(
            c_test_concept
            , c_pred_concept
            , output_dict=True
            , zero_division=np.nan # for precision, recall, f1, etc if no obvs
            ,labels=list(range(n_states)) # to handle cases where c_test doesnt have every class in - so sklearn assumes there are fewer classes
        )


        auc_ovo = roc_auc_score(c_test_concept, c_probs_concept
        , multi_class='ovo' # one vs rest and average
        ,labels=list(range(n_states)) # to handle cases where c_test doesnt have every class in - so sklearn assumes there are fewer classes
        )

        class_report_concept['auc'] = {'ovo': auc_ovo.item()}

        # adda confusion matrix
        confusion = {}

        for i in range(n_states):
            confusion[f'true_{i}'] = {}
            for j in range(n_states): 
                confusion[f'true_{i}'][f'pred_{j}'] = np.sum(c_pred_concept[c_test_concept==i]==j).item()

        class_report_concept['confusion'] = confusion

        concept_reports[f'concept_{n}'] = class_report_concept

        idx += n_states

    # add all concepts
    results['concepts'] = concept_reports


    return results
    





